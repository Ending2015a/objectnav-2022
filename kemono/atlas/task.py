# --- built in ---
import os
import copy
import glob
from typing import (
  Any,
  Dict,
  List,
  Tuple,
  Union,
  Optional
)
# --- 3rd party ---
import gym
import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import rlchemy
import einops
import omegaconf
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
# --- my module ---
import kemono
from kemono.atlas import model as atlas_model

EP_GLOB_PATTERN = '**/ep-*/'
SAMPLE_GLOB_PATTERN = '**/sample_*.npz'

def glob_filenames(
  root_path: str,
  pattern: str
) -> List[str]:
  root_path = os.path.abspath(root_path)
  glob_path = os.path.join(root_path, pattern)
  # glob abspaths
  abspaths = glob.glob(glob_path, recursive=True)
  # convert to relpaths
  relpaths = [os.path.relpath(path, start=root_path) for path in abspaths]
  relpaths = sorted(relpaths)
  return relpaths

class AtlasDataset(Dataset):
  def __init__(
    self,
    root_path,
    n_slices,
    chart_name: str = None,
    shuffle: bool = False,
    max_length: Optional[int] = None,
    get_chart_gt: bool = False,
    img_size: Optional[Tuple[int, int]] = None
  ):
    """_summary_

    Args:
      root_path (_type_): _description_
      n_slices (_type_): _description_
      chart_name (str, optional): name of the chart to use. If None, use
        ChartData.chart. Defaults to None.
      shuffle (bool, optional): _description_. Defaults to False.
      max_length (Optional[int], optional): _description_. Defaults to None.
      get_chart_gt (bool, optional): _description_. Defaults to False.
      img_size (Optional[Tuple[int, int]], optional): _description_. Defaults to None.
    """    

    self.root_path = root_path
    self.n_slices = n_slices
    self.chart_name = chart_name
    self.shuffle = shuffle
    self.max_length = max_length
    self.get_chart_gt = get_chart_gt
    self.img_size = img_size

    self.sample_list = glob_filenames(root_path, SAMPLE_GLOB_PATTERN)
    self.id_list = np.arange(len(self.sample_list))
    if shuffle:
      np.random.shuffle(self.id_list)
    if max_length is not None:
      self.id_list = self.id_list[:max_length]

  def __len__(self) -> int:
    return len(self.id_list)

  @torch.no_grad()
  def __getitem__(self, idx: int):
    sample_id = self.id_list[idx]
    return self._get_sample(sample_id)

  def _get_sample(self, sample_id: int):
    sample_path = os.path.join(self.root_path, self.sample_list[sample_id])
    gsm_data = kemono.atlas.ChartData.load(sample_path)
    objectgoal = np.asarray(gsm_data.objectgoal, dtype=np.int64)
    chart = np.asarray(gsm_data.chart, dtype=np.float32)
    if len(chart.shape) < 3:
      chart = np.expand_dims(chart, axis=0) # (c, h, w)
    # get suer specified charts for inputs
    if self.chart_name is not None:
      chart_inp = gsm_data.maps.get(self.chart_name, chart.copy())
    chart_inp = np.asarray(chart, dtype=np.float32)
    if len(chart_inp.shape) < 3:
      chart_inp = np.expand_dims(chart, axis=0) # (c, h, w)
    points = np.asarray(gsm_data.points, dtype=np.float32)
    distances = np.asarray(gsm_data.distances, dtype=np.float32)
    gradients = np.asarray(gsm_data.gradients, dtype=np.float32)

    assert len(points) == len(distances)
    assert len(distances) == len(gradients)
    total_samples = len(points)

    # random draw samples
    if total_samples >= self.n_slices:
      inds = np.random.choice(np.arange(total_samples), size=(self.n_slices,))
    else:
      inds = np.random.randint(total_samples, size=(self.n_slices,))
    objectgoals = einops.repeat(objectgoal, ' -> s', s=self.n_slices)
    # Note that here we assume the same chart is shared across all slices
    # so we dont repeat slices
    charts = einops.repeat(chart, '... -> 1 ...')
    chart_inps = einops.repeat(chart_inp, '... -> 1 ...')
    points = np.asarray(points[inds], dtype=np.float32)
    distances = np.asarray(distances[inds], dtype=np.float32)
    gradients = np.asarray(gradients[inds], dtype=np.float32)

    if self.img_size is not None:
      charts = kemono.atlas.utils.resize_image(
        charts,
        size = self.img_size,
        mode = 'nearest'
      )
      chart_inps = kemono.atlas.utils.resize_image(
        chart_inps,
        size = self.img_size,
        mode = 'nearest'
      )

    # prepare data
    data = {
      'objectgoal': objectgoals, # (s,)
      'chart': charts, # (1, 1, h, w)
      'chart_inp': chart_inps, # (1, c, h, w)
      'point': points, # (s, 2)
      'distance': distances, # (s,)
      'gradient': gradients # (s, 2)
    }
    if self.get_chart_gt:
      # (h, w, 3)
      chart_gt = np.asarray(gsm_data.chart_gt, dtype=np.float32)
      if self.img_size is not None:
        chart_gt = einops.rearrange(chart_gt, 'h w c -> 1 c h w')
        chart_gt = kemono.atlas.utils.resize_image(
          chart_gt,
          size = self.img_size,
          mode = 'nearest'
        )
        chart_gt = einops.rearrange(chart_gt, '1 c h w -> h w c')
      data['chart_gt'] = chart_gt # (h, w, 3)
    return data


class AtlasTask(pl.LightningModule):
  def __init__(
    self,
    config,
    inference_only: bool = False,
    input_space: Optional[gym.spaces.Space] = None,
  ):
    super().__init__()
    OmegaConf.resolve(config)
    # ---
    self.trainset = None
    self.valset = None
    self.predset = None
    self.model = None
    self.input_space = None

    self.config = config
    if not inference_only:
      self.setup_dataset()
    self.setup_input(input_space)
    self.setup_model()
    # this will be saved to hyperparameters
    input_space = self.input_space
    self.save_hyperparameters(ignore=["inference_only"])
    # create vector color palatte
    ph = np.linspace(0, 2*np.pi, 13)
    u = np.cos(ph)
    v = np.sin(ph)
    colors = np.arctan2(v, u)
    self.color_norm = Normalize()
    self.color_norm.autoscale(colors)

  def setup_input(self, input_space: Optional[gym.spaces.Space]=None):
    self.input_space = input_space

    if self.trainset is not None:
      data = self.trainset[0]
      
      point = gym.spaces.Box(low=-np.inf, high=np.inf,
        shape=data['point'].shape[1:], dtype=np.float32)
      objectgoal = gym.spaces.Discrete(self.config.num_classes)
      chart = gym.spaces.Box(low=-np.inf, high=np.inf,
        shape=data['chart_inp'].shape[1:], dtype=np.float32)
      self.input_space = gym.spaces.Tuple((point, objectgoal, chart))

  def setup_model(self):
    net = atlas_model.ToyNet(
      **self.config.toy_net
    )
    self.model = atlas_model.Energy(net)
    # forward dummy tensors
    dummy = rlchemy.utils.input_tensor(self.input_space)
    self.model(*dummy)
  
  def setup_dataset(self):
    self.trainset = AtlasDataset(**self.config.train_dataset)
    self.valset = AtlasDataset(**self.config.eval_dataset)
    self.predset = AtlasDataset(**self.config.pred_dataset)
  
  def configure_optimizers(self):
    optim = torch.optim.Adam(
      self.model.parameters(),
      lr = self.config.optimizer.learning_rate
    )
    return optim

  def train_dataloader(self) -> DataLoader:
    return DataLoader(
      self.trainset,
      batch_size = self.config.batch_size,
      num_workers = self.config.num_workers,
      shuffle = True
    )

  def val_dataloader(self) -> DataLoader:
    return DataLoader(
      self.valset,
      batch_size = self.config.batch_size,
      num_workers = self.config.num_workers,
      shuffle = False
    )

  def _forward_preprocess(
    self,
    x: torch.Tensor,
    cond: torch.Tensor,
    chart: Optional[torch.Tensor],
    chart_cache: Optional[torch.Tensor] = None
  ):
    x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
    cond = torch.as_tensor(cond, dtype=torch.int64, device=self.device)
    # expand batch dim
    one_sample = len(x.shape) == 1
    if one_sample:
      x = torch.unsqueeze(x, dim=0)
      cond = torch.unsqueeze(cond, dim=0)
    
    if chart is not None:
      chart = torch.as_tensor(chart, dtype=torch.float32, device=self.device)
      # expand channel dim
      if len(chart.shape) == 2:
        chart = torch.unsqueeze(chart, dim=0) # (1, h, w)
      # expand batch dim
      if one_sample:
        chart = torch.unsqueeze(chart, dim=0) # (1, 1, h, w)
      # resize chart to input size
      chart_size = self.config.chart_size
      batch_dims = chart.shape[:-3]
      chart = chart.reshape((-1, *chart.shape[-3:]))
      chart = nn.functional.interpolate(
        chart,
        size = (chart_size, chart_size),
        mode = 'nearest'
      )
      chart = chart.reshape((*batch_dims, *chart.shape[-3:]))
    if chart_cache is not None:
      chart_cache = torch.as_tensor(chart_cache, dtype=torch.float32,
        device=self.device)
    return x, cond, chart, chart_cache

  def forward(
    self,
    x: torch.Tensor,
    cond: torch.Tensor,
    chart: Optional[torch.Tensor],
    chart_cache: Optional[torch.Tensor] = None,
    get_score: bool = False,
  ) -> Union[
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
  ]:
    """Forward

    Args:
        x (torch.Tensor): _description_
        cond (torch.Tensor): _description_
        chart (Optional[torch.Tensor]): _description_
        chart_cache (Optional[torch.Tensor], optional): _description_. Defaults to None.
        get_score (bool, optional): _description_. Defaults to False.

    Returns:
        torch.Tensor: energy
        torch.Tensor, Optional: score if get_score is True.
        torch.Tensor: chart cache
    """
    res = self._forward_preprocess(x, cond, chart, chart_cache)
    return self.model(*res, get_score=get_score)

  def predict(
    self,
    x: torch.Tensor,
    cond: torch.Tensor,
    chart: Optional[torch.Tensor],
    chart_cache: Optional[torch.Tensor] = None,
    get_score: bool = False
  ):
    one_sample = len(x.shape) == 1
    *outs, cache = self(x, cond, chart, chart_cache, get_score=get_score)
    outs = rlchemy.utils.nested_to_numpy(tuple(outs), dtype=np.float32)
    if one_sample:
      outs = rlchemy.utils.map_nested(outs, op=lambda t: t[0])
    return (*outs, cache)
  
  def training_step(
    self,
    batch: Any,
    batch_idx: int
  ) -> torch.Tensor:
    x = batch['point'] # (b, s, 2)
    cond = batch['objectgoal'] # (b, s)
    chart = batch['chart_inp'] # (b, 1, c, h, w)
    gm = batch['distance'].unsqueeze(-1) # (b, s, 1)
    dgm = batch['gradient']# (b, s, 2)
    # predict score
    _, s, _ = self(x, cond, chart, get_score=True)
    # compute geodesic score
    gt_s = gm * dgm
    sigma = self.config.sigma
    loss = torch.sum((s + gt_s/(sigma**2)) ** 2, dim=-1)
    loss = loss.mean() * 0.5

    self.log(
      "train/loss",
      loss.item(),
      on_step = True,
      on_epoch = True,
      sync_dist = True,
      prog_bar = True
    )

    return loss

  def validation_step(
    self,
    batch: Any,
    batch_idx: int
  ) -> torch.Tensor:
    torch.set_grad_enabled(True)
    x = batch['point'] # (b, s, 2)
    cond = batch['objectgoal'] # (b, s)
    chart = batch['chart_inp'] # (b, s, c, h, w)
    gm = batch['distance'].unsqueeze(-1) # (b, s, 1)
    dgm = batch['gradient'] # (b, s, 2)
    # predict score
    _, s, _ = self(x, cond, chart, get_score=True)
    # compute geodesic score
    gt_s = gm * dgm
    sigma = self.config.sigma
    loss = torch.sum((s + gt_s/(sigma**2)) ** 2, dim=-1)
    loss = loss.mean() * 0.5

    self.log(
      "val/loss",
      loss.item(),
      on_step = False,
      on_epoch = True,
      sync_dist = True,
      prog_bar = True
    )

    return loss

  def _plot_energy(self, ax, energy_map, title=None):
    cmap = copy.copy(plt.cm.viridis)
    cmap.set_bad(color='black')
    ax.imshow(energy_map, cmap=cmap, interpolation='nearest')
    if title is not None:
      ax.set_title(title, fontsize='xx-large')

  def _plot_score(self, ax, score_map, chart, title=None):
    score_map[np.isinf(score_map)] = 0
    score_map[np.isnan(score_map)] = 0
    height, width = score_map.shape[:-1]
    mesh = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    mesh = np.stack(mesh, axis=-1).astype(np.float32)
    mesh = mesh.reshape((-1, 2))
    score_map = score_map.reshape((-1, 2))
    mask = np.logical_and(score_map[...,0]==0, score_map[...,1]==0)
    mesh = mesh[~mask]
    score_map = score_map[~mask]
    if len(chart.shape) < 3 or chart.shape[-1] == 1:
      ax.imshow(chart, cmap='hot', interpolation='nearest')
    else:
      ax.imshow(chart)
    theta = np.arctan2(score_map[...,1], score_map[...,0])

    # plot vector field
    ax.quiver(
      mesh[...,1], mesh[...,0],
      score_map[...,0], score_map[...,1],
      color = plt.cm.hsv(self.color_norm(theta)),
      angles = 'xy',
      width = 0.001
    )
    if title is not None:
      ax.set_title(title, fontsize='xx-large')

  def _preview_predictions(self, pred_idx: int):
    """Plot samples"""
    torch.set_grad_enabled(True)
    data = self.predset[pred_idx]
    cond = data['objectgoal'] # (1,)
    chart = data['chart'] # (1, 1, h, w)
    chart_inp = data['chart_inp'] # (1, c, h, w)
    chart_gt = data['chart_gt'] # (h, w, 3)
    height, width = chart_gt.shape[:-1]
    rsize = np.asarray((width, height), dtype=np.float32)
    energy_map = np.full((height, width), np.inf, dtype=np.float32)
    score_map = np.zeros((height, width, 2), dtype=np.float32)
    # get image cache
    dummy_x, dummy_cond, _ = rlchemy.utils.input_tensor(self.input_space)
    # cache: (1, vec)
    _, cache = self.predict(dummy_x, dummy_cond, chart_inp)

    for h in range(height):
      coords = []
      xs = []
      for w in range(width):
        if chart[0, 0, h, w]:
          x = np.asarray((w, h), dtype=np.float32)
          x = x / rsize * 2.0 - 1.0
          xs.append(x)
          coords.append(w)
      if len(coords) == 0:
        continue

      x_ = torch.tensor(xs, dtype=torch.float32, device=self.device)
      batch_size = x_.shape[0]
      cond_ = einops.repeat(cond, '1 -> b', b=batch_size)
      cache_ = einops.repeat(cache, '1 ... -> b ...', b=batch_size)
      e_, s_, _ = self.predict(x_, cond_, None, chart_cache=cache_, get_score=True)
      # _e: (b, 1)
      # _s: (b, 2)
      e = e_.squeeze(-1)
      for c, e, s in zip(coords, e_, s_):
        energy_map[h, c] = e
        score_map[h, c] = s

    fig, axs = plt.subplots(
      figsize=(12, 12),
      ncols=2, nrows=2,
      dpi=150,
      sharex = True,
      sharey = True
    )

    dist_map_gt = chart_gt[..., 0:1]
    grad_map_gt = chart_gt[..., 1:]
    # calculate ground truth score
    energy_map_gt = dist_map_gt[...,0] ** 2
    score_map_gt = -dist_map_gt * grad_map_gt
    self._plot_energy(
      axs[0][0],
      energy_map_gt,
      title = 'Ground truth energy'
    )
    self._plot_score(
      axs[0][1],
      score_map_gt,
      chart[0, 0],
      title = 'Ground truth score'
    )
    self._plot_energy(
      axs[1][0],
      energy_map,
      title = 'Predicted energy'
    )
    self._plot_score(
      axs[1][1],
      score_map,
      einops.rearrange(chart_inp[0], 'c h w -> h w c'),
      title = 'Predicted score'
    )
    plt.tight_layout()
    path = os.path.join(
      self.logger.log_dir,
      f'predictions/epoch_{self.current_epoch}',
      f'sample_{pred_idx}.png'
    )
    rlchemy.utils.safe_makedirs(filepath=path)
    fig.savefig(path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close('all')

  def on_save_checkpoint(self, checkpoint):
    if self.trainer.is_global_zero:
      if self.current_epoch % self.config.vis_freq == 0:
        num_samples = len(self.predset)
        with kemono.atlas.utils.evaluate(self):
          for n in range(num_samples):
            self._preview_predictions(n)
