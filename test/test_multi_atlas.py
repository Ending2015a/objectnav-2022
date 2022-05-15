# --- built in ---
import os
import glob
from typing import (
  Any,
  Dict,
  List,
  Optional
)
# --- 3rd party ---
import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import omegaconf
from omegaconf import OmegaConf
import einops
import rlchemy
import matplotlib.pyplot as plt
# --- my module ---
import kemono

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

class GsmDataset(Dataset):
  def __init__(
    self,
    root_path,
    n_slices,
    shuffle: bool = False,
    max_length: Optional[int] = None,
    get_chart_gt: bool = False,
  ):
    self.root_path = root_path
    self.n_slices = n_slices
    self.shuffle = shuffle
    self.max_length = max_length
    self.get_chart_gt = get_chart_gt

    self.sample_list = glob_filenames(root_path, SAMPLE_GLOB_PATTERN)
    self.id_list = np.arange(len(self.sample_list))
    if shuffle:
      np.random.shuffle(self.id_list)
    if max_length is not None:
      self.id_list = self.id_list[:max_length]

  def __len__(self) -> int:
    return len(self.sample_list)

  @torch.no_grad()
  def __getitem__(self, idx: int):
    sample_id = self.id_list[idx]
    return self._get_sample(sample_id)

  def _get_sample(self, sample_id: int):
    sample_path = os.path.join(self.root_path, self.sample_list[sample_id])
    gsm_data = kemono.gsm.GsmData.load(sample_path)
    objectgoal = np.asarray(gsm_data.objectgoal, dtype=np.int64)
    chart = np.asarray(gsm_data.chart, dtype=np.float32)
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
    charts = einops.repeat(chart, 'h w -> s 1 h w', s=self.n_slices)
    points = np.asarray(points[inds], dtype=np.float32)
    distances = np.asarray(distances[inds], dtype=np.float32)
    gradients = np.asarray(gradients[inds], dtype=np.float32)
    # prepare data
    data = {
      'objectgoals': objectgoals, # (s,)
      'charts': charts, # (s, 1, h, w)
      'points': points, # (s, 2)
      'distances': distances, # (s,)
      'gradients': gradients # (s, 2)
    }
    if self.get_chart_gt:
      # (h, w, 3)
      chart_gt = np.asarray(gsm_data.chart_gt, dtype=np.float32)
      data['chart_gt'] = chart_gt
    return data

class Swish(nn.Module):
  def __init__(self, dim=-1):
    super().__init__()
    if dim > 0:
      self.beta = nn.Parameter(torch.ones((dim,)))
    else:
      self.beta = torch.ones((1,))

  def forward(self, x):
    if len(x.size()) == 2:
      return x * torch.sigmoid(self.beta[None, :] * x)
    else:
      return x * torch.sigmoid(self.beta[None, :, None, None] * x)

class NatureCnn(nn.Module):
  def __init__(
    self,
    shape, # (c, h, w)
    mlp_units = [512]
  ):
    super().__init__()
    self.mlp_units = mlp_units

    dim = shape[0]
    cnn = nn.Sequential(
      nn.Conv2d(dim, 32, 8, 4, padding=0),
      Swish(32),
      nn.Conv2d(32, 64, 4, 2, padding=0),
      Swish(64),
      nn.Conv2d(64, 64, 3, 1, padding=0),
      Swish(64),
      nn.Flatten(start_dim=-3, end_dim=-1)
    )
    dummy = torch.zeros((1, *shape), dtype=torch.float32)
    outputs = cnn(dummy).detach()
    mlp = MLP(outputs.shape[-1], 0, units=self.mlp_units)
    self.net = nn.Sequential(*cnn, mlp)
    self.output_dim = self.mlp_units[-1]
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.net(x)


class MLP(nn.Module):
  def __init__(
    self,
    input_dim = 3,
    output_dim = 1,
    units = [300, 300],
    swish = True,
    dropout = False
  ):
    super().__init__()
    layers = []
    in_dim = input_dim
    for out_dim in units:
      layers.extend([
        nn.Linear(in_dim, out_dim),
        Swish(out_dim) if swish else nn.LeakyReLU(0.01),
        nn.Dropout(.5) if dropout else nn.Identity()
      ])
      in_dim = out_dim
    if output_dim > 0:
      layers.append(nn.Linear(in_dim, output_dim))
      out_dim = output_dim

    self.net = nn.Sequential(*layers)
    self.output_dim = out_dim
  
  def forward(self, x):
    return self.net(x)

class ToyNet(nn.Module):
  def __init__(
    self,
    chart_shape,
    input_dim = 3,
    output_dim = 1,
    mlp_units = [300],
    fuse_units = [300]
  ):
    super().__init__()
    self.cnn = NatureCnn(chart_shape, [512])
    self.mlp = MLP(input_dim, 0, mlp_units)
    fuse_dim = self.cnn.output_dim + self.mlp.output_dim
    self.fuse = MLP(fuse_dim, output_dim, fuse_units)
    self.output_dim = output_dim

  def forward(
    self,
    x: torch.Tensor,
    chart: torch.Tensor
  ) -> torch.Tensor:
    x_inp = self.mlp(x)
    x_chart = self.cnn(chart)
    x = torch.cat((x_inp, x_chart), dim=-1)
    return self.fuse(x)

class Energy(nn.Module):
  def __init__(self, net: ToyNet, num_classes):
    """A simple energy model

    E = -logp(x)

    Args:
      net (nn.Module): An energy function, the output shape of
        the energy function should be (b, 1). The score is
        computed by grad(-E(x))
    """
    super().__init__()
    self.net = net
    self.num_classes = num_classes

  def forward(self, x, cond, chart, get_score=False):
    cond = cond.to(dtype=torch.int64)
    cond = nn.functional.one_hot(cond, self.num_classes)
    cond = cond.to(dtype=torch.float32)
    if get_score:
      cond = cond.requires_grad_()
      x = x.requires_grad_()
    inp = torch.cat((x, cond), dim=-1)
    e = self.net(inp, chart)
    if get_score:
      logp = -e.sum()
      s = torch.autograd.grad(logp, x, create_graph=True)[0]
      return e, s
    return e

  def score(self, x, cond, chart):
    cond = cond.to(dtype=torch.int64)
    cond = nn.functional.one_hot(cond, self.num_classes)
    cond = cond.to(dtype=torch.float32)
    cond = cond.requires_grad_()
    x = x.requires_grad_()
    inp = torch.cat((x, cond), dim=-1)
    logp = -self.net(inp, chart).sum()
    return torch.autograd.grad(logp, x, create_graph=True)[0]

  def save(self, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(self.state_dict(), path)

  def load(self, path):
    self.load_state_dict(torch.load(path))
    return self


class GsmTask(pl.LightningModule):
  def __init__(
    self,
    config
  ):
    super().__init__()
    config = OmegaConf.create(config)
    OmegaConf.resolve(config)
    self.save_hyperparameters("config")
    # ---
    self.config = config
    self.setup_model()
    self.setup_dataset()
  
  def setup_model(self):
    config = self.config.model
    net = ToyNet(
      chart_shape = (1, config.chart_size, config.chart_size),
      input_dim = config.input_dim,
      output_dim = 1,
      mlp_units = config.mlp_units,
      fuse_units = config.fuse_units
    )
    self.model = Energy(
      net = net,
      num_classes = config.n_goals
    )

  def setup_dataset(self):
    self.trainset = GsmDataset(**self.config.train_dataset)
    self.valset = GsmDataset(**self.config.eval_dataset)
    self.predset = GsmDataset(**self.config.pred_dataset)

  def configure_optimizers(self):
    optim = torch.optim.Adam(
      self.model.parameters(),
      lr = self.config.train.learning_rate
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

  def _forward_preprocess(self, x, cond, chart):
    x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
    cond = torch.as_tensor(cond, dtype=torch.int64, device=self.device)
    chart = torch.as_tensor(chart, dtype=torch.float32, device=self.device)
    # expand batch dim
    if len(chart.shape) == 2:
      chart = torch.unsqueeze(chart, dim=0) # (1, h, w)
    one_sample = len(chart.shape) == 3
    if one_sample:
      x = torch.unsqueeze(x, dim=0)
      cond = torch.unsqueeze(cond, dim=0)
      chart = torch.unsqueeze(chart, dim=0)
    # resize chart to input size
    chart_size = self.config.model.chart_size
    chart = nn.functional.interpolate(
      chart,
      size = (chart_size, chart_size),
      mode = 'nearest',
      align_corners = True
    )
    return x, cond, chart

  def forward(
    self,
    x,
    cond,
    chart,
    get_score = False,
  ):
    x, cond, chart = self._forward_preprocess(x, cond, chart)
    return self.model(x, cond, chart, get_score=get_score)

  def score(
    self,
    x,
    cond,
    chart
  ):
    x, cond, chart = self._forward_preprocess(x, cond, chart)
    return self.model.score(x, cond, chart)

  @torch.no_grad()
  def predict(
    self,
    x,
    cond,
    chart,
    get_score = False
  ) -> np.ndarray:
    orig_shape = chart.shape
    one_sample = (len(orig_shape) <= 3)
    outs = self(x, cond, chart, get_score=get_score)
    outs = rlchemy.utils.nested_to_numpy(outs, dtype=np.float32)
    if one_sample:
      outs = rlchemy.utils.map_nested(outs, op=lambda t: t[0])
    return outs

  def predict_score(
    self,
    x,
    cond,
    chart
  ) -> np.ndarray:
    orig_shape = chart.shape
    one_sample = (len(orig_shape) <= 3)
    out = self.score(x, cond, chart)
    out = rlchemy.utils.to_numpy(out).astype(np.float32)
    if one_sample:
      out = out[0]
    return out

  def training_step(
    self,
    batch: Any,
    batch_idx: int
  ) -> torch.Tensor:
    cond = batch['objectgoals'].flatten(0, 1) # (b*s,)
    chart = batch['charts'].flatten(0, 1) # (b*s, 1, h, w)
    x = batch['points'].flatten(0, 1) # (b*s, 2)
    gm = batch['distances'].flatten(0, 1).unsqueeze(-1) # (b*s, 1)
    dgm = batch['gradients'].flatten(0, 1) # (b*s, 2)
    # predict score
    s = self.score(x, cond, chart)
    # compute geodesic score
    gt_s = gm * dgm
    sigma = self.config.model.sigma
    loss = torch.norm(s + gt_s/(sigma**2), dim=-1) ** 2
    loss = loss.mean() / 2.

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
    cond = batch['objectgoals'].flatten(0, 1) # (b*s,)
    chart = batch['charts'].flatten(0, 1) # (b*s, 1, h, w)
    x = batch['points'].flatten(0, 1) # (b*s, 2)
    gm = batch['distances'].flatten(0, 1).unsqueeze(-1) # (b*s, 1)
    dgm = batch['gradients'].flatten(0, 1) # (b*s, 2)
    # predict score
    s = self.score(x, cond, chart)
    # compute geodesic score
    gt_s = gm * dgm
    sigma = self.config.model.sigma
    loss = torch.norm(s + gt_s/(sigma**2), dim=-1) ** 2
    loss = loss.mean() / 2.

    self.log(
      "val/loss",
      loss.item(),
      on_step = False,
      on_epoch = True,
      sync_dist = True,
      prog_bar = True
    )

    return loss

  def _preview_chart_gt(self, chart_gt):


  def _preview_predictions(self, pred_idx: int):
    """Plot samples"""
    data = self.predset[pred_idx]
    cond = data['objectgoals'] # (1,)
    chart = data['charts'] # (1, 1, h, w)
    chart_gt = data['chart_gt'] # (h, w, 3)
    height, width = chart_gt.shape[:-1]
    rsize = np.asarray((width, height), dtype=np.float32)
    energy_map = np.zeros((height, width), dtype=np.float32)
    score_map = np.zeros((height, width, 2), dtype=np.float32)
    for h in range(height):
      coords = []
      xs = []
      for w in range(width):
        if chart[h, w]:
          x = np.asarray((w, h), dtype=np.float32)
          x = x / rsize * 2.0 - 1.0
          xs.append(x)
          coords.append(w)
      if len(coords) == 0:
        continue
      _x = torch.tensor(xs, dtype=torch.float32, device=self.device)
      batch_size = _x.shape[0]
      _cond = einops.repeat(cond, '1 -> b', b=batch_size)
      _chart = einops.repeat(cond, '1 ... -> b ...', b=batch_size)
      _e, _s = self.predict(_x, _cond, _chart, get_score=True)
      for c, e, s in zip(coords, _e, _s):
        energy_map[h, c] = e
        score_map[h, c] = np.asarray((s[1], s[0]), dtype=np.float32)
    
    fig, axs = plt.subplots(figsize=(6, 3), ncols=2, dpi=300)

    
    
    mesh = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    mesh = np.stack(mesh, axis=-1).astype(np.float32)




  def on_save_checkpoint(self, checkpoint):
    if self.trainer.is_global_zero:
      if self.current_epoch % self.config.vis_freq == 0:
        num_samples = len(self.predset)
        with kemono.gsm.utils.evaluate(self):
          for n in range(num_samples):
            self._preview_predictions(n)


def main(args):
  # create configurations
  configs = []
  if args.config is not None:
    configs.append(OmegaConf.load(args.config))
  configs.append(OmegaConf.from_dotlist(args.options))
  conf = OmegaConf.merge(*configs)
  OmegaConf.resolve(conf)
  # create model & trainer
  model = GsmTask(conf.task)

  checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor = 'val/loss',
    **conf.checkpoint
  )
  trainer = pl.Trainer(
    callbacks = checkpoint_callback,
    **conf.trainer
  )
  # start training
  trainer.fit(model)


if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(add_help=True)
  parser.add_argument('--config', type=str, required=True, help='configuration path')
  parser.add_argument('options', nargs=argparse.REMAINDER)
  args = parser.parse_args()
  main(args)
