# --- built in ---
import os
import sys
import time
import logging
import functools
# --- 3rd party ---
import cv2
import gym
import habitat
import numpy as np
from habitat.sims.habitat_simulator.actions import HabitatSimActions
import matplotlib.pyplot as plt
import torch
from torch import nn
import dungeon_maps as dmap
from torch.utils.tensorboard import SummaryWriter
# --- my module ---
import kemono
from kemono.data.dataset import Dataset
from kemono.semantics import utils

CONFIG_PATH = '/src/configs/test/test_hm3d.val_mini.rgbd.yaml'

def split_tiles(image, tile_size, stride):
  height, width = image.shape
  nrows = int(np.ceil(height/stride).item())
  ncols = int(np.ceil(width/stride).item())
  h_pad = int((nrows-1) * stride + tile_size - height)
  w_pad = int((ncols-1) * stride + tile_size - width)
  tiles = []
  centers = []
  padded = np.pad(image, ((0, h_pad), (0, w_pad)))
  for h_idx in range(nrows):
    for w_idx in range(ncols):
      start_h = h_idx * stride
      start_w = w_idx * stride
      center_h = start_h + tile_size//2
      center_w = start_w + tile_size//2
      tile = padded[start_h:start_h+tile_size, start_w:start_w+tile_size]
      tiles.append(tile)
      centers.append((center_w, center_h))
  return tiles, centers

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
        Swish(out_dim) if swish else nn.Softplus(),
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
    self._cached = None

  def forward(
    self,
    x: torch.Tensor,
    chart: torch.Tensor,
    reset_chart: bool = True
  ) -> torch.Tensor:
    x_inp = self.mlp(x)
    if self._cached is None or reset_chart:
      x_chart = self.cnn(chart)
    else:
      x_chart = self._cached
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

  def forward(self, x, cond, chart, reset_chart=True):
    cond = cond.to(dtype=torch.int64)
    cond = nn.functional.one_hot(cond, self.num_classes)
    cond = cond.to(dtype=torch.float32)
    inp = torch.cat((x, cond), dim=-1)
    return self.net(inp, chart, reset_chart=reset_chart)

  def score(self, x, cond, chart, reset_chart=True):
    cond = cond.to(dtype=torch.int64)
    cond = nn.functional.one_hot(cond, self.num_classes)
    cond = cond.to(dtype=torch.float32)
    cond = cond.requires_grad_()
    x = x.requires_grad_()
    inp = torch.cat((x, cond), dim=-1)
    logp = -self.net(inp, chart, reset_chart=reset_chart).sum()
    return torch.autograd.grad(logp, x, create_graph=True)[0]

  def save(self, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(self.state_dict(), path)

  def load(self, path):
    self.load_state_dict(torch.load(path))
    return self


class Trainer():
  def __init__(
    self,
    runner: "Runner",
    env,
    model,
    chart_size = 120,
    crop_size = 600,
    learning_rate = 1e-3,
    clipnorm = 100.,
    eps = 0.05,
    loss_type = 'gsm2',
    noise_type = 'uniform',
    device = 'cuda'
  ):
    """Energy based model trainer
    Args:
        model (nn.Module): energy-based model
        learning_rate (float, optional): learning rate. Defaults to 1e-4.
        clipnorm (float, optional): gradient clip. Defaults to 100..
        n_slices (int, optional): number of slices for sliced score matching loss.
            Defaults to 1.
        loss_type (str, optional): type of loss. Can be 'ssm-vr', 'ssm', 'deen',
            'dsm'. Defaults to 'ssm-vr'.
        noise_type (str, optional): type of noise. Can be 'radermacher', 'sphere'
            or 'gaussian'. Defaults to 'radermacher'.
        device (str, optional): torch device. Defaults to 'cuda'.
    """
    self.runner = runner
    self.env = env
    self.model = model
    self.learning_rate = learning_rate
    self.clipnorm = clipnorm
    self.eps = eps
    self.loss_type = loss_type.lower()
    self.noise_type = noise_type.lower()
    self.device = device
    self.crop_size = crop_size
    self.chart_size = chart_size

    self.model = self.model.to(device=self.device)
    # setup optimizer
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
    
    self.num_gradsteps = 0
    self.num_epochs = 0
    self.progress = 0
    self.tb_writer = None

    self.meters_per_pixel = runner.meters_per_pixel
    self.agent_height = runner.agent_height
    self.bounds = runner.bounds
    self.topdown_map = runner.topdown_map
    self.agent_pos = runner.agent_pos
    self.goals = runner.goals
    self.n_goals = runner.n_goals
    self.highres_meters_per_pixel = runner.highres_meters_per_pixel
    self.highres_topdown_map = runner.highres_topdown_map
    self.highres_topdown_map_th = torch.tensor(
      self.highres_topdown_map, dtype=torch.float32, device=device
    )
    

  def to_2D_points(self, x):
    """Convert world space 3D point to topdown space 2D point."""

    px = (x[..., 0] - self.bounds[0][0]) / self.meters_per_pixel
    py = (x[..., 2] - self.bounds[0][2]) / self.meters_per_pixel

    return np.stack((px, py), axis=-1).astype(np.float32)

  def to_3D_points(self, x, h=None):
    if h is None:
      h = self.agent_height

    x_3d = x[..., 0] * self.meters_per_pixel + self.bounds[0][0]
    y_3d = np.full(x_3d.shape, h, dtype=np.float32)
    z_3d = x[..., 1] * self.meters_per_pixel + self.bounds[0][2]

    return np.stack((x_3d, y_3d, z_3d), axis=-1).astype(np.float32)

  def highres_to_2D_points(self, x):
    """Convert world space 3D point to topdown space 2D point."""

    px = (x[..., 0] - self.bounds[0][0]) / self.highres_meters_per_pixel
    py = (x[..., 2] - self.bounds[0][2]) / self.highres_meters_per_pixel

    return np.stack((px, py), axis=-1).astype(np.float32)

  def highres_to_3D_points(self, x, h=None):
    if h is None:
      h = self.agent_height

    x_3d = x[..., 0] * self.highres_meters_per_pixel + self.bounds[0][0]
    y_3d = np.full(x_3d.shape, h, dtype=np.float32)
    z_3d = x[..., 1] * self.highres_meters_per_pixel + self.bounds[0][2]

    return np.stack((x_3d, y_3d, z_3d), axis=-1).astype(np.float32)

  def compute_geodesic_grad(
    self,
    xp: torch.Tensor,
    cond: torch.Tensor
  ) -> torch.Tensor:
    xp = xp.detach().cpu().numpy()
    cond = cond.detach().cpu().numpy()
    cond = cond.astype(np.int64)

    dgm = []

    eps_x = np.asarray((self.eps, 0., 0.), dtype=np.float32)
    eps_y = np.asarray((0., 0., self.eps), dtype=np.float32)

    for _xp, _cond in zip(xp, cond):
      goals = self.goals[_cond]
      goals = self.to_3D_points(goals)
      _gm = self.env.sim.geodesic_distance(_xp, goals)
      _ps_x_gm = self.env.sim.geodesic_distance(_xp+eps_x, goals)
      _ng_x_gm = self.env.sim.geodesic_distance(_xp-eps_x, goals)
      _ps_y_gm = self.env.sim.geodesic_distance(_xp+eps_y, goals)
      _ng_y_gm = self.env.sim.geodesic_distance(_xp-eps_y, goals)
      _dgm = np.asarray((_ps_x_gm-_ng_x_gm, _ps_y_gm-_ng_y_gm), dtype=np.float32)
      # enforce boundary case (TODO euclidean distance?)
      _gm = 0 if np.isinf(_gm) or np.isnan(_gm) else _gm
      _dgm[np.isinf(_dgm)] = 0
      _dgm[np.isnan(_dgm)] = 0
      _dgm /= 2 * self.eps
      dgm.append(_dgm * _gm)
    
    dgm = np.asarray(dgm, dtype=np.float32).reshape((-1, 2))
    dgm = torch.tensor(dgm).to(device=self.device)
    return dgm

  def gsm2_loss(
    self,
    xp: torch.Tensor,
    cond: torch.Tensor,
    chart: torch.Tensor,
    gt: torch.Tensor,
    sigma = 1.0
  ) -> torch.Tensor:
    """
    Args:
      xp (torch.Tensor): random point (b, 2)
      cond (torch.Tensor): goal index (b,)
      chart (torch.Tensor): chart (b, 1, h, w)
      gt (torch.Tensor): ground truth 3D position (b, 3)
      sigma (float): sigma
    """
    dgm = self.compute_geodesic_grad(gt, cond)
    s = self.model.score(xp, cond, chart)
    loss = torch.norm(s + dgm/(sigma**2), dim=-1) ** 2
    loss = loss.mean()/2.
    return loss

  @torch.no_grad()
  def get_random_charts(self, n_samples):
    """Sampling random charts
    Returns:
      torch.Tensor: sampled charts (b, 1, crop_size, crop_size)
      torch.Tensor: center points [w, h], (b, 2)
    """
    # sample center of the charts
    centers = []
    for n in range(n_samples):
      centers.append(self.env.sim.pathfinder.get_random_navigable_point())
    centers = np.asarray(centers, dtype=np.float32)
    centers_2D = self.highres_to_2D_points(centers) # (w, h)
    height, width = self.highres_topdown_map.shape
    grid = dmap.utils.generate_crop_grid(
      centers_2D,
      image_width = width,
      image_height = height,
      crop_width = self.crop_size,
      crop_height = self.crop_size
    ) # (b, crop_size, crop_size, 2)
    centers = torch.tensor(centers, dtype=torch.float32)
    centers = centers.to(device=self.device)
    map_th = torch.broadcast_to(
      self.highres_topdown_map_th,
      (n_samples, 1, height, width)
    )
    charts = dmap.utils.image_sample(
      image = map_th,
      grid = grid
    ) # (b, 1, h, w)
    return charts, centers

  @torch.no_grad()
  def get_random_points(self, charts, centers):
    xp = []
    gt = []
    for b_idx in range(len(charts)):
      chart = charts[b_idx, 0]
      chart_inds = chart.nonzero().cpu().numpy()
      center = centers[b_idx].cpu().numpy()
      height, width = chart.shape
      # random sample index (h, w)
      if len(chart_inds) > 0:
        idx = np.random.randint(len(chart_inds))
        chart_idx = chart_inds[idx]
      else:
        chart_idx = np.asarray([height//2, width//2])
      chart_idx = chart_idx / np.asarray((height, width), dtype=np.float32)
      chart_idx = chart_idx * 2.0 - 1.0
      # convert to (x, z)
      x = np.asarray((chart_idx[1], 0, chart_idx[0]), dtype=np.float32)
      g = x * np.asarray((width, 1.0, height))/2.0 * self.highres_meters_per_pixel + center
      xp.append(chart_idx[::-1])
      gt.append(g)

    xp = np.asarray(xp, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    return (
      torch.tensor(xp).to(device=self.device),
      torch.tensor(gt).to(device=self.device)
    )

  def get_loss(self, x):
    """Compute loss
    Args:
        x (torch.Tensor): input samples
    Returns:
        loss
    """
    if self.loss_type == 'gsm2':
      n_samples = len(x)
      charts, centers = self.get_random_charts(n_samples)
      xp, gt = self.get_random_points(charts, centers)
      charts = nn.functional.interpolate(charts, size=(self.chart_size, self.chart_size))
      loss = self.gsm2_loss(xp, x, charts, gt)
    else:
      raise NotImplementedError(
        f"Loss type '{self.loss_type}' not implemented."
      )
    return loss

  def train_step(self, batch, update=True):
    """Train one batch
    Args:
        batch (dict): batch data
        update (bool, optional): whether to update networks. 
            Defaults to True.
    Returns:
        loss
    """
    x = batch['samples']
    # move inputs to device
    x = torch.tensor(x, dtype=torch.int64, device=self.device)
    # compute losses
    loss = self.get_loss(x)
    # update model
    if update:
      # compute gradients
      loss.backward()
      # perform gradient updates
      nn.utils.clip_grad_norm_(self.model.parameters(), self.clipnorm)
      self.optimizer.step()
      self.optimizer.zero_grad()
    return loss.item()

  def train(self, dataset, batch_size):
    """Train one epoch
    Args:
        dataset (tf.data.Dataset): Tensorflow dataset
        batch_size (int): batch size
    Returns:
        np.ndarray: mean loss
    """        
    all_losses = []
    dataset = dataset.batch(batch_size)
    for batch_data in dataset:
      sample_batch = {
        'samples': batch_data
      }
      loss = self.train_step(sample_batch)
      self.num_gradsteps += 1
      all_losses.append(loss)
    m_loss = np.mean(all_losses).astype(np.float32)
    return m_loss

  def eval(self, dataset, batch_size):
    """Eval one epoch
    Args:
        dataset (tf.data.Dataset): Tensorflow dataset
        batch_size (int): batch size
    Returns:
        np.ndarray: mean loss
    """        
    all_losses = []
    dataset = dataset.batch(batch_size)
    for batch_data in dataset:
      sample_batch = {
        'samples': batch_data
      }
      loss = self.train_step(sample_batch, update=False)
      all_losses.append(loss)
    m_loss = np.mean(all_losses).astype(np.float32)
    return m_loss

  def learn(
    self,
    train_dataset,
    eval_dataset = None,
    n_epochs = 5,
    batch_size = 100,
    log_freq = 1,
    eval_freq = 1,
    vis_freq = 1,
    vis_callback = None,
    tb_logdir = None
  ):
    """Train the model
    Args:
      train_dataset (tf.data.Dataset): training dataset
      eval_dataset (tf.data.Dataset, optional): evaluation dataset.
          Defaults to None.
      n_epochs (int, optional): number of epochs to train. Defaults to 5.
      batch_size (int, optional): batch size. Defaults to 100.
      log_freq (int, optional): logging frequency (epoch). Defaults to 1.
      eval_freq (int, optional): evaluation frequency (epoch). Defaults to 1.
      vis_freq (int, optional): visualizing frequency (epoch). Defaults to 1.
      vis_callback (callable, optional): visualization function. Defaults to None.
      tb_logdir (str, optional): path to tensorboard files. Defaults to None.
    Returns:
      self
    """
    if tb_logdir is not None:
      self.tb_writer = SummaryWriter(tb_logdir)

    # initialize
    time_start = time.time()
    time_spent = 0
    total_epochs = n_epochs

    for epoch in range(n_epochs):
      self.num_epochs += 1
      self.progress = float(self.num_epochs) / float(n_epochs)
      # train one epoch
      loss = self.train(train_dataset, batch_size)
      # write tensorboard
      if self.tb_writer is not None:
        self.tb_writer.add_scalar(f'train/loss', loss, self.num_epochs)
      
      if (log_freq is not None) and (self.num_epochs % log_freq == 0):
        logging.info(
          f"[Epoch {self.num_epochs}/{total_epochs}]: loss: {loss}"
        )

      if (eval_dataset is not None) and (self.num_epochs % eval_freq == 0):
        # evaluate
        self.model.eval()
        eval_loss = self.eval(eval_dataset, batch_size)
        self.model.train()

        if self.tb_writer is not None:
          self.tb_writer.add_scalar(f'eval/loss', eval_loss, self.num_epochs)
        
        logging.info(
          f"[Eval {self.num_epochs}/{total_epochs}]: loss: {eval_loss}"
        )

      if (vis_callback is not None) and (self.num_epochs % vis_freq == 0):
        logging.debug("Visualizing")
        self.model.eval()
        vis_callback()
        self.model.train()
    return self

class Runner():
  def __init__(self, config_path, vis_path):
    config = kemono.get_config(config_path)
    env = habitat.Env(config=config)
    self.env = env
    self.obs = env.reset()
    self.agent_pos = env.sim.get_agent(0).state.position
    self.agent_height = self.agent_pos[1]
    self.meters_per_pixel = 0.1
    self.highres_meters_per_pixel = 0.03
    self.topdown_map = env.sim.pathfinder.get_topdown_view(
      self.meters_per_pixel, self.agent_height
    )
    self.highres_topdown_map = env.sim.pathfinder.get_topdown_view(
      self.highres_meters_per_pixel, self.agent_height
    )
    self.topdown_map = self.topdown_map
    self.highres_topdown_map = self.highres_topdown_map
    self.bounds = env.sim.pathfinder.get_bounds()
    # generate random goals 2D
    self.goals = [
      self.sample_points(2),
      #self.sample_points(4),
      #self.sample_points(3),
      #self.sample_points(2),
      #self.sample_points(5),
      #self.sample_points(8)
    ]
    self.n_goals = len(self.goals)
    self.vis_path = vis_path
    self.crop_size = 200
    self.chart_size = 128
    self.render_stride = int(self.crop_size * 0.9)

    self.render_centers = np.array([
      (99, 99),
      (99, 149),
      (299, 99),
      (289, 149),
      (399, 99),
      (399, 175)
    ]) # (h, w)


  @property
  def habitat_env(self):
    return self.env

  def sample_points(self, n_samples):
    height, width = self.topdown_map.shape
    available_points = []
    for h in range(height):
      for w in range(width):
        if self.topdown_map[h, w]:
          available_points.append((w, h))
    available_points = np.asarray(available_points, dtype=np.float32)
    point_idx = np.random.choice(np.arange(len(available_points)), size=(n_samples,))
    return np.asarray(available_points[point_idx])

  def plot_energy_field(self, ax, goal_id, mask=False):
    height, width = self.highres_topdown_map.shape
    energy_map = np.zeros((height, width), dtype=np.float32)
    average_map = np.zeros((height, width), dtype=np.float32)

    centers = np.asarray(self.render_centers, dtype=np.int64)
    centers = centers[..., ::-1].copy() # (w, h)
    b = centers.shape[0]
    topdown_map = torch.tensor(self.highres_topdown_map, dtype=torch.float32)
    topdown_map = torch.broadcast_to(topdown_map, (b, 1, height, width))
    grid = dmap.utils.generate_crop_grid(
      centers,
      image_width = width,
      image_height = height,
      crop_width = self.crop_size,
      crop_height = self.crop_size
    )
    tiles = dmap.utils.image_sample(topdown_map, grid).numpy() # (b, 1, h, w)

    for tile_idx, (tile, center) in enumerate(zip(tiles, centers)):
      print(f'Rendering energy, {tile_idx+1}/{len(tiles)}...')
      # prepare network inputs
      tile = tile[0] # (h, w)
      tile_height, tile_width = tile.shape
      chart = torch.tensor(tile, device=self.trainer.device).to(dtype=torch.float32)
      chart = chart.reshape((1, 1, tile_height, tile_width))
      # resize chart to input size
      chart = nn.functional.interpolate(chart, size=(self.chart_size, self.chart_size))
      cond = torch.tensor([goal_id], device=self.trainer.device)
      reset_chart = True
      for h in range(tile_height):
        for w in range(tile_width):
          gw = center[0] + w - tile_width//2
          gh = center[1] + h - tile_height//2
          # the padded size will out of the bounds
          if gh < height and gw < width:
            if (not mask) or (tile[h, w]):
              xp = np.asarray((w, h), dtype=np.float32)
              norm = np.asarray((tile_width, tile_height), dtype=np.float32)
              norm_xp = xp / norm * 2.0 - 1.0
              norm_xp = norm_xp.reshape((1, 2))
              norm_xp = torch.from_numpy(norm_xp).to(device=self.trainer.device)
              e = self.trainer.model(norm_xp, cond, chart, reset_chart).detach().cpu().numpy()
              energy_map[gh, gw] += e
            average_map[gh, gw] += 1
          reset_chart = False
    energy_map = np.divide(
      energy_map, average_map,
      out = np.zeros_like(energy_map),
      where = (average_map != 0)
    )

    target_height, target_width = self.topdown_map.shape
    energy_map = utils.resize_image(energy_map, (target_height, target_width)).numpy()

    # draw energy
    ax.grid(False)
    ax.axis('off')
    ax.imshow(energy_map[:145, :70], cmap=plt.cm.viridis)

    goals = self.goals[goal_id]
    ax.scatter(goals[..., 0], goals[..., 1], color='yellow', s=6)
    ax.set_title("Estimated energy", fontsize=16)

  def plot_score_field(self, ax, goal_id, mask=False):
    height, width = self.highres_topdown_map.shape
    score_map = np.zeros((height, width, 2), dtype=np.float32)
    average_map = np.zeros((height, width, 2), dtype=np.float32)

    centers = np.asarray(self.render_centers, dtype=np.int64)
    centers = centers[..., ::-1].copy() # (w, h)
    b = centers.shape[0]
    topdown_map = torch.tensor(self.highres_topdown_map, dtype=torch.float32)
    topdown_map = torch.broadcast_to(topdown_map, (b, 1, height, width))
    grid = dmap.utils.generate_crop_grid(
      centers,
      image_width = width,
      image_height = height,
      crop_width = self.crop_size,
      crop_height = self.crop_size
    )
    tiles = dmap.utils.image_sample(topdown_map, grid).numpy() # (b, 1, h, w)

    for tile_idx, (tile, center) in enumerate(zip(tiles, centers)):
      print(f'Rendering score, {tile_idx+1}/{len(tiles)}...')
      # prepare network inputs
      tile = tile[0] # (h, w)
      tile_height, tile_width = tile.shape
      chart = torch.tensor(tile, device=self.trainer.device).to(dtype=torch.float32)
      chart = chart.reshape((1, 1, tile_height, tile_width))
      # resize chart to input size
      chart = nn.functional.interpolate(chart, size=(self.chart_size, self.chart_size))
      cond = torch.tensor([goal_id], device=self.trainer.device)
      reset_chart = True
      for h in range(tile_height):
        for w in range(tile_width):
          gw = center[0] + w - tile_width//2
          gh = center[1] + h - tile_height//2
          # the padded size will out of the bounds
          if gh >= 0 and gw >= 0 and gh < height and gw < width:
            if (not mask) or (tile[h, w]):
              xp = np.asarray((w, h), dtype=np.float32)
              norm = np.asarray((tile_width, tile_height), dtype=np.float32)
              norm_xp = xp / norm * 2.0 - 1.0
              norm_xp = norm_xp.reshape((1, 2))
              norm_xp = torch.from_numpy(norm_xp).to(device=self.trainer.device)
              s = self.trainer.model.score(norm_xp, cond, chart, reset_chart).detach().cpu().numpy()
              score_map[gh, gw] += np.asarray((s[0][1], s[0][0]), dtype=np.float32)
            average_map[gh, gw] += 1
          reset_chart = False
    score_map = np.divide(
      score_map, average_map,
      out = np.zeros_like(score_map),
      where = (average_map != 0)
    )
    target_height, target_width = self.topdown_map.shape

    score_map = np.transpose(score_map, (2, 0, 1))
    score_map = utils.resize_image(score_map, (target_height, target_width)).numpy()
    score_map = np.transpose(score_map, (1, 2, 0))
    score_map = score_map[:145, :70]

    mesh = np.meshgrid(np.arange(target_height), np.arange(target_width), indexing='ij')
    mesh = np.stack(mesh, axis=-1).astype(np.float32)
    mesh = mesh[:145, :70]
    mesh = mesh.reshape((-1, 2))
    score_map = score_map.reshape((-1, 2))
    mask = np.logical_and(score_map[..., 0]==0, score_map[..., 1]==0)
    mesh = mesh[~mask]
    score_map = score_map[~mask]

    ax.grid(False)
    ax.axis('off')
    ax.imshow(self.topdown_map[:145, :70], cmap='hot')
    ax.quiver(mesh[...,1], mesh[...,0], score_map[...,1], score_map[...,0],
      angles='xy', color='b')

    goals = self.goals[goal_id]
    ax.scatter(goals[..., 0], goals[..., 1], color='yellow', s=6)
    ax.set_title("Estimated score", fontsize=16)

  def plot_ground_truth(self, ax, goal_id):
    topdown_map = self.topdown_map[:145, :70]
    height, width = topdown_map.shape
    score_map = np.zeros((height, width, 2), dtype=np.float32)
    for h in range(height):
      for w in range(width):
        if self.topdown_map[h, w]:
          xp = np.asarray((w, h), dtype=np.float32).reshape((-1, 2))
          xp = self.trainer.to_3D_points(xp)
          xp = torch.from_numpy(xp).to(device=self.trainer.device)
          cond = torch.tensor([goal_id], device=self.trainer.device)
          s = self.trainer.compute_geodesic_grad(xp, cond).detach().cpu().numpy()
          score_map[h, w] = -np.asarray((s[0][1], s[0][0]), dtype=np.float32)
    mesh = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    mesh = np.stack(mesh, axis=-1).astype(np.float32) # (y, x)
    mesh = mesh.reshape((-1, 2))
    score_map = score_map.reshape((-1, 2))
    mask = np.logical_and(score_map[..., 0]==0, score_map[..., 1]==0)
    mesh = mesh[~mask]
    score_map = score_map[~mask]
    ax.grid(False)
    ax.axis('off')
    ax.imshow(topdown_map, cmap='hot')
    ax.quiver(mesh[...,1], mesh[...,0], score_map[...,1], score_map[...,0],
      angles='xy', color='b')

    goals = self.goals[goal_id]
    ax.scatter(goals[..., 0], goals[..., 1], color='yellow', s=6)
    ax.set_title("Ground truth", fontsize=16)


  def visualize(self):
    logging.info('Visualizing data (masked)...')
    for idx in range(self.n_goals):
      name = f"goal{idx:02d}_{self.trainer.num_epochs:04d}_mask.png"
      os.makedirs(self.vis_path, exist_ok=True)
      vis_path = os.path.join(self.vis_path, name)
      fig, axs = plt.subplots(figsize=(9, 6), ncols=3, dpi=300)
      self.plot_ground_truth(axs[0], idx)
      self.plot_score_field(axs[1], idx, mask=True)
      self.plot_energy_field(axs[2], idx, mask=True)
      plt.tight_layout()
      fig.savefig(vis_path, bbox_inches='tight', dpi=300, facecolor='white')
      plt.close('all')
    logging.info('Visualizing data ...')
    for idx in range(self.n_goals):
      name = f"goal{idx:02d}_{self.trainer.num_epochs:04d}.png"
      os.makedirs(self.vis_path, exist_ok=True)
      vis_path = os.path.join(self.vis_path, name)
      fig, axs = plt.subplots(figsize=(9, 6), ncols=3, dpi=300)
      self.plot_ground_truth(axs[0], idx)
      self.plot_score_field(axs[1], idx, mask=False)
      self.plot_energy_field(axs[2], idx, mask=False)
      plt.tight_layout()
      fig.savefig(vis_path, bbox_inches='tight', dpi=300, facecolor='white')
      plt.close('all')

  def run(self):
    num_train_samples = 400000
    num_eval_samples = 40000
    num_train_batches = num_train_samples // self.n_goals
    num_eval_batches = num_eval_samples // self.n_goals
    input_dim = 2 + self.n_goals
    crop_size = self.crop_size
    chart_size = self.chart_size
    # create model
    net = ToyNet(
      chart_shape = (1, chart_size, chart_size),
      input_dim = input_dim,
      output_dim = 1,
      mlp_units = [256],
      fuse_units = [256, 256]
    )
    self.model = Energy(
      net = net,
      num_classes = self.n_goals
    )
    self.trainer = Trainer(
      self, self.env, self.model,
      chart_size = chart_size,
      crop_size = crop_size,
      learning_rate = 1e-3,
      clipnorm = 100.,
      eps = 0.1,
      loss_type = 'gsm2',
      noise_type = 'sim',
    )
    # create datasets
    train_dataset = (
      Dataset.range(self.n_goals)
             .repeat(num_train_batches)
             .shuffle(self.n_goals)
    )
    eval_dataset = (
      Dataset.range(self.n_goals).repeat(num_eval_batches)
    )
    # debug
    #self.visualize()
    #exit(0)

    self.trainer.learn(
      train_dataset = train_dataset,
      eval_dataset = eval_dataset,
      n_epochs = 100,
      batch_size = 4,
      vis_callback = self.visualize
    )


def example():
  vis_path = '/src/logs/gsm-multi-goal-chart/visualize/'
  import logging
  logging.basicConfig()
  logging.root.setLevel(logging.INFO)

  Runner(CONFIG_PATH, vis_path).run()



if __name__ == '__main__':
  example()
