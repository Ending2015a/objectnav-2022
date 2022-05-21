# --- built in ---
import os
import copy
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
from torch.utils.tensorboard import SummaryWriter
from matplotlib.colors import Normalize
# --- my module ---
import kemono
from kemono.data.dataset import Dataset

CONFIG_PATH = '/src/configs/test/test_hm3d.val_mini.rgbd.yaml'


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

class ToyMLP(nn.Module):
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
    layers.append(nn.Linear(in_dim, output_dim))

    self.net = nn.Sequential(*layers)
  
  def forward(self, x):
    return self.net(x)


class Energy(nn.Module):
  def __init__(self, net, num_classes):
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

  def forward(self, x, cond):
    cond = cond.to(dtype=torch.int64)
    cond = nn.functional.one_hot(cond, self.num_classes)
    cond = cond.to(dtype=torch.float32)
    inp = torch.cat((x, cond), dim=-1)
    return self.net(inp)

  def score(self, x, cond):
    cond = cond.to(dtype=torch.int64)
    cond = nn.functional.one_hot(cond, self.num_classes)
    cond = cond.to(dtype=torch.float32)
    cond = cond.requires_grad_()
    x = x.requires_grad_()
    inp = torch.cat((x, cond), dim=-1)
    logp = -self.net(inp).sum()
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
    runner,
    env,
    model,
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
      _dgm = np.asarray((_ps_x_gm-_ng_x_gm, 0., _ps_y_gm-_ng_y_gm), dtype=np.float32)
      # enforce boundary case (TODO euclidean distance?)
      _gm = 0 if np.isinf(_gm) or np.isnan(_gm) else _gm
      _dgm[np.isinf(_dgm)] = 0
      _dgm[np.isnan(_dgm)] = 0
      _dgm /= 2 * self.eps
      dgm.append(_dgm * _gm)
    
    dgm = np.asarray(dgm, dtype=np.float32).reshape((-1, 3))
    dgm = torch.tensor(dgm).to(device=self.device)
    return dgm

  def gsm2_loss(
    self,
    xp: torch.Tensor,
    cond: torch.Tensor,
    sigma = 1.0
  ) -> torch.Tensor:
    """
    Args:
      xp (torch.Tensor): random point (b, 3)
      cond (torch.Tensor): goal index (b,)
      sigma (float): sigma
    """
    dgm = self.compute_geodesic_grad(xp, cond)
    s = self.model.score(xp, cond)
    loss = torch.norm(s + dgm/(sigma**2), dim=-1) ** 2
    loss = loss.mean()/2.
    return loss

  def get_random_points(self, n_samples):
    """Sampling random noises
    Args:
      x (torch.Tensor): input samples
      n_slices (int, optional): number of slices. Defaults to None.
    Returns:
      torch.Tensor: sampled noises
    """
    if self.noise_type == 'map':
      height, width = self.topdown_map.shape
      available_points = []
      for h in range(height):
        for w in range(width):
          if self.topdown_map[h, w]:
            available_points.append((w, h))
      inds = np.random.randint(len(available_points), size=(n_samples,))
      points = np.asarray(available_points, dtype=np.float32)[inds]
      xp = self.to_3D_points(points)
    elif self.noise_type == 'sim':
      xp = []
      for n in range(n_samples):
        xp.append(self.env.sim.pathfinder.get_random_navigable_point())
      xp = np.asarray(xp, dtype=np.float32)
    elif self.noise_type == 'uniform':
      height, width = self.topdown_map.shape
      h = np.random.uniform(size=(n_samples,)) * height
      w = np.random.uniform(size=(n_samples,)) * width
      points = np.stack((w, h), axis=-1).astype(np.float32)
      xp = self.to_3D_points(points)
    else:
      raise NotImplementedError(
          f"Noise type '{self.noise_type}' not implemented."
      )
    return torch.tensor(xp).to(device=self.device)


  def get_loss(self, x):
    """Compute loss
    Args:
        x (torch.Tensor): input samples
    Returns:
        loss
    """
    if self.loss_type == 'gsm2':
      n_samples = len(x)
      xp = self.get_random_points(n_samples)
      loss = self.gsm2_loss(xp, x)
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
    cv2.imwrite(
      os.path.join(vis_path, 'rgb.png'),
      self.obs['rgb'][...,::-1]
    )
    cv2.imshow(os.path.join(vis_path, 'depth.png'),
      np.concatenate((self.obs['depth']*255,)*3, axis=-1).astype(np.uint8))
    self.agent_pos = env.sim.get_agent(0).state.position
    self.agent_height = self.agent_pos[1]
    self.meters_per_pixel = 0.1
    self.topdown_map = env.sim.pathfinder.get_topdown_view(
      self.meters_per_pixel, self.agent_height
    )
    self.topdown_map = self.topdown_map[:145, :70]
    self.bounds = env.sim.pathfinder.get_bounds()
    # generate random goals 2D
    self.goals = [
      self.sample_points(2),
      self.sample_points(4),
      self.sample_points(3),
      self.sample_points(2),
      self.sample_points(5),
      self.sample_points(8)
    ]
    self.n_goals = len(self.goals)
    self.vis_path = vis_path

    # create vector color palatte
    ph = np.linspace(0, 2*np.pi, 13)
    u = np.cos(ph)
    v = np.sin(ph)
    colors = np.arctan2(v, u)
    self.color_norm = Normalize()
    self.color_norm.autoscale(colors)

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
    height, width = self.topdown_map.shape
    energy_map = np.full((height, width), np.inf, dtype=np.float32)
    for h in range(height):
      for w in range(width):
        if (not mask) or (self.topdown_map[h, w]):
          xp = np.asarray((w, h), dtype=np.float32).reshape((1, 2))
          xp = self.trainer.to_3D_points(xp)
          xp = torch.from_numpy(xp).to(device=self.trainer.device)
          cond = torch.tensor([goal_id], device=self.trainer.device)
          e = self.trainer.model(xp, cond).detach().cpu().numpy()
          energy_map[h, w] = e
    # draw energy
    ax.grid(False)
    ax.axis('off')
    cmap = copy.copy(plt.cm.viridis)
    cmap.set_bad(color='black')
    ax.imshow(energy_map, cmap=cmap, interpolation='nearest')

    goals = self.goals[goal_id]
    ax.scatter(goals[..., 0], goals[..., 1], color='red', s=6)
    ax.set_title("Estimated energy", fontsize='xx-large')

  def plot_score_field(self, ax, goal_id, mask=False):
    height, width = self.topdown_map.shape
    score_map = np.zeros((height, width, 2), dtype=np.float32)
    for h in range(height):
      for w in range(width):
        if (not mask) or (self.topdown_map[h, w]):
          xp = np.asarray((w, h), dtype=np.float32).reshape((1, 2))
          xp = self.trainer.to_3D_points(xp)
          xp = torch.from_numpy(xp).to(device=self.trainer.device)
          cond = torch.tensor([goal_id], device=self.trainer.device)
          s = self.trainer.model.score(xp, cond).detach().cpu().numpy()
          score_map[h, w] = np.asarray((s[0][0], s[0][2]), dtype=np.float32)
    mesh = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    mesh = np.stack(mesh, axis=-1).astype(np.float32) # (y, x)
    mesh = mesh.reshape((-1, 2))
    score_map = score_map.reshape((-1, 2))
    mask = np.logical_and(score_map[..., 0]==0, score_map[..., 1]==0)
    mesh = mesh[~mask]
    score_map = score_map[~mask]

    theta = np.arctan2(score_map[...,1], score_map[...,0])

    ax.grid(False)
    ax.axis('off')
    ax.imshow(self.topdown_map, cmap='hot', interpolation='nearest')
    ax.quiver(
      mesh[...,1], mesh[...,0],
      score_map[...,0], score_map[...,1],
      color = plt.cm.hsv(self.color_norm(theta)),
      angles = 'xy',
      width = 0.002
    )

    goals = self.goals[goal_id]
    ax.scatter(goals[..., 0], goals[..., 1], color='red', s=6)
    ax.set_title("Estimated score", fontsize='xx-large')

  def plot_ground_truth(self, ax, goal_id):
    height, width = self.topdown_map.shape
    score_map = np.zeros((height, width, 2), dtype=np.float32)
    for h in range(height):
      for w in range(width):
        if self.topdown_map[h, w]:
          xp = np.asarray((w, h), dtype=np.float32).reshape((-1, 2))
          xp = self.trainer.to_3D_points(xp)
          xp = torch.from_numpy(xp).to(device=self.trainer.device)
          cond = torch.tensor([goal_id], device=self.trainer.device)
          s = self.trainer.compute_geodesic_grad(xp, cond).detach().cpu().numpy()
          score_map[h, w] = -np.asarray((s[0][0], s[0][2]), dtype=np.float32)
    mesh = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    mesh = np.stack(mesh, axis=-1).astype(np.float32) # (y, x)
    mesh = mesh.reshape((-1, 2))
    score_map = score_map.reshape((-1, 2))
    mask = np.logical_and(score_map[..., 0]==0, score_map[..., 1]==0)
    mesh = mesh[~mask]
    score_map = score_map[~mask]

    theta = np.arctan2(score_map[...,1], score_map[...,0])

    ax.grid(False)
    ax.axis('off')
    ax.imshow(self.topdown_map, cmap='hot', interpolation='nearest')
    ax.quiver(
      mesh[...,1], mesh[...,0],
      score_map[...,0], score_map[...,1],
      color = plt.cm.hsv(self.color_norm(theta)),
      angles = 'xy',
      width = 0.002,
      headwidth = 4,
      headlength = 6,
    )

    goals = self.goals[goal_id]
    ax.scatter(goals[..., 0], goals[..., 1], color='red', s=6)
    ax.set_title("Ground truth score", fontsize='xx-large')


  def visualize(self):
    logging.info('Visualizing data ...')
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
    # for idx in range(self.n_goals):
    #   name = f"goal{idx:02d}_{self.trainer.num_epochs:04d}.png"
    #   os.makedirs(self.vis_path, exist_ok=True)
    #   vis_path = os.path.join(self.vis_path, name)
    #   fig, axs = plt.subplots(figsize=(9, 6), ncols=3, dpi=300)
    #   self.plot_ground_truth(axs[0], idx)
    #   self.plot_score_field(axs[1], idx, mask=False)
    #   self.plot_energy_field(axs[2], idx, mask=False)
    #   plt.tight_layout()
    #   fig.savefig(vis_path, bbox_inches='tight', dpi=300, facecolor='white')
    #   plt.close('all')

  def run(self):
    num_train_samples = 400000
    num_eval_samples = 40000
    num_train_batches = num_train_samples // self.n_goals
    num_eval_batches = num_eval_samples // self.n_goals
    input_dim = 3 + self.n_goals
    # create model
    net = ToyMLP(
      input_dim = input_dim,
      output_dim = 1,
      units = [256, 256]
    )
    self.model = Energy(
      net = net,
      num_classes = self.n_goals
    )
    self.trainer = Trainer(
      self, self.env, self.model,
      learning_rate = 1e-3,
      clipnorm = 100.,
      eps = 0.1,
      loss_type = 'gsm2',
      noise_type = 'map',
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
    self.visualize()

    self.trainer.learn(
      train_dataset = train_dataset,
      eval_dataset = eval_dataset,
      n_epochs = 100,
      batch_size = 100,
      vis_callback = self.visualize
    )


def example():
  vis_path = '/src/logs/gsm-multi-goal-uniform_color/visualize/'
  import logging
  logging.basicConfig()
  logging.root.setLevel(logging.INFO)

  Runner(CONFIG_PATH, vis_path).run()



if __name__ == '__main__':
  example()
