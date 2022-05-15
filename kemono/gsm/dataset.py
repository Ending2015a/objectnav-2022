# --- built in ---
import os
import dataclasses
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
# --- 3rd party ---
import cv2
import habitat
import numpy as np
import gym
import dungeon_maps as dmap
import torch
import rlchemy
import matplotlib.pyplot as plt
import einops
# --- my module ---

"""
/src/logs/kemono_gsm_gt/train/xdjlkd3js98f/ep-0001/
topdown_view_low.png
topdown_view_high.png

topdown_view_low_gt.npy
topdown_view_high_gt.npy

metadata.json

sample_%05d.npy
  objectgoal: 0~6 (), np.int64
  centers: chart centers, [w, h], 0 ~ W,H, (2,) np.int64
  points: normalized points on charts, [w, h], -1.0 ~ 1.0, (1000, 2) np.float32
  distances: (1000,) np.float32 for each goals
  gradients: (1000, 2)
"""

@dataclass
class GsmData():
  """
  objectgoal: 0~6, (), np.int64
  chart: chart, (h, w), bool
  chart_gt: chart ground truth, (h, w, 3), [dist, v0, v1], np.float32
    np.inf for invalid cells
  center: chart center, [w, h], (2,), np.int64
  points: normalized points on charts [w, h], -1.0 ~ 1.0, (-1, 2), np.float32
  distances: (-1,), np.float32
  gradients: (-1, 2), np.float32
  """
  objectgoal: int = None
  chart: np.ndarray = None
  chart_gt: np.ndarray = None
  center: np.ndarray = None
  points: np.ndarray = None
  distances: np.ndarray = None
  gradients: np.ndarray = None

  def save(self, filepath):
    rlchemy.utils.safe_makedirs(filepath=filepath)
    np.savez(
      filepath,
      objectgoal = self.objectgoal,
      chart = self.chart.astype(bool),
      chart_gt = self.chart_gt.astype(np.float32),
      center = self.center.astype(np.int64),
      points = self.points.astype(np.float32),
      distances = self.distances.astype(np.float32),
      gradients = self.gradients.astype(np.float32)
    )

  @classmethod
  def load(cls, filepath) -> "GsmData":
    assert os.path.isfile(filepath)
    npz = np.load(filepath)
    self = cls()
    self.objectgoal = npz['objectgoal'].item()
    self.chart = npz['chart']
    self.chart_gt = npz['chart_gt']
    self.center = npz['center']
    self.points = npz['points']
    self.distances = npz['distances']
    self.gradients = npz['gradients']
    return self

@dataclass
class GsmEpisodeData():
  topdown_low: np.ndarray = None
  topdown_high: np.ndarray = None
  topdown_low_gt: np.ndarray = None
  topdown_high_gt: np.ndarray = None
  paths: List[str] = dataclasses.field(default_factory=list)
  info: Dict[str, Any] = dataclasses.field(default_factory=dict)

  def save(self, dirpath):
    rlchemy.utils.safe_makedirs(dirpath=dirpath)
    # save topdown maps
    if self.topdown_low is not None:
      topdown_low = self.topdown_low.astype(np.uint8) * 255
      topdown_low_path = os.path.join(dirpath, 'topdown_view_low.png')
      cv2.imwrite(topdown_low_path, topdown_low)
    if self.topdown_high is not None:
      topdown_high = self.topdown_high.astype(np.uint8) * 255
      topdown_high_path = os.path.join(dirpath, 'topdown_view_high.png')
      cv2.imwrite(topdown_high_path, topdown_high)

    # save topdown gt
    if self.topdown_low_gt is not None:
      topdown_low_gt_path = os.path.join(dirpath, 'topdown_view_low_gt.npy')
      np.save(topdown_low_gt_path, self.topdown_low_gt)
    if self.topdown_high_gt is not None:
      topdown_high_gt_path = os.path.join(dirpath, 'topdown_view_high_gt.npy')
      np.save(topdown_high_gt_path, self.topdown_high_gt)

    # save paths (relative)
    paths = [
      os.path.relpath(path, dirpath)
      for path in self.paths
    ]
    info_path = os.path.join(dirpath, 'contents.json')
    info = {
      'paths': paths,
      'info': self.info
    }
    rlchemy.utils.safe_json_dump(info_path, info)

  @classmethod
  def load(cls, dirpath) -> "GsmEpisodeData":
    self = cls()
    # load topdown maps
    topdown_low_path = os.path.join(dirpath, 'topdown_view_low.png')
    if os.path.isfile(topdown_low_path):
      topdown_low = cv2.imread(topdown_low_path, cv2.IMREAD_GRAYSCALE)
      self.topdown_low = topdown_low.astype(bool)
    topdown_high_path = os.path.join(dirpath, 'topdown_view_high.png')
    if os.path.isfile(topdown_high_path):
      topdown_high = cv2.imread(topdown_high_path, cv2.IMREAD_GRAYSCALE)
      self.topdown_high = topdown_high.astype(bool)

    # load topdown gt
    topdown_low_gt_path = os.path.join(dirpath, 'topdown_view_low_gt.npy')
    if os.path.isfile(topdown_low_gt_path):
      self.topdown_low_gt = np.load(topdown_low_gt_path)
    topdown_high_gt_path = os.path.join(dirpath, 'topdown_view_high_gt.npy')
    if os.path.isfile(topdown_high_gt_path):
      self.topdown_high_gt = np.load(topdown_high_gt_path)

    # load info
    contents_path = os.path.join(dirpath, 'contents.json')
    if os.path.isfile(contents_path):
      contents = rlchemy.utils.safe_json_load(contents_path)
      self.paths = [
        os.path.abspath(os.path.join(dirpath, path))
        for path in contents['paths']
      ]
      self.info = contents['info']
    return self

class GsmDataSampler():
  def __init__(
    self,
    env: habitat.Env,
    goal_mapping,
    meters_per_pixel_low: float = 0.1,
    meters_per_pixel_high: float = 0.03,
    eps: float = 0.1,
  ):
    self.env = env
    self.sim = env.sim
    self.goal_mapping = goal_mapping
    self.goal_mapping_inv = {v:k for k, v in goal_mapping.items()}
    self.meters_per_pixel_low = meters_per_pixel_low
    self.meters_per_pixel_high = meters_per_pixel_high
    self.eps = eps

    self.agent_height = None
    self.topdown_low = None
    self.topdown_high = None
    self.bounds = None
    self.episode = None
    self.goals = None

  def sample(
    self,
    dirpath: str,
    max_charts: int = 1000,
    min_points: int = 1000,
    chart_width: int = 300,
    chart_height: int = 300,
    masking: bool = False
  ):

    data = GsmEpisodeData()
    data.info['args'] = dict(
      max_charts = max_charts,
      min_points = min_points,
      chart_width = chart_width,
      chart_height = chart_height,
      masking = masking
    )

    agent_height = self.sim.get_agent(0).state.position[1]
    self.agent_height = agent_height
    self.topdown_low = self.sim.pathfinder.get_topdown_view(
      self.meters_per_pixel_low, agent_height
    )
    self.topdown_high = self.sim.pathfinder.get_topdown_view(
      self.meters_per_pixel_high, agent_height
    )
    self.bounds = self.sim.pathfinder.get_bounds()

    data.info['agent_height'] = agent_height
    data.info['bounds'] = self.bounds
    data.info['meters_per_pixel_low'] = self.meters_per_pixel_low
    data.info['meters_per_pixel_high'] = self.meters_per_pixel_high
    data.info['eps'] = self.eps
    data.topdown_low = self.topdown_low.copy()
    data.topdown_high = self.topdown_high.copy()

    episode = self.env.current_episode
    self.episode = episode
    # self.goals = [
    #   goal.position
    #   for goal in episode.goals
    # ]
    # goals view points
    self.goals = np.asarray([
      view_point.agent_state.position
      for goal in episode.goals
      for view_point in goal.view_points
    ])
    episode_id = int(self.env.current_episode.episode_id)
    scene_id = os.path.basename(episode.scene_id).split('.')[0]
    object_category = episode.object_category
    objectgoal = self.goal_mapping_inv[object_category]
    goals_pos = np.asarray([goal.position for goal in episode.goals])

    data.info['episode_id'] = episode_id
    data.info['scene_id'] = scene_id
    data.info['object_category'] = object_category
    data.info['objectgoal'] = objectgoal
    data.info['goals'] = goals_pos.copy()
    data.info['view_goals'] = self.goals.copy()

    ep_dirpath = os.path.join(dirpath, scene_id, f'ep-{episode_id:04d}')

    # compute map gt
    self.topdown_low_gt = self.compute_gt(
      self.topdown_low,
      self.meters_per_pixel_low,
      masking
    )
    self.topdown_high_gt = self.compute_gt(
      self.topdown_high,
      self.meters_per_pixel_high,
      masking
    )
    data.topdown_low_gt = self.topdown_low_gt
    data.topdown_high_gt = self.topdown_high_gt

    # sample charts from topdown high
    charts, chart_gts, centers = self.sample_charts(
      max_charts = max_charts,
      chart_width = chart_width,
      chart_height = chart_height
    )

    total_points = 0
    total_charts = 0

    for idx, (chart, chart_gt, center) in \
        enumerate(zip(charts, chart_gts, centers)):
      results = self.get_samples(
        chart = chart,
        chart_gt = chart_gt,
        center = center,
        min_points = min_points
      )
      # skip empty charts
      if results is None:
        continue
      chart_gt, points, dists, grads = results
      num_points = len(points)
      _data = GsmData(
        objectgoal = objectgoal,
        chart = chart,
        chart_gt = chart_gt,
        center = center,
        points = points,
        distances = dists,
        gradients = grads
      )
      filename = f'sample_{idx:06d}.npz'
      filepath = os.path.join(ep_dirpath, filename)
      _data.save(filepath)
      data.paths.append(filepath)

      total_charts += 1
      total_points += num_points

    data.info['total_charts'] = total_charts
    data.info['total_points'] = total_points

    # save episode data
    data.save(ep_dirpath)

  def compute_gt(self, topdown_map, meters_per_pixel, masking):
    height, width = topdown_map.shape
    gradient_map = np.zeros((height, width, 2), dtype=np.float32)
    distance_map = np.zeros((height, width, 1), dtype=np.float32)
    for h in range(height):
      for w in range(width):
        if (not masking) or topdown_map[h, w]:
          xp = np.asarray((w, h), dtype=np.float32).reshape((-1, 2))
          xp = self.to_3D_points(xp, meters_per_pixel)
          dist, grad = self.compute_geodesic_and_grads(xp, self.goals)
          dist = dist[0]
          grad = grad[0]
          gradient_map[h, w] = grad
          distance_map[h, w] = dist
    gt = np.concatenate((distance_map, gradient_map), axis=-1)
    return gt

  def to_2D_points(self, x, meters_per_pixel):
    x2d = (x[..., 0] - self.bounds[0][0]) / meters_per_pixel
    y2d = (x[..., 2] - self.bounds[0][2]) / meters_per_pixel

    return np.stack((x2d, y2d), axis=-1).astype(np.float32)

  def to_3D_points(self, x, meters_per_pixel, h=None):
    if h is None:
      h = self.agent_height
    
    x3d = x[..., 0] * meters_per_pixel + self.bounds[0][0]
    y3d = np.full(x3d.shape, h, dtype=np.float32)
    z3d = x[..., 1] * meters_per_pixel + self.bounds[0][2]

    return np.stack((x3d, y3d, z3d), axis=-1).astype(np.float32)

  def compute_geodesic_and_grads(
    self,
    xp: np.ndarray,
    goals: np.ndarray
  ):
    eps_x = np.asarray((self.eps, 0, 0), dtype=np.float32)
    eps_z = np.asarray((0, 0, self.eps), dtype=np.float32)

    gms = []
    dgms = []

    for _xp in xp:
      gm = self.sim.geodesic_distance(_xp, goals)
      ps_x_gm = self.sim.geodesic_distance(_xp+eps_x, goals)
      ng_x_gm = self.sim.geodesic_distance(_xp-eps_x, goals)
      ps_z_gm = self.sim.geodesic_distance(_xp+eps_z, goals)
      ng_z_gm = self.sim.geodesic_distance(_xp-eps_z, goals)
      dgm = np.asarray((ps_x_gm-ng_x_gm, ps_z_gm-ng_z_gm), dtype=np.float32)
      # enforce boundary
      gm = np.inf if np.isinf(gm) or np.isnan(gm) else gm
      dgm[np.isinf(dgm)] = 0
      dgm[np.isnan(dgm)] = 0
      dgm /= 2 * self.eps

      gms.append(gm)
      dgms.append(dgm)
    gms = np.asarray(gms, dtype=np.float32)
    dgms = np.asarray(dgms, dtype=np.float32).reshape((-1, 2))
    return gms, dgms

  def sample_charts(
    self,
    max_charts: int = 1000,
    chart_width: int = 300,
    chart_height: int = 300
  ):
    """_summary_

    Args:
        max_charts (int, optional): _description_. Defaults to 1000.
        chart_width (int, optional): _description_. Defaults to 300.
        chart_height (int, optional): _description_. Defaults to 300.

    Returns:
      np.ndarray: sampled charts (h, w)
      np.ndarray:
    """
    height, width = self.topdown_high.shape

    available_points = []
    for h in range(height):
      for w in range(width):
        if self.topdown_high[h, w]:
          available_points.append((w, h))
    available_points = np.asarray(available_points, dtype=np.float32)

    n_samples = min(max_charts, len(available_points))
    inds = np.random.randint(len(available_points), size=(n_samples,))
    centers2d = available_points[inds]

    map_th = torch.tensor(self.topdown_high, dtype=torch.float32)
    map_th = einops.repeat(map_th, 'h w -> 1 1 h w')
    map_gt_th = torch.tensor(self.topdown_high_gt, dtype=torch.float32)
    map_gt_th = einops.rearrange(map_gt_th, 'h w c -> 1 c h w')

    charts = []
    chart_gts = []
    for center in centers2d:
      grid = dmap.utils.generate_crop_grid(
        center,
        image_width = width,
        image_height = height,
        crop_width = chart_width,
        crop_height = chart_height
      )
      chart = dmap.utils.image_sample(
        image = map_th,
        grid = grid,
        mode = 'nearest'
      ) # (1, 1, h, w)
      chart_gt = dmap.utils.image_sample(
        image = map_gt_th,
        grid = grid,
        mode = 'nearest'
      )
      chart = chart.detach().cpu().numpy()[0, 0]
      charts.append(chart)
      chart_gt = chart_gt.detach().cpu().numpy()[0] # (c, h, w)
      chart_gt = einops.rearrange(chart_gt, 'c h w -> h w c')
      chart_gts.append(chart_gt)
    charts = np.stack(charts, axis=0)
    chart_gts = np.stack(chart_gts, axis=0)
    return charts, chart_gts, centers2d

  def get_samples(
    self,
    chart: np.ndarray,
    chart_gt: np.ndarray,
    center: np.ndarray,
    min_points: int = 10
  ) -> Optional[Tuple[np.ndarray, ...]]:
    height, width = chart.shape

    chart_th = torch.from_numpy(chart)
    point_inds = chart_th.nonzero().numpy() # [h, w]
    if len(point_inds) < min_points:
      # discard this sample
      return
    xp = point_inds[...,::-1].astype(np.int64) # [w, h]
    dist = chart_gt[xp[...,1], xp[...,0], 0]
    grad = chart_gt[xp[...,1], xp[...,0], 1:]

    rsize = np.asarray((width, height), dtype=np.float32)
    half = rsize / 2.
    glb_xp = xp.astype(np.float32) - half + center.astype(np.float32)
    glb_xp3d = self.to_3D_points(glb_xp, self.meters_per_pixel_high)

    # filter out invalid points
    res = []
    for idx in range(len(xp)):
      if not self.sim.pathfinder.is_navigable(glb_xp3d[idx]):
        continue
      if np.isinf(dist[idx]):
        continue
      if grad[idx][0] == 0 and grad[idx][1] == 0:
        continue
      res.append((xp[idx], dist[idx], grad[idx]))
    if len(res) < min_points:
      # discard this sample
      return

    xp, dist, grad = zip(*res)
    xp = np.asarray(xp, dtype=np.int64) # (-1, 2), [w, h]
    rsize = np.asarray((width, height), dtype=np.float32)
    norm_xp = xp.astype(np.float32) / rsize * 2.0 - 1.0 # (-1, 2)
    dist = np.asarray(dist, dtype=np.float32) # (-1,)
    grad = np.asarray(grad, dtype=np.float32) # (-1, 2)
    gt = np.concatenate((dist.reshape((-1, 1)), grad), axis=-1) # (-1, 3) [d, v0, v1]
    chart_gt = np.full((height, width, 3), np.inf, dtype=np.float32)
    chart_gt[xp[...,1], xp[...,0]] = gt
    return chart_gt, norm_xp, dist, grad
