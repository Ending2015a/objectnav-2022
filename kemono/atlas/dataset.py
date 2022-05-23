# --- built in ---
import os
import dataclasses
from dataclasses import dataclass
from typing import (
  List,
  Dict,
  Any,
  Tuple,
  Union,
  Optional
)
# --- 3rd party ---
import cv2
import habitat
import numpy as np
import dungeon_maps as dmap
import torch
import rlchemy
import einops
import matplotlib.pyplot as plt
# --- my module ---
from kemono.atlas import utils

@dataclass
class ChartData():
  """
  objectgoal: goal index, 0~6, (), np.int64
  chart: ground truth chart, (h, w), bool
  chart_gt: ground truth values (distance, gradient) of the chart, (h, w, 3),
    [dist, v0, v1], np.float32. np.inf for invalid cells
  maps: other kind of chart, e.g. agent observed maps. Stored as a key value pair
    key is the name of the map, value is the map. (c, h, w), Any
  center: chart center, [w, h], (2,), np.int64
  points: normalized points on charts [w, h], -1.0~1.0, (b, 2), np.float32
  distances: ground truth distances for each point. (b,), np.float32
  gradients: ground truth gradients for each point. (n, 2), np.float32
  """
  
  objectgoal: int = None
  chart: np.ndarray = None
  chart_gt: np.ndarray = None
  maps: Dict[str, np.ndarray] = dataclasses.field(default_factory=dict)
  center: np.ndarray = None
  points: np.ndarray = None
  distances: np.ndarray = None
  gradients: np.ndarray = None

  def save(self, filepath):
    rlchemy.utils.safe_makedirs(filepath=filepath)
    # convert self to dict
    data = dataclasses.asdict(self)
    # flatten structure
    struct, flat_data = rlchemy.utils.unpack_structure(data, sortkey=False)
    # save data
    np.savez(
      filepath,
      *flat_data,
      _struct = struct
    )

  @classmethod
  def load(cls, filepath) -> "ChartData":
    assert os.path.isfile(filepath)
    npz = np.load(filepath, allow_pickle=True)
    assert '_struct' in npz.files
    struct = npz['_struct'].item()
    npz.files.remove('_struct')
    flat_data = [npz[arr] for arr in npz.files]
    # recover chart data
    data = rlchemy.utils.pack_sequence(struct, flat_data)
    self = cls(**data)
    return self


@dataclass
class AtlasData():
  """
  topdown_low: low-res topdown map (h, w), bool
  topdown_high: high-res topdown map (h, w), bool
  topdown_low_gt: ground truth values for low-res topdown map
    (h, w, 3), np.float32
  topdown_high_gt: ground truth values for high-res topdown map
    (h, w, 3), np.float32
  paths: sampled charts path
  info: atlas information
  """
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
  def load(cls, dirpath) -> "AtlasData":
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


class AtlasSampler():
  def __init__(
    self,
    goal_mapping,
    meters_per_pixel_low: float = 0.1,
    meters_per_pixel_high: float = 0.03,
    eps: float = 0.1,
  ):
    self.goal_mapping = goal_mapping
    self.goal_mapping_inv = {v:k for k, v in goal_mapping.items()}
    self.meters_per_pixel_low = meters_per_pixel_low
    self.meters_per_pixel_high = meters_per_pixel_high
    self.eps = eps

    self.reset()

  def dmap_to_topdown_map(
    self,
    dmap_map: dmap.TopdownMap,
    bad: float = 0.0
  ):
    topdown_map = dmap_map.height_map.cpu().numpy()[0, 0]
    mask = (topdown_map == bad)
    topdown_map[mask] = 0.0
    topdown_map[~mask] = 1.0
    return topdown_map.astype(bool)

  def get_episode_path(
    self,
    episode
  ):
    episode_id = int(episode.episode_id)
    scene_id = os.path.basename(episode.scene_id).split('.')[0]
    return os.path.join(scene_id, f'ep-{episode_id:04d}')

  def reset(self):
    self.agent_height = 0.0
    self.topdown_low = None
    self.topdown_high = None
    self.bounds = None
    self.episode = None
    self.goals = None
    self._atlas = None
    self._charts = None
    self.dmap_low = None
    self.dmap_high = None
    self.start_dmap_low = None
    self.start_dmap_high = None
    self.start_topdown_low = None
    self.start_topdown_high = None
    self.start_topdown_low_gt = None
    self.start_topdown_high_gt = None


  def create_atlas(
    self,
    env: habitat.Env,
    masking: bool = True,
    compute_gt: bool = True,
    find_gt_from_path: Optional[str] = None
  ):
    """Create a new atlas dataset

    Args:
      env (habitat.Env): habitat environment
    """
    self.reset()
    self.env = env
    self.sim = env.sim

    data = AtlasData()
    self._atlas = data
    self._charts = []

    self.agent_height = self.sim.get_agent(0).state.position[1]
    self.bounds = self.sim.pathfinder.get_bounds()
    # get topdown map wrapped in dungeon_maps.TopdownMap object
    # for coordinate transform
    self.dmap_low = utils.get_topdown_map(
      self.env,
      self.meters_per_pixel_low/2.,
      device = 'cuda'
    )
    self.dmap_high = utils.get_topdown_map(
      self.env,
      self.meters_per_pixel_high/2.,
      device = 'cuda'
    )
    # dungeon_maps.TopdownMap convert to global view
    self.start_dmap_low = utils.topdown_map_to_global_space(
      self.dmap_low,
      self.meters_per_pixel_low
    )
    self.start_dmap_high = utils.topdown_map_to_global_space(
      self.dmap_high,
      self.meters_per_pixel_high
    )
    self.start_topdown_low = self.dmap_to_topdown_map(
      self.start_dmap_low,
      bad = -np.inf
    )
    self.start_topdown_high = self.dmap_to_topdown_map(
      self.start_dmap_high,
      bad = -np.inf
    )

    data.info['masking'] = masking
    data.info['agent_height'] = self.agent_height
    data.info['bounds'] = self.bounds
    data.info['meters_per_pixel_low'] = self.meters_per_pixel_low
    data.info['meters_per_pixel_high'] = self.meters_per_pixel_high
    data.info['eps'] = self.eps
    data.topdown_low = self.start_topdown_low.copy()
    data.topdown_high = self.start_topdown_high.copy()

    episode = self.env.current_episode
    self.episode = episode
    # goals view points
    self.goals = np.asarray([
      view_point.agent_state.position
      for goal in episode.goals
      for view_point in goal.view_points
    ])
    episode_id = int(episode.episode_id)
    scene_id = os.path.basename(episode.scene_id).split('.')[0]
    object_category = episode.object_category
    objectgoal = self.goal_mapping_inv[object_category]
    goals_pos = np.asarray([goal.position for goal in episode.goals])

    data.info['episode_id'] = episode_id
    data.info['scene_id'] = scene_id
    data.info['object_category'] = object_category
    data.info['objectgoal'] = objectgoal
    data.info['goals'] = goals_pos.copy().tolist()
    data.info['view_goals'] = self.goals.copy().tolist()
    data.info['episode_path'] = self.get_episode_path(episode)

    # find ground truth map from path
    if find_gt_from_path is not None:
      path = find_gt_from_path
      episode_path = self.get_episode_path(episode)
      fullpath = os.path.join(path, episode_path)
      if os.path.exists(fullpath):
        data_gt = AtlasData.load(fullpath)
        if data_gt.topdown_low_gt is not None:
          self.start_topdown_low_gt = data_gt.topdown_low_gt
        if data_gt.topdown_high_gt is not None:
          self.start_topdown_high_gt = data_gt.topdown_high_gt

    # compute ground truth map
    if compute_gt:
      if self.start_topdown_low_gt is None:
        self.start_topdown_low_gt = self.compute_ground_truth(
          self.start_topdown_low,
          self.start_dmap_low,
          self.dmap_low,
          masking
        )
      if self.start_topdown_high_gt is None:
        self.start_topdown_high_gt = self.compute_ground_truth(
          self.start_topdown_high,
          self.start_dmap_high,
          self.dmap_high,
          masking
        )
    data.topdown_low_gt = self.start_topdown_low_gt
    data.topdown_high_gt = self.start_topdown_high_gt

  def add_chart(
    self,
    center3d: np.ndarray,
    min_points: int = 10,
    chart_width: int = 300,
    chart_height: int = 300,
    maps: Dict[str, Any] = {},
  ):
    assert self._atlas is not None
    center3d = np.asarray(center3d, dtype=np.float32)
    center2d = self.start_dmap_high.get_coords(center3d, is_global=True) # (1, 1, 2)
    center = center2d.cpu().numpy()[0, 0]
    # crop ground truth maps
    chart, chart_gt = self._crop_topdown_maps(
      center = center,
      crop_width = chart_width,
      crop_height = chart_height,
      topdown_map = self.start_topdown_high,
      topdown_map_gt = self.start_topdown_high_gt
    )
    # sample points from ground truth maps
    results = self._sample_points(
      chart = chart,
      chart_gt = chart_gt,
      min_points = min_points
    )
    # skip empty charts
    if results is None:
      return
    chart_gt, points, dists, grads = results
    num_points = len(points)
    data = ChartData(
      objectgoal = self._atlas.info['objectgoal'],
      chart = chart.astype(bool),
      chart_gt = chart_gt.astype(np.float32),
      maps = maps,
      center = center.astype(np.int64),
      points = points.astype(np.float32),
      distances = dists.astype(np.float32),
      gradients = grads.astype(np.float32),
    )
    self._charts.append(data)

    total_points = self._atlas.info.get('total_points', 0)
    total_charts = self._atlas.info.get('total_charts', 0)

    total_charts += 1
    total_points += num_points

    self._atlas.info['total_points'] = total_points
    self._atlas.info['total_charts'] = total_charts

  def sample_charts(
    self,
    max_charts: int = 1000,
    min_points: int = 10,
    chart_width: int = 300,
    chart_height: int = 300
  ):
    """_summary_

    Args:
      max_charts (int, optional): _description_. Defaults to 1000.
      min_points (int, optional): charts containing points less than this number
        is discarded. Defaults to 10.
      chart_width (int, optional): _description_. Defaults to 300.
      chart_height (int, optional): _description_. Defaults to 300.
    """
    assert self._atlas is not None
    data = self._atlas

    # sample charts from topdown high
    charts, chart_gts, centers = self._sample_charts(
      max_charts = max_charts,
      chart_width = chart_width,
      chart_height = chart_height
    )

    total_points = data.info.get('total_points', 0)
    total_charts = data.info.get('total_charts', 0)

    for chart, chart_gt, center in zip(charts, chart_gts, centers):
      results = self._sample_points(
        chart = chart,
        chart_gt = chart_gt,
        min_points = min_points
      )
      # skip empty charts
      if results is None:
        continue
      chart_gt, points, dists, grads = results
      num_points = len(points)
      _data = ChartData(
        objectgoal = data.info['objectgoal'],
        chart = chart.astype(bool),
        chart_gt = chart_gt.astype(np.float32),
        center = center.astype(np.int64),
        points = points.astype(np.float32),
        distances = dists.astype(np.float32),
        gradients = grads.astype(np.float32)
      )
      self._charts.append(_data)

      total_charts += 1
      total_points += num_points

    data.info['total_charts'] = total_charts
    data.info['total_points'] = total_points

  def save_atlas(
    self,
    dirpath
  ):
    assert self._atlas is not None
    data = self._atlas
    data.info['dirpath'] = dirpath
    atlas_path = os.path.join(dirpath, data.info['episode_path'])
    for idx, chart in enumerate(self._charts):
      filename = f'sample_{idx:06d}.npz'
      chart_path = os.path.join(atlas_path, filename)
      chart.save(chart_path)
      data.paths.append(chart_path)
    data.save(atlas_path)

  def load_atlas(
    self,
    dirpath: str,
    gt_only: bool = False
  ):
    """Load atlas from the directory

    Args:
      dirpath (str): directory path.
      gt_only (bool, optional): load ground truth maps only.
        Defaults to False.
    """
    data = AtlasData.load(dirpath)
    if gt_only:
      assert self._atlas is not None
      self.start_topdown_low_gt = data.topdown_low_gt
      self.start_topdown_high_gt = data.topdown_high_gt
    else:
      self._atlas = data

  def _sample_charts(
    self,
    max_charts: int = 1000,
    chart_width: int = 300,
    chart_height: int = 300
  ):
    height, width = self.start_topdown_high.shape
    
    available_points = []
    for h in range(height):
      for w in range(width):
        if self.start_topdown_high[h, w]:
          available_points.append((w, h))
    available_points = np.asarray(available_points, dtype=np.float32)

    n_samples = min(max_charts, len(available_points))
    inds = np.random.randint(len(available_points), size=(n_samples,))
    centers2d = available_points[inds]

    map_th = torch.tensor(self.start_topdown_high, dtype=torch.float32)
    map_th = einops.repeat(map_th, 'h w -> 1 1 h w')
    map_gt_th = torch.tensor(self.start_topdown_high_gt, dtype=torch.float32)
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

  def _crop_topdown_maps(
    self,
    center: np.ndarray,
    crop_width: int,
    crop_height: int,
    topdown_map: np.ndarray,
    topdown_map_gt: np.ndarray
  ) -> Tuple[np.ndarray, np.ndarray]:
    assert np.array_equal(topdown_map.shape, topdown_map_gt.shape[:-1])
    height, width = topdown_map.shape

    map_th = torch.tensor(topdown_map, dtype=torch.float32)
    map_th = einops.repeat(map_th, 'h w -> 1 1 h w')
    map_gt_th = torch.tensor(topdown_map_gt, dtype=torch.float32)
    map_gt_th = einops.rearrange(map_gt_th, 'h w c -> 1 c h w')

    grid = dmap.utils.generate_crop_grid(
      center,
      image_width = width,
      image_height = height,
      crop_width = crop_width,
      crop_height = crop_height
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
    chart_gt = chart_gt.detach().cpu().numpy()[0] # (c, h, w)
    chart_gt = einops.rearrange(chart_gt, 'c h w -> h w c')
    return chart, chart_gt

  def _sample_points(
    self,
    chart: np.ndarray,
    chart_gt: np.ndarray,
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
    # filter out invalid points
    res = []
    for idx in range(len(xp)):
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

  # ===

  def compute_ground_truth(
    self,
    topdown_map: np.ndarray,
    start_dmap_map: dmap.TopdownMap,
    dmap_map: dmap.TopdownMap,
    masking: bool = True
  ):
    plt.figure(figsize=(6, 6), dpi=300)
    plt.imshow(topdown_map, cmap='hot', interpolation='nearest')

    height, width = topdown_map.shape
    grad_map = np.zeros((height, width, 2), dtype=np.float32)
    dist_map = np.zeros((height, width, 1), dtype=np.float32)
    # find all available pixels on manifold
    coords = []
    for h in range(height):
      for w in range(width):
        if (not masking) or topdown_map[h, w]:
          coords.append((w, h))
    coords = np.array(coords, dtype=np.float32) # (n, 2)
    # convert to global coordinates (start position/rotation coordinates)
    points_xz_th = start_dmap_map.get_points(coords) # (1, n, 2)
    # for computing directional derivative (finite-difference method)
    # note that the finite-difference must be in start position/rotation
    # coordinates, not in the habitat simulator coordinates
    eps_x = torch.tensor((self.eps, 0)).type_as(points_xz_th)
    # note that the global z coordinate is inverted in dungeon_maps'
    # coordinate system
    eps_z = torch.tensor((0, -self.eps)).type_as(points_xz_th)
    pos_x = points_xz_th + eps_x
    neg_x = points_xz_th - eps_x
    pos_z = points_xz_th + eps_z
    neg_z = points_xz_th - eps_z
    points_xz_th = torch.cat(
      (points_xz_th, pos_x, neg_x, pos_z, neg_z),
      dim=0
    ) # (5, n, 2)
    points_xz_th = einops.rearrange(points_xz_th, 'b n c -> 1 (b n) c')
    points_y_th = torch.full_like(points_xz_th[...,0], self.agent_height)
    points_xyz_th = torch.stack(
      (points_xz_th[...,0], points_y_th, points_xz_th[...,1]),
      dim=-1
    ) # (1, 5*n, 3)
    # convert to world coordinates (habitat simulator coordinates)
    # firstly we convert the 3D world coordinates to the image coordinates
    # then we convert the image coordinates to habitat simulator coordinates
    points_xyz_th = dmap_map.proj.global_to_local_space(points=points_xyz_th) # (1, b*n, 3)
    points_x_th = points_xyz_th[...,0]
    points_z_th = points_xyz_th[...,2]
    points_x_th = points_x_th / dmap_map.proj.map_res + dmap_map.proj.width_offset
    points_z_th = points_z_th / dmap_map.proj.map_res + dmap_map.proj.height_offset
    if dmap_map.proj.flip_h:
      map_height = torch.tensor(dmap_map.proj.map_height).type_as(points_z_th)
      points_z_th = map_height - 1 - points_z_th
    # convert the image coordinates to habitat simulator coordinates
    points_x_th = points_x_th * dmap_map.proj.map_res + self.bounds[0][0]
    points_z_th = points_z_th * dmap_map.proj.map_res + self.bounds[0][2]
    points_th = torch.stack((points_x_th, points_y_th, points_z_th), dim=-1) # (1, b*n, 2)
    # finalize
    points_th = einops.rearrange(points_th, '1 (b n) c -> n b c', b=5)
    points = points_th.cpu().numpy() # (n, b, 3)
    # calculate geodesic distance and directional derivative for each point
    for coord, point in zip(coords, points):
      w, h = coord.astype(np.int64)
      dist, grad = self.compute_geodesic_and_grads(point, self.goals)
      grad_map[h, w] = grad
      dist_map[h, w] = dist
    gt = np.concatenate((dist_map, grad_map), axis=-1)
    return gt

  def compute_geodesic_and_grads(
    self,
    xp: np.ndarray,
    goals: np.ndarray
  ):
    gm = self.sim.geodesic_distance(xp[0], goals)
    ps_x_gm = self.sim.geodesic_distance(xp[1], goals)
    ng_x_gm = self.sim.geodesic_distance(xp[2], goals)
    ps_z_gm = self.sim.geodesic_distance(xp[3], goals)
    ng_z_gm = self.sim.geodesic_distance(xp[4], goals)
    dgm = np.asarray((ps_x_gm-ng_x_gm, ps_z_gm-ng_z_gm), dtype=np.float32)
    # enforce boundary
    gm = np.inf if np.isinf(gm) or np.isnan(gm) else gm
    dgm[np.isinf(dgm)] = 0
    dgm[np.isnan(dgm)] = 0
    dgm /= 2 * self.eps

    return gm, dgm
