# --- built in ---
from dataclasses import dataclass, asdict
from typing import (
  Any,
  Optional,
  List,
  Dict,
  Union,
  Tuple
)
# --- 3rd party ---
import cv2
import gym
import torch
from torch import nn
import habitat
import numpy as np
import dungeon_maps as dmap
# --- my module ---

__all__ = [
  "MapBuilderEnv"
]

# Colors [b, g, r]
hex2rgb = lambda hex: [int(hex[i:i+2], 16) for i in (1, 3, 5)]
FLOOR_COLOR   = hex2rgb('#90D5C3')
WALL_COLOR    = hex2rgb('#6798D0')
INVALID_COLOR = hex2rgb('#F4F7FA')
CAMERA_COLOR  = hex2rgb('#EC5565')
ORIGIN_COLOR  = hex2rgb('#F49FBC')
GOAL_COLOR    = hex2rgb('#FFC300')
TRAJECTORY_COLOR = hex2rgb('#EC5565')

HEIGHT_THRESHOLD = 0.25

def draw_map(topdown_map: dmap.TopdownMap):
  occ_map = draw_occlusion_map(topdown_map.height_map, topdown_map.mask)
  occ_map = draw_origin(occ_map, topdown_map)
  occ_map = draw_camera(occ_map, topdown_map)
  return occ_map

def draw_occlusion_map(height_map, mask):
  """Draw occulution map: floor, wall, invalid area
  Args:
      height_map (torch.Tensor, np.ndarray): height map (b, c, h, w).
      mask (torch.Tensor, np.ndarray): mask (b, c, h, w).
  """
  height_map = dmap.utils.to_numpy(height_map[0, 0]) # (h, w)
  mask = dmap.utils.to_numpy(mask[0, 0]) # (h, w)
  height_threshold = HEIGHT_THRESHOLD
  floor_area = (height_map <= height_threshold) & mask
  wall_area = (height_map > height_threshold) & mask
  invalid_area = ~mask
  topdown_map = np.full(
    height_map.shape + (3,),
    fill_value=255, dtype=np.uint8
  ) # canvas (h, w, 3)
  topdown_map[invalid_area] = INVALID_COLOR
  topdown_map[floor_area] = FLOOR_COLOR
  topdown_map[wall_area] = WALL_COLOR
  return topdown_map

def draw_origin(
  image: np.ndarray,
  topdown_map: dmap.TopdownMap,
  color: np.ndarray = ORIGIN_COLOR,
  size: int = 4
):
  assert len(image.shape) == 3 # (h, w, 3)
  assert image.dtype == np.uint8
  assert topdown_map.proj is not None
  pos = np.array([
    [0., 0., 0.], # camera position
    [0., 0., 1.], # forward vector
    [0., 0., -1], # backward vector
    [-1, 0., 0.], # left-back vector
    [1., 0., 0.], # right-back vector
  ], dtype=np.float32)
  pos = topdown_map.get_coords(pos, is_global=True) # (b, 5, 2)
  pos = dmap.utils.to_numpy(pos)[0] # (5, 2)
  return draw_diamond(image, pos, color=color, size=size)

def draw_goal(
  image: np.ndarray,
  topdown_map: dmap.TopdownMap,
  position: np.ndarray,
  color: np.ndarray = GOAL_COLOR,
  size: int = 3
):
  assert len(image.shape) == 3 # (h, w, 3)
  assert image.dtype == np.uint8
  assert topdown_map.proj is not None
  pos = np.array([
    [0., 0., 0.], # camera position
    [0., 0., 1.], # forward vector
    [0., 0., -1], # backward vector
    [-1, 0., 0.], # left-back vector
    [1., 0., 0.], # right-back vector
  ], dtype=np.float32) + position
  pos = topdown_map.get_coords(pos, is_global=True) # (b, 5, 2)
  pos = dmap.utils.to_numpy(pos)[0] # (5, 2)
  return draw_diamond(image, pos, color=color, size=size)

def draw_camera(
  image: np.ndarray,
  topdown_map: dmap.TopdownMap,
  color: np.ndarray = CAMERA_COLOR,
  size: int = 6
):
  assert len(image.shape) == 3 # (h, w, 3)
  assert image.dtype == np.uint8
  assert topdown_map.proj is not None
  pos = np.array([
    [0., 0., 0.], # camera position
    [0., 0., 1.], # forward vector
    [-1, 0., -1], # left-back vector
    [1., 0., -1], # right-back vector
  ], dtype=np.float32)
  pos = topdown_map.get_coords(pos, is_global=False) # (b, 4, 2)
  pos = dmap.utils.to_numpy(pos)[0] # (4, 2)
  return draw_arrow(image, pos, color=color, size=size)

def draw_trajectory(
  image: np.ndarray,
  topdown_map: dmap.TopdownMap,
  trajectory: np.ndarray,
  color: np.ndarray = TRAJECTORY_COLOR,
  size: int = 2
):
  assert len(image.shape) == 3 # (h, w, 3)
  assert image.dtype == np.uint8
  assert topdown_map.proj is not None
  pos = np.asarray(trajectory, dtype=np.float32)
  pos = topdown_map.get_coords(pos, is_global=True)
  pos = dmap.utils.to_numpy(pos)[0]
  return draw_segments(image, pos, color=color, size=size)

def draw_arrow(image, points, color, size=2):
  # points: [center, forward, left, right]
  norm = lambda p: p/np.linalg.norm(p)
  c = points[0]
  f = norm(points[1] - points[0]) * (size*2) + points[0]
  l = norm(points[2] - points[0]) * (size*2) + points[0]
  r = norm(points[3] - points[0]) * (size*2) + points[0]
  pts = np.asarray([f, l, c, r], dtype=np.int32)
  return cv2.fillPoly(image, [pts], color=color)

def draw_diamond(image, points, color, size=2):
  # points [center, forward, back, left, right]
  norm = lambda p: p/np.linalg.norm(p)
  c = points[0]
  f = norm(points[1] - points[0]) * (size*2) + points[0]
  b = norm(points[2] - points[0]) * (size*2) + points[0]
  l = norm(points[3] - points[0]) * (size*2) + points[0]
  r = norm(points[4] - points[0]) * (size*2) + points[0]
  pts = np.asarray([f, l, b, r], dtype=np.int32)
  return cv2.fillPoly(image, [pts], color=color)

def draw_mark(image, point, color, size=2):
  radius = size
  thickness = radius + 2
  image = cv2.circle(image, (int(point[0]), int(point[1])),
      radius=radius, color=color, thickness=thickness)
  return image

def draw_segments(image, points, color, size=2):
  for index in range(1, len(points)):
    prev = index - 1
    cur = index
    if np.all(points[prev] == points[cur]):
      continue
    image = draw_segment(image, points[prev], points[cur], color, size)
  return image

def draw_segment(image, p1, p2, color, size=2):
  image = cv2.line(image, (p1[0], p1[1]), (p2[0], p2[1]),
    color=color, thickness=size)
  return image

def compute_relative_goals(init_pos, init_rot, goal_pos):
  from habitat.utils.geometry_utils import (
    quaternion_rotate_vector,
    quaternion_from_coeff
  )
  init_rot = quaternion_from_coeff(init_rot)
  goal_pos = quaternion_rotate_vector(
    init_rot.inverse(), goal_pos - init_pos
  )
  return np.asarray([goal_pos[0], goal_pos[1], -goal_pos[2]])

def dilate_tensor(x, size, iterations=1):
  if isinstance(size, int):
    padding = size // 2
  else:
    padding = tuple([v // 2 for v in size])
  for i in range(iterations):
    x = nn.functional.max_pool2d(x, size, stride=1, padding=padding)
  return x

def erode_tensor(x, size, iterations=1):
  if isinstance(size, int):
    padding = size // 2
  else:
    padding = tuple([v // 2 for v in size])
  for i in range(iterations):
    x = -nn.functional.max_pool2d(-x, size, stride=1, padding=padding)
  return x

def smooth_tensor(
  x,
  dilate_size = 3,
  dilate_iter = 0,
  erode_size = 3,
  erode_iter = 0,
):
  if dilate_size > 0 and dilate_iter > 0:
    x = dilate_tensor(x, dilate_size, dilate_iter)
  if erode_size > 0 and erode_iter > 0:
    x = erode_tensor(x, erode_size, erode_iter)
  return x

class MapBuilder():
  def __init__(
    self,
    config: habitat.Config,
    map_res: float = 0.03,
    map_width: int = 600,
    map_height: int = 600,
    trunc_height: float = 3.0
  ):
    self.map_res = map_res
    self.map_width = map_width
    self.map_height = map_height
    self.trunc_height = trunc_height
    self.min_depth = config.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH
    self.max_depth = config.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH
    self.proj = dmap.MapProjector(
      width = config.SIMULATOR.DEPTH_SENSOR.WIDTH,
      height = config.SIMULATOR.DEPTH_SENSOR.HEIGHT,
      hfov = np.radians(config.SIMULATOR.DEPTH_SENSOR.HFOV),
      vfov = None,
      cam_pose = [0., 0., 0.],
      width_offset = 0.,
      height_offset = 0.,
      cam_pitch = np.radians(config.SIMULATOR.DEPTH_SENSOR.ORIENTATION[1]),
      cam_height = config.SIMULATOR.AGENT_0.HEIGHT,
      map_res = map_res,
      map_width = map_width,
      map_height = map_height,
      trunc_depth_min = self.min_depth * 1.05,
      trunc_depth_max = self.max_depth * 0.95,
      trunc_height_max = trunc_height,
      clip_border = 10,
      fill_value = -np.inf,
      to_global = True,
      device = 'cuda'
    )
    self.builder = dmap.MapBuilder(
      map_projector = self.proj
    )
  
  def reset(self):
    self.builder.reset()

  def step(self, obs):
    depth_map = obs['depth']
    # denormalize depth map [0, 1] -> [min_depth, max_depth]
    depth_map = depth_map * (self.max_depth - self.min_depth) + self.min_depth
    depth_map = torch.tensor(depth_map, device='cuda')
    depth_map = depth_map.permute(2, 0, 1) # hwc -> chw
    cam_pose = [obs['gps'][1], obs['gps'][0], obs['compass'][0]]
    cam_pitch = obs['compass'][1]
    local_map = self.builder.step(
      depth_map = depth_map,
      cam_pose = cam_pose,
      cam_pitch = cam_pitch,
      to_global = False,
      map_res = self.map_res,
      width_offset = self.map_width/2.,
      height_offset = 0.,
      map_width = self.map_width,
      map_height = self.map_height,
      center_mode = dmap.CenterMode.none,
      merge = False,
    )
    self.builder.merge(local_map, keep_pose=False)
    return local_map

  @property
  def world_map(self):
    cam_pos = self.builder.world_map.get_camera()
    world_map = self.builder.world_map.select(
      cam_pos, self.map_width, self.map_height
    )
    return world_map

class MapBuilderWrapper(gym.Wrapper):
  def __init__(
    self,
    env: habitat.RLEnv,
    draw_goals: bool = False
  ):
    super().__init__(env=env)
    config = env.config
    self.map_builder = MapBuilder(
      config = config
    )
    self.draw_goals = draw_goals
    self.local_map_key = "local_map"
    self.world_map_key = "world_map"
    self._cached_obs = None
    self._agent_init_pos = None
    self._goals = None

  def step(self, action):
    obs, rew, done, info = self.env.step(action)
    obs = self.get_observations(obs)
    self._cached_obs = obs
    print(self._goals[0])
    return obs, rew, done, info
  
  def reset(self):
    obs = self.env.reset()
    self.map_builder.reset()
    if self.draw_goals:
      # cache goals
      self._goals = self.get_goals()
    obs = self.get_observations(obs)
    self._cached_obs = obs
    return obs

  def get_observations(self, obs):
    # update map builder
    local_map = self.map_builder.step(obs)
    world_map = self.map_builder.world_map
    local_map_image = draw_map(local_map)
    world_map_image = draw_map(world_map)
    if self.draw_goals:
      for goal in self._goals:
        local_map_image = draw_goal(local_map_image, local_map, goal)
        world_map_image = draw_goal(world_map_image, world_map, goal)
    obs[self.local_map_key] = local_map_image
    obs[self.world_map_key] = world_map_image
    return obs

  def get_goals(self):
    init_pos = self.env.habitat_env.current_episode.start_position
    init_pos = np.asarray(init_pos)
    init_rot = self.env.habitat_env.current_episode.start_rotation
    init_rot = np.asarray(init_rot)
    goals = self.env.habitat_env.current_episode.goals
    goal_pos = [
      compute_relative_goals(init_pos, init_rot, goal.position)
      for goal in goals
    ]
    return np.asarray(goal_pos)

  def render(self, mode="human"):
    res = self.env.render(mode=mode)
    if mode == 'human' or mode == 'interact':
      cv2.imshow("local_map", self._cached_obs[self.local_map_key])
      cv2.imshow("world_map", self._cached_obs[self.world_map_key])
    return res

@dataclass
class CropConfig:
  center: Union[str, list] = 'camera'
  width: int = 300
  height: int = 300


@dataclass
class SmoothConfig:
  dilate_size: int = 3
  dilate_iter: int = 0
  erode_size: int = 3
  erode_iter: int = 0

@dataclass
class ColorizeConfig:
  draw_origin: bool = False
  draw_camera: bool = False
  draw_trajectory: bool = False
  draw_goals: bool = False
  draw_planner_goal: bool = False

@dataclass
class MapConfig:
  type: str = 'world'
  crop: Optional[CropConfig] = None
  smooth: Optional[SmoothConfig] = None
  colorize: Optional[ColorizeConfig] = None
  def __post_init__(self):
    if self.crop is not None:
      self.crop = CropConfig(**self.crop)
    if self.smooth is not None:
      self.smooth = SmoothConfig(**self.smooth)
    else:
      self.smooth = SmoothConfig()
    if self.colorize is not None:
      self.colorize = ColorizeConfig(**self.colorize)

class SemanticMapBuilder():
  depth_key: str = 'depth'
  seg_key: str = 'seg'
  obstacle_channel: int = 0
  walkable_channel: int = 1
  layer_channel_start: int = 2
  goal_channel: int = -1
  def __init__(
    self,
    config: habitat.Config,
    local_config: dict,
    world_config: dict,
    num_classes: int,
    walkable_labels: List[int],
    ignore_labels: List[int],
    layers_labels: List[List[int]],
    color_palette: List[str]
  ):
    self.local_config = local_config
    self.world_config = world_config
    self.num_classes = num_classes
    self.walkable_labels = walkable_labels
    self.ignore_labels = ignore_labels
    self.layers_labels = layers_labels
    self.color_palette = color_palette
    # 4 contains: background + obstacles + walkable + goal
    assert len(color_palette) == len(self.layers_labels) + 4
    # -1: goal
    # 0: obstacles
    # 1: walkable
    # 2~n: layer 0~
    self.channels = len(self.layers_labels) + 3
    self.min_depth = config.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH
    self.max_depth = config.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH
    self.proj = dmap.MapProjector(
      width = config.SIMULATOR.DEPTH_SENSOR.WIDTH,
      height = config.SIMULATOR.DEPTH_SENSOR.HEIGHT,
      hfov = np.radians(config.SIMULATOR.DEPTH_SENSOR.HFOV),
      vfov = None,
      cam_pose = [0., 0., 0.],
      cam_pitch = np.radians(config.SIMULATOR.DEPTH_SENSOR.ORIENTATION[1]),
      cam_height = config.SIMULATOR.AGENT_0.HEIGHT,
      trunc_depth_min = self.min_depth * 1.05,
      trunc_depth_max = self.max_depth * 0.95,
      **self.world_config
    )
    self.builder = dmap.MapBuilder(
      map_projector = self.proj
    )
    self._oh_mask = None
    self._valid_mask = None
    self._world2local_map = None
  
  def _create_onehot_mask(self, goal_id: int):
    # 0: obstacles
    # 1: walkable
    # 2: goal
    # 3...: layer 0...
    self._oh_mask = np.zeros((self.num_classes, self.channels), dtype=np.float32)
    self._oh_mask[..., self.obstacle_channel] = 1
    for cat_id in self.walkable_labels:
      self._oh_mask[cat_id, self.obstacle_channel] = 0
      self._oh_mask[cat_id, self.walkable_channel] = 1
    for cat_id in self.ignore_labels:
      self._oh_mask[cat_id, :] = 0
    for layer_id, labels in enumerate(self.layers_labels):
      for cat_id in labels:
        channel_id = layer_id + self.layer_channel_start
        self._oh_mask[cat_id, channel_id] = 1
    # set goal channel
    self._oh_mask[goal_id, self.goal_channel] = 1
  
  def _create_valid_mask(self):
    self._valid_mask = np.ones((self.num_classes,), dtype=bool)
    for cat_id in self.ignore_labels:
      self._valid_mask[cat_id] = False

  def reset(self, goal_id: int):
    """Reset map builder, goal_id is the mpcat40 ID of the goal"""
    self.builder.reset()
    self._create_onehot_mask(goal_id)
    self._create_valid_mask()
    self._world2local_map = None

  def step(self, obs: dict):
    # create depth map
    depth_map = obs[self.depth_key]
    # denormalize depth map [0, 1] -> [min_depth, max_depth]
    depth_map = depth_map * (self.max_depth - self.min_depth) + self.min_depth
    depth_map = torch.tensor(depth_map, device='cuda')
    depth_map = depth_map.permute(2, 0, 1) # (c, h, w)
    # create value map
    seg_map = obs[self.seg_key] # (h, w), max=num_classes
    value_map = self._oh_mask[seg_map] # (h, w, channels)
    value_map = torch.tensor(value_map, device='cuda')
    value_map = value_map.permute(2, 0, 1) # (c, h, w)
    # create valid map
    valid_map = self._valid_mask[seg_map] # (h, w)
    valid_map = torch.tensor(valid_map, device='cuda')
    
    cam_pose = [obs['gps'][1], obs['gps'][0], obs['compass'][0]]
    cam_pitch = obs['compass'][1]
    local_map = self.builder.step(
      depth_map = depth_map,
      value_map = value_map,
      valid_map = valid_map,
      cam_pose = cam_pose,
      cam_pitch = cam_pitch,
      merge = False,
      **self.local_config
    )
    self.builder.merge(local_map, keep_pose=False)
    self._world2local_map = dmap.fuse_topdown_maps(
        self.world_map,
        map_projector = self.world_map.proj.clone(
          cam_pose = local_map.proj.cam_pose,
          to_global = False
        )
      )
    return local_map

  @property
  def world_map(self) -> dmap.TopdownMap:
    return self.builder.world_map

  @property
  def world2local_map(self) -> dmap.TopdownMap:
    return self._world2local_map

  def get_value_map(
    self,
    topdown_map: dmap.TopdownMap,
    colorize: bool = False,
    **smooth_kwargs
  ) -> np.ndarray:
    value_map = topdown_map.topdown_map
    value_map = smooth_tensor(value_map, **smooth_kwargs)
    value_map = dmap.utils.to_numpy(value_map[0]) # (c, h, w)
    if colorize:
      h, w = value_map.shape[-2:]
      canvas = np.zeros((h, w, 3), dtype=np.uint8)
      # reset canvas to backgroun color
      bg_color = hex2rgb(self.color_palette[0])
      canvas[..., :] = bg_color
      # coloring each layers
      for ch_id in range(value_map.shape[0]):
        color = hex2rgb(self.color_palette[ch_id+1])
        canvas[value_map[ch_id] > 0.0] = color
      value_map = canvas # (h, w, 3)
    else:
      value_map = np.transpose(value_map, (1, 2, 0)) # (h, w, c)
    return value_map

class SemanticMapBuilderWrapper(gym.Wrapper):
  objectgoal_key = 'objectgoal'
  def __init__(
    self,
    env: habitat.RLEnv,
    world_config: dict,
    local_config: dict,
    num_classes: int,
    walkable_labels: List[int],
    ignore_labels: List[int],
    layers_labels: List[List[int]],
    color_palette: List[str],
    goal_mapping: List[Dict[int, str]]
  ):
    super().__init__(env=env)
    config = env.config
    self.map_builder = SemanticMapBuilder(
      config = config,
      world_config = world_config,
      local_config = local_config,
      num_classes = num_classes,
      walkable_labels = walkable_labels,
      ignore_labels = ignore_labels,
      layers_labels = layers_labels,
      color_palette = color_palette
    )
    self.color_palette = color_palette
    self.goal_mapping = goal_mapping
    self._cached_maps = None
    self._goals = None
    from kemono.semantics import CategoryMapping
    self._category_mapping = CategoryMapping()
    self._cached_local_map = None
    self._cached_world_map = None
    self._cached_world2local_map = None

  def get_local_map(self):
    return self._cached_local_map

  def get_world_map(self):
    return self._cached_world_map
  
  def get_world2local_map(self):
    return self._cached_world2local_map

  def get_map_builder(self):
    return self.map_builder

  def step(self, action, *args, **kwargs):
    # step environment
    obs, rew, done, info = self.env.step(action, *args, **kwargs)
    # render observations
    obs = self.get_observations(obs)
    return obs, rew, done, info

  def reset(self, *args, **kwargs):
    # reset environment
    obs = self.env.reset(*args, **kwargs)
    objectgoal = obs[self.objectgoal_key]
    # parse goal id to mpcat40 id
    goal_name = self.goal_mapping[np.asarray(objectgoal).item()]
    goal_id = self._category_mapping.get_mpcat40_id_by_category_name(goal_name)
    # reset map builder
    self.map_builder.reset(goal_id)
    obs = self.get_observations(obs)
    return obs

  def get_observations(self, obs):
    # update map builder
    local_map = self.map_builder.step(obs)
    world_map = self.map_builder.world_map
    world2local_map = self.map_builder.world2local_map
    self._cached_local_map = local_map
    self._cached_world_map = world_map
    self._cached_world2local_map = world2local_map
    return obs

  def render(self, mode="human"):
    return self.env.render(mode=mode)


class SemanticMapObserver(gym.Wrapper):
  objectgoal_key = 'objectgoal'
  gps_key = 'gps'
  def __init__(
    self,
    env: habitat.RLEnv,
    maps_config: dict,
    render_maps: List[str],
    enable_goals: bool = False,
  ):
    super().__init__(env=env)
    self.maps_config = self._structure_maps_config(maps_config)
    self.render_maps = render_maps
    self.enable_goals = enable_goals
    self._cached_maps = None
    self._cached_color_maps = None
    self._cached_trajectory = []
    self._goals = None
    self.observation_space = self.make_observation_space()
    assert hasattr(self.env, 'get_map_builder')
    self.map_builder = self.env.get_map_builder()

  def get_cached_map(self, name: str) -> dmap.TopdownMap:
    if name in self._cached_maps:
      return self._cached_maps[name]
    if hasattr(self.env, 'get_cached_map'):
      return self.env.get_cached_map(name)
    return None

  def _structure_maps_config(self, _map_configs):
    maps_config = dict()
    for map_name, map_config in _map_configs.items():
      maps_config[map_name] = MapConfig(**map_config)
    return maps_config

  def step(self, action, *args, **kwargs):
    # step environment
    obs, rew, done, info = self.env.step(action, *args, **kwargs)
    # cache agent positions
    gps = obs[self.gps_key]
    self._cached_trajectory.append((gps[1], 0, gps[0]))
    # render observations
    obs = self.get_observations(obs)
    return obs, rew, done, info

  def reset(self, *args, **kwargs):
    # reset environment
    obs = self.env.reset(*args, **kwargs)
    gps = obs[self.gps_key]
    # cache agent positions
    self._cached_trajectory = []
    self._cached_trajectory.append((gps[1], 0, gps[0]))
    if self.enable_goals:
      # cache goals
      self._goals = self.get_goals()
    obs = self.get_observations(obs)
    return obs

  def get_observations(self, obs):
    # get base maps from SemanticMapBuilderWrapper
    local_map = self.env.get_local_map()
    world_map = self.env.get_world_map()
    world2local_map = self.env.get_world2local_map()
    # render observations from config
    self._cached_maps = {}
    self._cached_color_maps = {}
    for map_name, map_config in self.maps_config.items():
      if map_config.type == 'local':
        base_map = local_map
      elif map_config.type == 'world2local':
        base_map = world2local_map
      else:
        base_map = world_map
      base_map = self.generate_map(map_config, base_map)
      value_map = self.colorize_map(map_config, base_map)
      obs[map_name] = value_map
      self._cached_maps[map_name] = base_map
      self._cached_color_maps[map_name] = value_map
    return obs

  def generate_map(
    self,
    map_config: MapConfig,
    base_map: dmap.TopdownMap
  ):
    # crop map
    if map_config.crop:
      if map_config.crop.center == 'camera':
        center = base_map.get_camera()
      elif map_config.crop.center == 'origin':
        center = base_map.get_origin()
      base_map = base_map.select(
        center,
        map_config.crop.width,
        map_config.crop.height
      )
    return base_map

  def colorize_map(
    self,
    map_config: MapConfig,
    base_map: dmap.TopdownMap
  ):
    if map_config.colorize:
      value_map = self.map_builder.get_value_map(
        base_map,
        colorize = True,
        **asdict(map_config.smooth)
      )
      # draw trajectory
      if map_config.colorize.draw_trajectory:
        value_map = draw_trajectory(value_map, base_map,
            self._cached_trajectory)
      # draw goal
      if map_config.colorize.draw_goals and self.enable_goals:
        for goal in self._goals:
          value_map = draw_goal(value_map, base_map, goal)
      # draw planner goal
      if map_config.colorize.draw_planner_goal:
        if hasattr(self.env, 'get_plan_state'):
          plan_state = self.env.get_plan_state()
          if plan_state is not None:
            goal = plan_state.plan.goal
            plan_goal = (goal[0], 0, goal[1])
            draw_goal(value_map, base_map, plan_goal)
      # draw origin, camera
      if map_config.colorize.draw_origin:
        value_map = draw_origin(value_map, base_map)
      if map_config.colorize.draw_camera:
        value_map = draw_camera(value_map, base_map)
    else:
      value_map = self.map_builder.get_value_map(
        base_map,
        colorize = False,
        **asdict(map_config.smooth)
      )
    return value_map

  def make_observation_space(self):
    if self.env.observation_space is None:
      return None
    obs_space = self.env.observation_space
    new_obs_space = {key: obs_space[key] for key in obs_space}
    for map_name, map_config in self.maps_config.items():
      if map_config.crop:
        width = map_config.crop.width
        height = map_config.crop.height
      else:
        assert RuntimeError("Size of the map does not fixed")
      if map_config.colorize:
        low = 0
        high = 255
        dtype = np.uint8
        channel = 3
      else:
        low = 0.0
        high = 1.0
        dtype = np.float32
        channel = self.map_builder.channels
      space = gym.spaces.Box(
        low = low,
        high = high,
        dtype = dtype,
        shape = (width, height, channel)
      )
      new_obs_space[map_name] = space
    return new_obs_space

  def get_goals(self):
    init_pos = self.env.habitat_env.current_episode.start_position
    init_pos = np.asarray(init_pos)
    init_rot = self.env.habitat_env.current_episode.start_rotation
    init_rot = np.asarray(init_rot)
    goals = self.env.habitat_env.current_episode.goals
    goal_pos = [
      compute_relative_goals(init_pos, init_rot, goal.position)
      for goal in goals
    ]
    return np.asarray(goal_pos)

  def render(self, mode="human", planner=None):
    res = self.env.render(mode=mode)
    if mode == 'human' or mode == 'interact':
      for map_name in self.render_maps:
        if map_name in self._cached_color_maps:
          map_image = self._cached_color_maps[map_name]
          cv2.imshow(map_name, map_image[...,::-1])
    return res