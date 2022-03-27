# --- built in ---
# --- 3rd party ---
import cv2
import gym
import torch
import habitat
import numpy as np
import dungeon_maps as dmap
# --- my module ---

__all__ = [
  "MapBuilderEnv"
]

# Colors [b, g, r]
hex2bgr = lambda hex: [int(hex[i:i+2], 16) for i in (0, 2, 4)][::-1]
FLOOR_COLOR   = hex2bgr('90D5C3')
WALL_COLOR    = hex2bgr('6798D0')
INVALID_COLOR = hex2bgr('F4F7FA')
CAMERA_COLOR  = hex2bgr('EC5565')
ORIGIN_COLOR  = hex2bgr('FFC300')

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

def draw_camera(
  image: np.ndarray,
  topdown_map: dmap.TopdownMap,
  color: np.ndarray = CAMERA_COLOR,
  size: int = 4
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

class MapBuilder():
  def __init__(
    self,
    config: habitat.Config,
    map_res: float = 0.03,
    map_width: int = 600,
    map_height: int = 600,
    trunc_height: float = 1.5
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
      trunc_depth_min = self.min_depth,
      trunc_depth_max = self.max_depth,
      trunc_height = trunc_height,
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
  def __init__(self, env: habitat.RLEnv):
    super().__init__(env=env)
    config = env.config
    self.map_builder = MapBuilder(
      config = config
    )
    self.local_map_key = "local_map"
    self.world_map_key = "world_map"
    self._cached_obs = None

  def step(self, action):
    obs, rew, done, info = self.env.step(action)
    obs = self.get_observations(obs)
    self._cached_obs = obs
    return obs, rew, done, info
  
  def reset(self):
    obs = self.env.reset()
    self.map_builder.reset()
    obs = self.get_observations(obs)
    self._cached_obs = obs
    return obs

  def get_observations(self, obs):
    # update map builder
    local_map = self.map_builder.step(obs)
    world_map = self.map_builder.world_map
    obs[self.local_map_key] = draw_map(local_map)
    obs[self.world_map_key] = draw_map(world_map)
    return obs

  def render(self, mode="human"):
    res = self.env.render(mode=mode)
    if mode == 'human' or mode == 'interact':
      cv2.imshow("local_map", self._cached_obs[self.local_map_key])
      cv2.imshow("world_map", self._cached_obs[self.world_map_key])
    return res