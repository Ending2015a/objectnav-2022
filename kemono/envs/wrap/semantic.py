# --- built in ---
import enum
# --- 3rd party ---
import cv2
import gym
import habitat
import numpy as np
# --- my module ---
from kemono.semantics import SemanticMapping

__all__ = [
  'SemanticWrapper'
]

class _GTSemanticPredictor():
  def __init__(
    self,
    semantic_mapping: SemanticMapping,
    simulator: habitat.Simulator
  ):
    self.semap = semantic_mapping
    self.sim = simulator
  
  def reset(self):
    self.semap.parse_semantics(
      self.sim.semantic_annotations(),
      reset = True
    )

  def predict(self, obs):
    assert 'semantic' in obs
    seg = self.semap.get_categorical_map(obs['semantic'])
    return seg

class _RedNetSemanticPredictor():
  # TODO: load pretrained rednet model
  # TODO: predict mpcat40 labels
  pass

class SemanticWrapper(gym.Wrapper):
  def __init__(
    self,
    env: habitat.RLEnv,
    predictor_type: str = 'none',
    colorized: bool = False
  ):
    """SemanticWrapper used to generate semantic segmentation
    observations

    Args:
      env (habitat.RLEnv): habitat environment
      predictor_type (bool, optional): type of predictor used to predict
        categorical maps, either ['gt', 'rednet', 'none']
      whether to use ground truth
        semantic annotations. Defaults to False.
      colorized (bool, optional): colorize semantic segmentation map.
        Defaults to False.
    """
    super().__init__(env=env)
    self._setup_interact = False
    self._cached_obs = None
    self.predictor_type = predictor_type.lower()
    self.colorized = colorized
    self.semap = SemanticMapping(
      dataset = self.env.dataset
    )
    if self.predictor_type == 'gt':
      self.predictor = _GTSemanticPredictor(
        self.semap, self.env.sim
      )
    elif self.predictor_type == 'none':
      self.predictor = None
    else:
      raise NotImplementedError(f'Unknown predictor: {self.predictor_type}')

    self.observation_space = self.make_observation_space()
  
  @property
  def semantic_mapping(self):
    return self.semap

  def step(self, action):
    obs, rew, done, info = self.env.step(action)
    obs = self.get_observations(obs)
    info = self.get_info(obs, info)
    return obs, rew, done, info

  def reset(self):
    obs = self.env.reset()
    if self.predictor is not None:
      self.predictor.reset()
    obs = self.get_observations(obs)
    return obs

  def make_observation_space(self):
    if self.predictor is None:
      return self.observation_space
    width = self.config.SIMULATOR.SEMANTIC_SENSOR.WIDTH
    height = self.config.SIMULATOR.SEMANTIC_SENSOR.HEIGHT
    obs_space = self.observation_space
    if self.colorized:
      seg_space = gym.spaces.Box(
        low = 0,
        high = 255,
        shape = (height, width, 3),
        dtype = np.uint32
      )
    else:
      seg_space = gym.spaces.Box(
        low = 0,
        high = 41,
        shape = (height, width),
        dtype = np.uint32
      )
    new_obs_spaces = {key: obs_space[key] for key in obs_space}
    new_obs_spaces['seg'] = seg_space
    new_obs_space = gym.spaces.Dict(new_obs_spaces)
    return new_obs_space

  def get_observations(self, obs):
    if self.predictor is None:
      return obs
    # predict category id map
    seg = self.predictor.predict(obs)
    # colorize segmentation map
    if self.colorized:
      seg = self.semap.colorize_categorical_map(seg, rgb=True)
    obs['seg'] = seg
    self._cached_obs = obs
    return obs
  
  def get_info(self, obs, info):
    info['goal'] = {
      'id': self.semap.get_goal_category_id(obs['objectgoal']),
      'name': self.semap.get_goal_category_name(obs['objectgoal'])
    }
    return info

  def render(self, mode='human'):
    res = self.env.render(mode=mode)
    if (self._cached_obs is None
        or 'seg' not in self._cached_obs):
      return res
    # colorize segmentation map
    seg = self._cached_obs['seg']
    if not self.colorized:
      seg = self.semap.colorize_categorical_map(seg, rgb=True)
    # make scene
    if mode == 'rgb_array':
      return np.concatenate((res, seg), axis=1)
    elif mode == 'human':
      window_name = 'semantic'
      cv2.imshow(window_name, seg[...,::-1])
    elif mode == 'interact':
      window_name = "semantic (interact)"
      self.setup_interact(window_name)
      cv2.imshow(window_name, seg[...,::-1])
    return res

  def setup_interact(self, window_name):
    assert self.predictor_type == 'gt'
    if not self._setup_interact:
      cv2.namedWindow(window_name)
      cv2.setMouseCallback(window_name, self.on_dubclick_probe_info)
      self._setup_interact = True

  def on_dubclick_probe_info(self, event, x, y, flags, param):
    semantic = self._cached_obs.get('semantic', None)
    if event == cv2.EVENT_LBUTTONDBLCLK:
      if semantic is None:
        return
      obj_id = semantic[y][x].item()
      print(f"On click (x, y) = ({x}, {y})")
      self.semap.print_object_info(obj_id, verbose=True)
