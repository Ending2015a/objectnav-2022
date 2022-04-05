# --- built in ---
import enum
from typing import Any, Dict
# --- 3rd party ---
import cv2
import gym
import habitat
import numpy as np
import rlchemy
from rlchemy import registry
# --- my module ---
from kemono.semantics import SemanticMapping

__all__ = [
  'SemanticWrapper'
]

@registry.register.semantic_predictor('gt', default=True)
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

class SemanticWrapper(gym.Wrapper):
  seg_key = 'seg'
  seg_color_key = 'seg_color'
  def __init__(
    self,
    env: habitat.RLEnv,
    predictor_name: str = 'none',
    colorized: bool = False,
    predictor_kwargs: Dict[str, Any] = {},
  ):
    """SemanticWrapper used to generate semantic segmentation
    observations

    Args:
      env (habitat.RLEnv): habitat environment
      predictor_name (bool, optional): type of predictor used to predict
        categorical maps, either ['gt', 'model', 'none']
      whether to use ground truth
        semantic annotations. Defaults to False.
      colorized (bool, optional): colorize semantic segmentation map.
        Defaults to False.
    """
    super().__init__(env=env)
    self._setup_interact = False
    self._cached_obs = None
    self.predictor_name = predictor_name.lower().strip()
    self.colorized = colorized
    self.semap = SemanticMapping(
      dataset = self.env.dataset
    )
    if self.predictor_name == 'gt':
      self.predictor = _GTSemanticPredictor(
        self.semap, self.env.sim
      )
    else:
      predictor_class = registry.get.semantic_predictor(self.predictor_name)
      if predictor_class is None:
        print(f'Predictor not found... {self.predictor_name}')
        self.predictor = None
      else:
        self.predictor = predictor_class(**predictor_kwargs)

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
    new_obs_spaces = {key: obs_space[key] for key in obs_space}
    # raw semantic id spaces (category ids)
    seg_space = gym.spaces.Box(
      low = 0,
      high = 41,
      shape = (height, width),
      dtype = np.int32
    )
    new_obs_spaces[self.seg_key] = seg_space
    # colorized semantic spaces (RGB image)
    if self.colorized:
      seg_color_space = gym.spaces.Box(
        low = 0,
        high = 255,
        shape = (height, width, 3),
        dtype = np.uint8
      )
      new_obs_spaces[self.seg_color_key] = seg_color_space
    # create new Dict spaces
    new_obs_space = gym.spaces.Dict(new_obs_spaces)
    return new_obs_space

  def get_observations(self, obs):
    if self.predictor is None:
      return obs
    # predict category id map
    seg = self.predictor.predict(obs)
    obs[self.seg_key] = seg
    # colorize segmentation map
    if self.colorized:
      seg_color = self.semap.colorize_categorical_map(seg, rgb=True)
      obs[self.seg_color_key] = seg_color
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
        or self.seg_key not in self._cached_obs):
      return res
    # if the colorized flag is not set
    # seg_color (RGB image) does not contain in the observation dict
    # hense we have to colorize from the raw seg map (ID map)
    # otherwise, we can directly use the seg_color in the observation dict
    if not self.colorized:
      seg = self._cached_obs[self.seg_key]
      seg_color = self.semap.colorize_categorical_map(seg, rgb=True)
    else:
      seg_color = self._cached_obs[self.seg_color_key]
    # make scene
    if mode == 'rgb_array':
      # return concatenated RGB image
      return np.concatenate((res, seg_color), axis=1)
    elif mode == 'human':
      window_name = 'semantic'
      cv2.imshow(window_name, seg_color[...,::-1])
    elif mode == 'interact':
      window_name = "semantic (interact)"
      self.setup_interact(window_name)
      cv2.imshow(window_name, seg_color[...,::-1])
    return res

  def setup_interact(self, window_name):
    assert self.predictor_name == 'gt'
    if not self._setup_interact:
      cv2.namedWindow(window_name)
      cv2.setMouseCallback(window_name, self.on_dubclick_probe_info)
      self._setup_interact = True

  def on_dubclick_probe_info(self, event, x, y, flags, param):
    semantic = self._cached_obs.get('semantic', None)
    if event == cv2.EVENT_LBUTTONDOWN:
      if semantic is None:
        return
      obj_id = semantic[y][x].item()
      print(f"On click (x, y) = ({x}, {y})")
      self.semap.print_object_info(obj_id, verbose=True)
