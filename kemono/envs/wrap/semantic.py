# --- built in ---
# --- 3rd party ---
import cv2
import gym
import habitat
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
    seg = self.semap.get_categorical_map(obs['semantic'], bgr=True)
    return seg

class SemanticWrapper(gym.Wrapper):
  def __init__(
    self,
    env: habitat.RLEnv,
    use_ground_truth: bool = True
  ):
    """SemanticWrapper used to generate semantic segmentation
    observations

    Args:
        env (habitat.RLEnv): habitat environment
        use_ground_truth (bool, optional): whether to use ground truth
          semantic annotations. Defaults to False.
    """
    super().__init__(env=env)
    self._setup_interact = False
    self._cached_obs = None
    self.use_ground_truth = use_ground_truth
    self.semap = SemanticMapping(
      dataset = self.env.dataset
    )
    if use_ground_truth:
      self.predictor = _GTSemanticPredictor(
        self.semap, self.env.sim
      )
    else:
      raise NotImplementedError
  
  def step(self, action):
    obs, rew, done, info = self.env.step(action)
    obs = self.get_observations(obs)
    info = self.get_info(obs, info)
    return obs, rew, done, info

  def reset(self):
    obs = self.env.reset()
    self.predictor.reset()
    obs = self.get_observations(obs)
    return obs

  def get_observations(self, obs):
    obs['seg'] = self.predictor.predict(obs)
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
    if self._cached_obs is None:
      return res
    if mode == 'human':
      window_name = 'semantic'
      cv2.imshow(window_name, self._cached_obs['seg'])
    elif mode == 'interact':
      window_name = "semantic (interact)"
      self.setup_interact(window_name)
      cv2.imshow(window_name, self._cached_obs['seg'])
    return res

  def setup_interact(self, window_name):
    assert self.use_ground_truth
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
