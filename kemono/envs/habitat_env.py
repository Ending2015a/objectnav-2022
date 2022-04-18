# --- built in ---
import enum
from typing import (
  Any,
  Dict,
  List,
  Optional
)
from dataclasses import dataclass
# --- 3rd party ---
import cv2
import gym
import habitat
import numpy as np
import habitat
from habitat.core.dataset import Episode, EpisodeIterator
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from rlchemy import registry
# --- my module ---

__all__ = [
  'make',
  'HabitatEnv',
  'DummyHabitatEnv'
]

class HabitatEnvSpec:
  version: int = 0
  id: str = "HabitatEnv-v{}"
  def __init__(self, version: int=0):
    self.version = 0
    self.id = self.id.format(version)

class DummyHabitatEnvSpec:
  version: int = 0
  id: str = "DummyHabitatEnv-v{}"
  def __init__(self, version: int=0):
    self.version = 0
    self.id = self.id.format(version)

class _HabitatEnvCore(habitat.RLEnv):
  metadata = {"render.modes": ['rgb_array', 'human', 'interact']}
  reward_range = {-float("inf"), float("inf")}
  spec = HabitatEnvSpec()
  def __init__(
    self,
    config: habitat.Config,
    auto_stop: bool = False,
    enable_pitch: bool = False,
    enable_stop: bool = False,
    version: int = 0,
    spec_class: type = HabitatEnvSpec
  ):
    self.config = config
    self.tilt_angle = config.SIMULATOR.TILT_ANGLE
    self.auto_stop = auto_stop
    self.enable_pitch = enable_pitch
    self.enable_stop = enable_stop
    self.version = int(version)
    self.spec = spec_class(version=self.version)
    self.stop_distance = config.TASK.SUCCESS.SUCCESS_DISTANCE
    # Create reward function
    reward_class = registry.get.reward(f'v{self.version}')
    assert reward_class is not None, f"Reward version not defined: {self.version}"
    self.reward_fn = reward_class()
    # ---
    # Call self.make_env to create env
    self._agent_tilt_angle = 0
    self._cached_obs = None
    self._env = None
    self.observation_space = None
    self.action_space = None
    self.number_of_episodes = None
    # custom actions to habitat sim actions
    self.map_to_habitat_action = None
    # habitat sim actions to custom actions
    self.map_to_action = None

  @property
  def dataset(self) -> habitat.Dataset:
    if self._env is not None:
      return self._env._dataset
    else:
      return None

  @property
  def sim(self) -> habitat.Simulator:
    if self._env is not None:
      return self._env.sim
    else:
      return None

  @property
  def has_env(self) -> bool:
    return self._env is not None

  def to_habitat_action(self, action):
    if action not in self.map_to_habitat_action:
      return None
    return self.map_to_habitat_action[action]

  def from_habitat_action(self, habitat_action):
    if habitat_action not in self.map_to_action:
      return None
    return self.map_to_action[habitat_action]

  def make_env(self):
    self._env = habitat.Env(self.config)
    self.observation_space = self.make_observation_space()
    self.action_space = self.make_action_space()
    self.number_of_episodes = self._env.number_of_episodes
    self.reward_range = self.reward_fn.reward_range
  
  def make_observation_space(self):
    # augmenting compass
    # compass [heading, pitch]
    obs_space = self._env.observation_space
    compass_space = obs_space['compass']
    new_compass_space = gym.spaces.Box(
      high = compass_space.high.max(),
      low = compass_space.low.min(),
      shape = (2,),
      dtype = compass_space.dtype
    )
    new_obs_spaces = {key: obs_space[key] for key in obs_space}
    new_obs_spaces['compass'] = new_compass_space
    # redefine objectgoal space
    # the original objectgoal space is
    #  Box(0, 5, (1,), np.int64)
    # we modified this to
    #  Discrete(6)
    objectgoal_space = obs_space['objectgoal']
    n_goals = objectgoal_space.high.item() + 1
    new_objectgoal_space = gym.spaces.Discrete(n_goals)
    new_obs_spaces['objectgoal'] = new_objectgoal_space
    new_obs_space = gym.spaces.Dict(new_obs_spaces)
    return new_obs_space
  
  def make_action_space(self):
    act_map = [
      HabitatSimActions.MOVE_FORWARD,
      HabitatSimActions.TURN_LEFT,
      HabitatSimActions.TURN_RIGHT
    ]
    if self.enable_pitch:
      act_map.extend([
        HabitatSimActions.LOOK_UP,
        HabitatSimActions.LOOK_DOWN
      ])
    if self.enable_stop:
      act_map.extend([
        HabitatSimActions.STOP
      ])
    self.map_to_habitat_action = {idx: act for idx, act in enumerate(act_map)}
    self.map_to_action = {act: idx for idx, act in enumerate(act_map)}
    return gym.spaces.Discrete(len(act_map))

  def reset(self, *args, obs=None, **kwargs):
    self._agent_tilt_angle = 0
    if self.has_env:
      obs = self._env.reset()
    obs = self.get_observation(obs)
    return obs

  def step(self, action, *args, obs=None, **kwargs):
    action = np.asarray(action).item()
    action = self.to_habitat_action(action)
    if self.has_env:
      obs = self._env.step(action)
    # update agent tilt angle
    if action == HabitatSimActions.LOOK_UP:
      self._agent_tilt_angle += self.tilt_angle
    elif action == HabitatSimActions.LOOK_DOWN:
      self._agent_tilt_angle -= self.tilt_angle
    if obs is not None:
      obs = self.get_observation(obs)
    info = self.get_info(obs)
    done = self.get_done(obs)
    rew = self.get_reward(action, obs, done, info)
    if (not done and self.auto_stop
        and self.should_stop(obs, info)):
      obs, rew, done, info = self.force_stop()
    return obs, rew, done, info

  def should_stop(self, obs, info):
    distance_to_goal = info['metrics'].get('distance_to_goal', None)
    if distance_to_goal is None:
      return False
    return distance_to_goal < self.stop_distance

  def force_stop(self, obs=None):
    action = HabitatSimActions.STOP
    if self.has_env:
      obs = self._env.step(action)
    if obs is not None:
      obs = self.get_observation(obs)
    info = self.get_info(obs)
    done = self.get_done(obs)
    rew = self.get_reward(action, obs, done, info)
    return obs, rew, done, info

  def get_observation(self, obs):
    if obs is None:
      return obs
    # compass: [hading, pitch]
    compass = obs['compass']
    pitch = np.radians(self._agent_tilt_angle)
    obs['compass'] = np.concatenate((compass, [pitch]), axis=0)
    self._cached_obs = obs
    return obs
  
  def get_done(self, obs):
    if self.has_env:
      return self._env.episode_over
    return False
  
  def get_reward_range(self):
    return self.reward_range

  def get_reward(self, act, obs, done, info):
    return self.reward_fn(self, self._cached_obs, act, obs, done, info)

  def get_info(self, obs):
    if self.has_env:
      metrics = self._env.get_metrics()
      return {'metrics': metrics}
    else:
      # empty metrics
      return {'metrics': {}}

  def render(self, mode="human"):
    if self._cached_obs is None:
      return
    scene = self.render_scene()
    if mode == 'rgb_array':
      return scene[...,::-1] # rgb
    else:
      cv2.imshow("rgb + depth", scene)

  def render_scene(self):
    assert self._cached_obs is not None
    obs = self._cached_obs
    rgb = obs['rgb']
    depth = obs['depth']
    depth = (np.concatenate((depth,)*3, axis=-1) * 255.0).astype(np.uint8)
    bgr = rgb[...,::-1]
    scene = np.concatenate((bgr, depth), axis=1)
    return scene

  def seed(self, seed: Optional[int]=None) -> None:
    if self._env is not None:
      self._env.seed(seed)


class HabitatEnv(_HabitatEnvCore):
  def __init__(self, *args, spec_class=None, **kwargs):
    super().__init__(*args, spec_class=HabitatEnvSpec, **kwargs)
    self.make_env()


class DummyHabitatEnv(_HabitatEnvCore):
  def __init__(self, *args, spec_class=None, **kwargs):
    super().__init__(*args, spec_class=DummyHabitatEnvSpec, **kwargs)

  def make_env(self):
    pass

def make(id, *args, **kwargs) -> _HabitatEnvCore:
  env_name, version = id.split('-v')
  version = int(version)
  if env_name == 'HabitatEnv':
    return HabitatEnv(
      *args, **kwargs, version=version
    )
  elif env_name == 'DummyHabitatEnv':
    return DummyHabitatEnv(
      *args, **kwargs, version=version
    )
  else:
    raise ValueError(f"Unknown environment ID: {id}")