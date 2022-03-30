# --- built in ---
from typing import Any, Dict
from dataclasses import dataclass
# --- 3rd party ---
import cv2
import gym
import habitat
import numpy as np
from habitat.sims.habitat_simulator.actions import HabitatSimActions
# --- my module ---

__all__ = [
  'make',
  'TrainEnv'
]

def reward_v0(
  self: "TrainEnv",
  obs: Dict[str, Any],
  act: int,
  next_obs: Dict[str, Any],
  done: bool,
  info: Dict[str, Any]
):
  return 0

@dataclass
class TrainEnvSpec():
  id: str = "HabitatTrain-v0"

class TrainEnv(habitat.RLEnv):
  metadata = {"render.modes": ['rgb_array', 'human', 'interact']}
  reward_range = {-float("inf"), float("inf")}
  spec = TrainEnvSpec()
  def __init__(
    self,
    config: habitat.Config,
    dataset: habitat.Dataset = None,
    auto_stop: bool = False,
    version: int = 0
  ):
    super().__init__(config=config, dataset=dataset)
    self.spec = TrainEnvSpec(f"HabitatTrain-v{version}")
    self.config = config
    self.tilt_angle = config.SIMULATOR.TILT_ANGLE
    self.version = version
    self.auto_stop = auto_stop
    self.stop_distance = config.TASK.SUCCESS.SUCCESS_DISTANCE
    # ---
    self._agent_tilt_angle = 0
    self._cached_obs = None

    self.observation_space = self.make_observation_space()

  @property
  def dataset(self) -> habitat.Dataset:
    return self._env._dataset

  @property
  def sim(self) -> habitat.Simulator:
    return self._env.sim

  def step(self, action):
    obs = self._env.step(action)
    # update agent tilt angle
    if action == HabitatSimActions.LOOK_UP:
      self._agent_tilt_angle += self.tilt_angle
    elif action == HabitatSimActions.LOOK_DOWN:
      self._agent_tilt_angle -= self.tilt_angle
    obs = self.get_observations(obs)
    info = self.get_info(obs)
    done = self.get_done(obs)
    rew = self.get_reward(action, obs, done, info)
    if not done and self.auto_stop:
      distance_to_goal = info['metrics']['distance_to_goal']
      if distance_to_goal < self.stop_distance:
        obs, rew, done, info = self.step(HabitatSimActions.STOP)
    return obs, rew, done, info
  
  def reset(self):
    obs = self._env.reset()
    self._agent_tilt_angle = 0
    obs = self.get_observations(obs)
    return obs

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
    new_obs_space = gym.spaces.Dict(new_obs_spaces)
    return new_obs_space

  def get_observations(self, obs):
    # compass: [heading, pitch]
    compass = obs['compass']
    pitch = np.radians(self._agent_tilt_angle)
    obs['compass'] = np.concatenate((compass, [pitch]), axis=0)
    self._cached_obs = obs
    return obs

  def get_done(self, obs):
    return self._env.episode_over

  def get_reward_range(self):
    return [-float('inf'), float('inf')]

  def get_reward(self, act, obs, done, info):
    if self.version == 0:
      return reward_v0(self, self._cached_obs, act, obs, done, info)
    else:
      raise NotImplementedError(f"Unknown env version: {self.version}")

  def get_info(self, obs):
    """Get environment info
    Available key:
    * metrics
      * distance_to_goal: nearest goal
      * success
      * spl
      * softspl

    Returns:
        _type_: _description_
    """
    metrics = self._env.get_metrics()
    return {'metrics': metrics}

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

def make(id, *args, **kwargs):
  assert 'HabitatTrain-v' in id
  version = int(id.split('-v')[-1])
  return TrainEnv(
    *args, **kwargs, version=version
  )
