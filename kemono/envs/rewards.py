# --- built in ---
from typing import (
  Any,
  Dict,
  Optional
)
# --- 3rd party ---
import numpy as np
from rlchemy import registry
# --- my module ---
from kemono.envs.habitat_env import HabitatEnv

class BaseReward:
  def reset(self, *args, **kwargs):
    pass

  def __call__(
    self,
    env: HabitatEnv,
    obs: Dict[str, Any],
    act: int,
    next_obs: Dict[str, Any],
    done: bool,
    info: Dict[str, Any]
  ):
    return 0

@registry.register.reward('v0', default=True)
class Reward_v0(BaseReward):
  reward_range = (-float("inf"), float("inf"))
  def __call__(
    self,
    env: HabitatEnv,
    obs: Dict[str, Any],
    act: int,
    next_obs: Dict[str, Any],
    done: bool,
    info: Dict[str, Any]
  ):
    """This reward function is used for habitat challenge"""
    return 0

@registry.register.reward('v1')
class Reward_v1(BaseReward):
  reward_range = (-10.0, 10.0)
  def __call__(
    self,
    env: HabitatEnv,
    obs: Dict[str, Any],
    act: int,
    next_obs: Dict[str, Any],
    done: bool,
    info: Dict[str, Any]
  ):
    metrics = info["metrics"]
    distance_to_goal = metrics['distance_to_goal']
    if metrics['success']:
      return 10.0
    return max(-distance_to_goal/10.0, -10.0)


@registry.register.reward('v2')
class Reward_v2(BaseReward):
  reward_range = (-10.0, 10.0)
  def __call__(
    self,
    env: HabitatEnv,
    obs: Dict[str, Any],
    act: int,
    next_obs: Dict[str, Any],
    done: bool,
    info: Dict[str, Any]
  ):
    """logarithm distance"""
    metrics = info["metrics"]
    distance_to_goal = metrics['distance_to_goal']
    distance_to_goal = np.clip(distance_to_goal, 0.01, 10)
    if metrics['success']:
      return 10.0
    return -np.log(distance_to_goal * 10.0) / (np.log(3) * 10.0)


@registry.register.reward('v3')
class Reward_v3(BaseReward):
  reward_range = (-1.001, 4.0)
  slack_reward = -1e-3
  success_reward = 3.0
  max_delta = 0.3
  def dlog(self, x, d):
    c = self.max_delta
    return np.log((x+c)/(x-d+c))

  def __call__(
    self,
    env: HabitatEnv,
    obs: Dict[str, Any],
    act: int,
    next_obs: Dict[str, Any],
    done: bool,
    info: Dict[str, Any]
  ):
    metrics = info["metrics"]
    prev_metrics = info['prev_metrics']
    d2g = metrics['distance_to_goal']
    prev_d2g = prev_metrics['distance_to_goal']
    delta = np.clip(d2g - prev_d2g, -self.max_delta, self.max_delta)
    reward = self.dlog(d2g, delta) + self.slack_reward
    if metrics['success']:
      reward += self.success_reward
    return np.clip(reward, self.reward_range[0], self.reward_range[1])
