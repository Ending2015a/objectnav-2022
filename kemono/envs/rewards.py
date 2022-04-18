# --- built in ---
from typing import (
  Any,
  Dict,
  Optional
)
# --- 3rd party ---
from rlchemy import registry
# --- my module ---
from kemono.envs.habitat_env import HabitatEnv

@registry.register.reward('v0', default=True)
class Reward_v0:
  reward_range = {-float("inf"), float("inf")}
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
class Reward_v1:
  reward_range = {-10.0, 10.0}
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
