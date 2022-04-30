# --- built in ---
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Union
# --- 3rd party ---
import cv2
import numpy as np
import gym
import habitat
from rlchemy import registry
from habitat.sims.habitat_simulator.actions import HabitatSimActions
# --- my module ---
from kemono.planner.base import PlanState


class PlannerWrapper(gym.Wrapper):
  plan_distance_key = 'plan_distance'
  plan_angle_key = 'plan_angle'
  plan_time_key = 'plan_time'
  gps_key = 'gps'
  compass_key = 'compass'
  def __init__(
    self,
    env: habitat.RLEnv,
    planner_name: str = 'none',
    planner_kwargs: Dict[str, Any] = {},
    enable_reward: bool = False
  ):
    """_summary_

    Args:
      env (habitat.RLEnv): _description_
      planner_name (str, optional): _description_. Defaults to 'none'.
      planner_kwargs (Dict[str, Any], optional): _description_. Defaults to {}.
    """
    super().__init__(env=env)
    self.planner_name = planner_name
    self.planner_kwargs = planner_kwargs
    planner_class = registry.get.planner(self.planner_name)
    if planner_class is None:
      print(f'Planner not found ... {self.planner_class}')
      self.planner = None
    else:
      self.planner = planner_class(env, **planner_kwargs)
    self.enable_reward = enable_reward
    self.observation_space = self.make_observation_space()
    self._cached_plan_state: Optional[PlanState] = None
    if self.enable_reward:
      self.reward_fn = Reward()

  def get_plan_state(self) -> Optional[PlanState]:
    return self._cached_plan_state

  def step(self, action, obs=None):
    obs, rew, done, info = self.env.step(action, obs=obs)
    obs, plan_state = self.get_observations_and_plan_state(obs, action)
    info = self.get_info(obs, info, plan_state)
    rew = self.get_reward(obs, action, rew, done, info, plan_state)
    return obs, rew, done, info

  def reset(self, obs=None):
    obs = self.env.reset(obs=obs)
    if self.planner is not None:
      self.planner.reset()
      self._cached_plan_state = None
    obs, _ = self.get_observations_and_plan_state(obs)
    return obs

  def make_observation_space(self):
    if self.planner is None:
      return
    if self.env.observation_space is None:
      return
    obs_space = self.observation_space
    new_obs_spaces = {key: obs_space[key] for key in obs_space}
    # (x, y, duration)
    plan_distance_obs_space = gym.spaces.Box(
      low = -float('inf'),
      high = float('inf'),
      shape = [],
      dtype = np.float32
    )
    plan_angle_obs_space = gym.spaces.Box(
      low = -float('inf'),
      high = float('inf'),
      shape = [],
      dtype = np.float32
    )
    plan_time_obs_space = gym.spaces.Box(
      low = 0,
      high = float('inf'),
      shape = [],
      dtype = np.float32
    )
    new_obs_spaces[self.plan_distance_key] = plan_distance_obs_space
    new_obs_spaces[self.plan_angle_key] = plan_angle_obs_space
    new_obs_spaces[self.plan_time_key] = plan_time_obs_space
    new_obs_space = gym.spaces.Dict(new_obs_spaces)
    return new_obs_space

  def get_observations_and_plan_state(
    self, obs: Dict[str, Any], action = None
  ) -> Union[Dict[str, Any], PlanState]:
    if self.planner is None:
      return obs, None
    plan_state = self._cached_plan_state
    # update plan states
    plan_state = self.planner.step(obs, action, plan_state)
    self._cached_plan_state = plan_state
    obs[self.plan_distance_key] = \
      np.asarray(plan_state.plan.distance).item()
    obs[self.plan_angle_key] = \
      np.asarray(plan_state.plan.angle).item()
    obs[self.plan_time_key] = \
      np.asarray(plan_state.plan.time).item()
    return obs, plan_state

  def get_info(self, obs, info, plan_state: PlanState):
    if plan_state is None:
      return info
    info['plan'] = plan_state
    return info

  def get_reward(self, obs, act, rew, done, info, plan_state):
    if self.enable_reward:
      return self.reward_fn(self, obs, act, rew, done, info, plan_state)
    else:
      return rew

  def render(self, mode='human'):
    return self.env.render(mode=mode)

class Reward():
  reward_range = (-1.0, 5.0)
  slack_reward = -1e-3
  success_reward = 5.0
  max_delta = 0.3
  def __call__(
    self,
    env,
    obs,
    act,
    rew,
    done,
    info,
    plan_state: PlanState
  ):
    mult = 1.0
    if env.to_habitat_action(act) == \
        HabitatSimActions.MOVE_FORWARD:
      mult = 2.0
    reward = self.slack_reward * mult
    plan = plan_state.plan
    old_plan = plan_state.old_plan
    if plan_state.archived is not None:
      if plan_state.succeed:
        reward += self.success_reward
      plan = plan_state.archived
    prev_dist = old_plan.distance
    dist = plan.distance
    move_reward = np.clip((prev_dist - dist)/self.max_delta, -1.0, 1.0)
    reward += move_reward * self.max_delta
    return np.clip(reward, self.reward_range[0], self.reward_range[1])
