# --- built in ---
import abc
from dataclasses import dataclass
from typing import Dict, Any, Optional
# --- 3rd party ---
import numpy as np
# --- my module ---
from kemono.planner import utils

@dataclass
class Plan:
  goal: np.ndarray
  distance: float
  angle: float
  time: float
  expired: bool
  succeed: bool
  need_replan: bool
  # stores some private states from the planners
  _state: Optional[Dict[str, Any]] = None

@dataclass
class PlanState:
  plan: Plan
  old_plan: Optional[Plan] = None
  # the old plan is archived for one step if it is expired or succeed
  archived: Optional[Plan] = None
  expired: bool = False
  succeed: bool = False
  need_replan: bool = False
  def __post_init__(self):
    if self.archived is not None:
      self.expired = self.archived.expired
      self.succeed = self.archived.succeed
      self.need_replan = self.archived.need_replan

class BasePlanner():
  gps_key = 'gps'
  compass_key = 'compass'
  @abc.abstractmethod
  def reset(self):
    pass

  @abc.abstractmethod
  def step(self, obs, plan_state: PlanState=None):
    """Predict new plans or update old plans"""
    # generate plan
    pass

  @abc.abstractmethod
  def predict(self, obs) -> np.ndarray:
    # predict global goal [x, z]
    return np.asarray((0, 0), dtype=np.float32)

  def create_plan(self, obs, goal):
    agent_pose = self.get_agent_pose(obs)
    local_goal = utils.global_to_local(goal, agent_pose)
    polar_goal = utils.cart_to_polar(local_goal)
    return Plan(
      goal = goal,
      distance = polar_goal[0],
      angle = polar_goal[1],
      time = 0,
      expired = False,
      succeed = False,
      need_replan = False,
      _state = {
        'global': goal,
        'local': local_goal,
        'polar': polar_goal,
        'pose': agent_pose
      }
    )

  def get_agent_pose(self, obs):
    return np.asarray(
      (obs[self.gps_key][1],
        obs[self.gps_key][0],
        obs[self.compass_key][0]),
      dtype = np.float32
    )