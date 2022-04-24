# --- built in ---
from dataclasses import dataclass, asdict
# --- 3rd party ---
import numpy as np
from rlchemy import registry
# --- my module ---
from kemono.planner.base import Plan, PlanState, BasePlanner
from kemono.planner import utils


@registry.register.planner('random', default=True)
class RandomPlanner(BasePlanner):
  gps_key = 'gps'
  def __init__(
    self,
    max_distance: float = 5.0,
    min_distance: float = 2.0,
    replan_duration: int = 0,
    replan_distance: float = 0.1
  ):
    self.max_distance = max_distance
    self.min_distance = min_distance
    self.replan_duration = replan_duration
    self.replan_distance = replan_distance
    self.will_expire = self.replan_duration > 0

  def reset(self):
    pass

  def step(self, obs, plan_state: PlanState=None):
    if plan_state is None:
      plan = self.replan(obs)
      plan_state = PlanState(plan=plan)
    else:
      old_plan = plan_state.plan
      plan = self.update_plan(obs, old_plan)
      # check if the plan is expired or succeed
      if plan.need_replan:
        plan_state = PlanState(
          plan = self.replan(obs),
          old_plan = old_plan,
          archived = plan
        )
      else:
        plan_state = PlanState(
          plan = plan,
          old_plan = old_plan
        )
    return plan_state

  def replan(self, obs):
    goal = self.predict(obs)
    plan = self.create_plan(obs, goal)
    plan.time = self.replan_duration
    return plan

  def update_plan(self, obs, old_plan):
    goal = old_plan.goal
    plan = self.create_plan(obs, goal)
    if self.will_expire:
      plan.time = old_plan.time - 1
    else:
      plan.time = 0
    plan.expired = (self.will_expire and plan.time == 0)
    plan.succeed = plan.distance <= self.replan_distance
    plan.need_replan = (plan.expired or plan.succeed)
    return plan

  def predict(self, obs):
    d = np.random.uniform(-1.0, 1.0, size=(2,))
    g = np.abs(d) * (self.max_distance - self.min_distance)
    goal_delta = (g + self.min_distance) * np.sign(d)
    agent_pos = self.get_agent_pose(obs)[:2]
    goal = agent_pos + goal_delta
    return goal
