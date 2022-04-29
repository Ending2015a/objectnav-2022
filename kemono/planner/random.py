# --- built in ---
from dataclasses import dataclass, asdict
# --- 3rd party ---
import numpy as np
import habitat
from rlchemy import registry
from habitat.sims.habitat_simulator.actions import HabitatSimActions
# --- my module ---
from kemono.planner.base import Plan, PlanState, BasePlanner
from kemono.planner import utils


@registry.register.planner('random', default=True)
class RandomPlanner(BasePlanner):
  gps_key = 'gps'
  def __init__(
    self,
    env: habitat.RLEnv,
    max_distance: float = 5.0,
    min_distance: float = 2.0,
    replan_duration: int = 0,
    replan_distance: float = 0.1,
    max_replan_duration: int = 0
  ):
    assert env.habitat_env.sim is not None
    self.env = env
    self.sim = env.habitat_env.sim
    self.max_distance = max_distance
    self.min_distance = min_distance
    self.replan_duration = replan_duration
    self.replan_distance = replan_distance
    self.max_replan_duration = max_replan_duration
    self.will_expire = self.replan_duration > 0
    self._current_height = 0
    self._gt_map = None
    self._meters_per_pixel = 0.1

  def reset(self):
    self._current_height = self.get_current_height()
    self._gt_map = self.get_gt_map()

  def step(
    self,
    obs,
    act = None,
    plan_state: PlanState = None
  ):
    agent_height = self.get_current_height()
    if abs(agent_height - self._current_height > 1.0):
      # regenerate gt topdown map
      self._current_height = agent_height
      self._gt_map = self.get_gt_map()

    if plan_state is None:
      plan = self.replan(obs)
      plan_state = PlanState(plan=plan)
    else:
      old_plan = plan_state.plan
      plan = self.update_plan(obs, act, old_plan)
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
    plan.max_time = self.max_replan_duration
    self._need_replan = False
    return plan

  def update_plan(self, obs, act, old_plan):
    goal = old_plan.goal
    plan = self.create_plan(obs, goal)
    if self.will_expire:
      plan.max_time = old_plan.max_time - 1
      plan.time = old_plan.time
      if self.env.to_habitat_action(act) == \
          HabitatSimActions.MOVE_FORWARD:
        plan.time = old_plan.time - 1
    else:
      plan.max_time = 0
      plan.time = 0
    print(plan.max_time, plan.time)
    plan.expired = self.will_expire and (
      plan.max_time == 0 or plan.time == 0
    )
    plan.succeed = plan.distance <= self.replan_distance
    plan.need_replan = (plan.expired or plan.succeed)
    return plan

  def predict(self, obs):
    from habitat.utils.geometry_utils import (
      quaternion_rotate_vector,
      quaternion_from_coeff
    )
    bounds = self.sim.pathfinder.get_bounds()
    init_pos = self.env.habitat_env.current_episode.start_position
    init_rot = self.env.habitat_env.current_episode.start_rotation
    agent_pos = self.sim.get_agent(0).state.position
    height, width = self._gt_map.shape
    available_locs = []
    candidate_locs = []
    for h in range(height):
      for w in range(width):
        if self._gt_map[h, w]:
          loc = np.array((
            w * self._meters_per_pixel + bounds[0][0],
            agent_pos[1],
            h * self._meters_per_pixel + bounds[0][2]
          ), dtype=np.float32)
          available_locs.append(loc)
          # compute distance
          dist = np.linalg.norm(loc - agent_pos)
          if dist >= self.min_distance and dist <= self.max_distance:
            candidate_locs.append(loc)
    if len(candidate_locs) == 0:
      candidate_locs = available_locs

    # shuffle and random draw the first navigable point
    loc_indices = np.arange(len(candidate_locs))
    np.random.shuffle(loc_indices)
    for loc_idx in loc_indices:
      loc = candidate_locs[loc_idx]
      if self.sim.pathfinder.is_navigable(loc):
        break
    sampled_loc = loc

    # convert to episode coordinate (0, 0) at the starting position
    init_rot = quaternion_from_coeff(init_rot)
    global_loc = quaternion_rotate_vector(
      init_rot.inverse(), sampled_loc - init_pos
    )
    global_loc = np.array((global_loc[0], -global_loc[2]), dtype=np.float32)
    return global_loc

  def get_gt_map(self):
    return self.sim.pathfinder.get_topdown_view(
      self._meters_per_pixel, self._current_height
    )

  def get_current_height(self):
    return self.sim.get_agent(0).state.position[1]
