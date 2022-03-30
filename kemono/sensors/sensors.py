# --- built in ---
import os
import sys
import time
import logging

from typing import Any, Dict, List, Optional, Type, Union

# --- 3rd party ---
import gym
import habitat

import numpy as np

from habitat.config import Config
from habitat.core.registry import registry
from habitat.core.simulator import (AgentState,
                  Sensor,
                  SensorTypes,
                  Simulator)
from habitat.core.utils import try_cv2_import
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import (quaternion_rotate_vector,
                      quaternion_from_coeff)
from habitat.tasks.nav.nav import PointGoalSensor, DistanceToGoal, Success
from habitat.core.embodied_task import (EmbodiedTask, Measure)
from habitat.utils.visualizations import maps
cv2 = try_cv2_import()

# --- my module ---

# copy from OccupancyAnticipation
@habitat.registry.register_sensor(name='GTPoseSensor')
class GTPoseSensor(Sensor):
  r"""The agents current ground-truth pose in the coordinate frame defined by
  the episode, i.e. the axis it faces along and the origin is defined by
  its state at t=0.
  Observations:
    [0]: X coordinate in start coordinate system
    [1]: Z coordinate in start coordinate system
    [2]: Agent's heading
  Args:
    sim: reference to the simulator for calculating task observations.
    config: not needed
  """
  def __init__(self, sim: Simulator, config: Config, *args, **kwargs):
    self._sim = sim

    super().__init__(config=config)

  def _get_uuid(self, *args, **kwargs):
    return 'pose_gt'

  def _get_sensor_type(self, *args, **kwargs):
    return SensorTypes.POSITION

  def _get_observation_space(self, *args, **kwargs):
    return gym.spaces.Box(low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                shape=(3,),
                dtype=np.float32)
  def _quat_to_xy_heading(self, quat):
    direction_vector = np.array([0, 0, -1])

    heading_vector = quaternion_rotate_vector(quat, direction_vector)

    phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
    return np.array(phi)

  def get_observation(self, *args: Any, observations, episode, **kwargs: Any):
    agent_state = self._sim.get_agent_state()

    origin = np.array(episode.start_position, dtype=np.float32)
    rotation_world_start = quaternion_from_coeff(episode.start_rotation)

    agent_position = agent_state.position

    agent_position = quaternion_rotate_vector(
      rotation_world_start.inverse(), agent_position - origin
    )

    rotation_world_agent = agent_state.rotation
    rotation_world_start = quaternion_from_coeff(episode.start_rotation)

    agent_heading = self._quat_to_xy_heading(
      rotation_world_agent.inverse() * rotation_world_start
    )
    # This is rotation from -Z to -X. We want -Z to X for this particular sensor.
    #agent_heading = -agent_heading

    return np.array(
      [agent_position[0], -agent_position[2], agent_heading], dtype=np.float32,
    )


@registry.register_measure
class FakeSPL(Measure):
  r"""SPL (Success weighted by Path Length)
  ref: On Evaluation of Embodied Agents - Anderson et. al
  https://arxiv.org/pdf/1807.06757.pdf
  The measure depends on Distance to Goal measure and Success measure
  to improve computational
  performance for sophisticated goal areas.
  """

  def __init__(
    self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
  ):
    self._previous_position = None
    self._start_end_episode_distance = None
    self._agent_episode_distance: Optional[float] = None
    self._episode_view_points = None
    self._sim = sim
    self._config = config

    super().__init__()

  def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
    return "fake_spl"

  def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
    task.measurements.check_measure_dependencies(
      self.uuid, [DistanceToGoal.cls_uuid, Success.cls_uuid]
    )

    self._previous_position = self._sim.get_agent_state().position
    self._agent_episode_distance = 0.0
    self._start_end_episode_distance = task.measurements.measures[
      DistanceToGoal.cls_uuid
    ].get_metric()
    self.update_metric(  # type:ignore
      episode=episode, task=task, *args, **kwargs
    )

  def _euclidean_distance(self, position_a, position_b):
    return np.linalg.norm(position_b - position_a, ord=2)

  def update_metric(
    self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any
  ):

    current_position = self._sim.get_agent_state().position
    self._agent_episode_distance += self._euclidean_distance(
      current_position, self._previous_position
    )

    self._previous_position = current_position

    self._metric = (
      self._start_end_episode_distance
      / max(
        self._start_end_episode_distance, self._agent_episode_distance
      )
    )
