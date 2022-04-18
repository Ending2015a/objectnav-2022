# --- built in ---
from typing import (
  Callable
)
# --- 3rd party ---
import cv2
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
# --- my module ---
from kemono.envs.control.base_controller import BaseController

QUIT = 'q'

class ShortestPathController(BaseController):
  def __init__(self, *args, goal_radius: float=0.1, **kwargs):
    super().__init__(*args, **kwargs)
    if self.action_map is None:
      if hasattr(self.env, 'from_habitat_action'):
        self.action_map = self.env.from_habitat_action
      else:
        self.action_map = lambda x: x
    self.follower = ShortestPathFollower(self.env.sim, goal_radius, False)
    self.goal = None

  def reset(self, *args, goal, **kwargs):
    self.goal = goal
  
  def act(self, observations):
    keystroke = cv2.waitKey(1)
    if keystroke == ord(QUIT):
      print('Exit simulator')
      exit(0)
    action = self.follower.get_next_action(self.goal.position)
    action = self.action_map(action)
    return action
