# --- built in ---
from typing import (
  Callable
)
# --- 3rd party ---
import cv2
from habitat.sims.habitat_simulator.actions import HabitatSimActions
# --- my module ---
from kemono.envs.control.base_controller import BaseController

FORWARD_KEY = "w"
LEFT_KEY    = "a"
RIGHT_KEY   = "d"
FINISH      = "f"
LOOK_UP     = "x"
LOOK_DOWN   = "z"
QUIT        = "q"

class ManualController(BaseController):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self._key_func = {}

    if self.action_map is None:
      if hasattr(self.env, 'from_habitat_action'):
        self.action_map = self.env.from_habitat_action
      else:
        self.action_map = lambda x: x

  def reset(self, *args, **kwargs):
    pass

  def act(self, observations):
    while True:
      keystroke = cv2.waitKey(0)
      if keystroke == ord(FORWARD_KEY):
        action = HabitatSimActions.MOVE_FORWARD
        print("action: FORWARD")
      elif keystroke == ord(LEFT_KEY):
        action = HabitatSimActions.TURN_LEFT
        print("action: LEFT")
      elif keystroke == ord(RIGHT_KEY):
        action = HabitatSimActions.TURN_RIGHT
        print("action: RIGHT")
      elif keystroke == ord(LOOK_UP):
        action = HabitatSimActions.LOOK_UP
        print("action: LOOK_UP")
      elif keystroke == ord(LOOK_DOWN):
        action = HabitatSimActions.LOOK_DOWN
        print("action: LOOK_DOWN")
      elif keystroke == ord(FINISH):
        action = HabitatSimActions.STOP
        print("action: FINISH")
      elif keystroke == ord(QUIT):
        print('Exit simulator')
        exit(0)
      else:
        if keystroke in self._key_func:
          func = self._key_func[keystroke]
          action = func(observations)
          print(f"Invoke {func}")
        else:
          action = None
          print("INVALID KEY")
      if action is not None:
        return self.action_map(action)

  def register_key(self, key: int, func: Callable):
    assert key not in self._key_func, 'Conflict'
    self._key_func[key] = func
