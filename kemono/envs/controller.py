# --- built in ---
# --- 3rd party ---]
import cv2
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
# --- my module ---

__all__ = [
  'ManualController'
]

FORWARD_KEY = "w"
LEFT_KEY    = "a"
RIGHT_KEY   = "d"
FINISH      = "f"
LOOK_UP     = "x"
LOOK_DOWN   = "z"
QUIT        = "q"

class ManualController:
  def __init__(self, *args, **kwargs):
    pass

  def reset(self, *args, **kwargs):
    pass

  def step(self):
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
        action = None
        print("INVALID KEY")
      if action is not None:
        return action