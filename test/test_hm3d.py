# --- built in ---
import os
import sys
# --- 3rd party ---
import cv2
import gym
import habitat
import numpy as np
from habitat.sims.habitat_simulator.actions import HabitatSimActions
# --- my module ---
sys.path.append('/src/')
import lib

FORWARD_KEY = "w"
LEFT_KEY    = "a"
RIGHT_KEY   = "d"
FINISH      = "f"
QUIT        = "q"

CONFIG_PATH = '/src/configs/challenge/challenge_objectnav2022.local.val_mini.rgbd.yaml'
# bgr

def make_graph(observations):
  rgb = observations['rgb']
  depth = observations['depth']
  #semantic = observations['semantic']
  #print(semantic.dtype)
  # depth map (h, w, 1) -> (h, w, 3)
  depth = (np.concatenate((depth,)*3, axis=-1) * 255.0).astype(np.uint8)
  # color image rgb -> bgr
  bgr = rgb[:, :, [2,1,0]]
  # semantic
  #seg = semantic_colors[semantic]
  scene = np.concatenate((bgr, depth), axis=1)
  #scene = np.concatenate((bgr, depth, seg), axis=1)
  return scene

def example():
  config = lib.get_config(CONFIG_PATH)
  env = habitat.Env(
    config=config
  )

  print("Environment creation successful")
  while True:
    observations = env.reset()
    print('Episode id: {}, scene id: {}'.format(env.current_episode.episode_id, env.current_episode.scene_id))

      # --- show observations ---
    print('Observations:')
    print(f'  Object goal: {observations["objectgoal"]}')
    print(f"  GPS: ({observations['gps'][1]:.5f}, {observations['gps'][0]:.5f}), compass: {observations['compass'][0]:.5f}")

    cv2.imshow('Sensors', make_graph(observations))

    print("Agent stepping around inside environment.")

    count_steps = 0
    while not env.episode_over:
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
      elif keystroke == ord(FINISH):
        action = HabitatSimActions.STOP
        print("action: FINISH")
      elif keystroke == ord(QUIT):
        print('Exit simulator')
        exit(0)
      else:
        print("INVALID KEY")
        continue

      observations = env.step(action)
      count_steps += 1
      # --- show observations ---
      print('Observations:')
      print(f"  GPS: ({observations['gps'][1]:.5f}, {observations['gps'][0]:.5f}), compass: {observations['compass'][0]:.5f}")


      cv2.imshow('Sensors', make_graph(observations))

      metrics = env.get_metrics()
      print('Metrics:')
      print('  distance to goal: {:.3f}'.format(metrics['distance_to_goal']))
      print('  success: {}, spl: {:.3f}, softspl: {:.3f}'.format(
              metrics['success'] == 1.0,
              metrics['spl'], metrics['softspl']))

    print("Episode finished after {} steps.".format(count_steps))


if __name__ == "__main__":
  example()