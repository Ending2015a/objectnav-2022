# --- built in ---
from dataclasses import dataclass
# --- 3rd party ---
import cv2
import gym
import habitat
from omegaconf import OmegaConf
import numpy as np
from habitat.sims.habitat_simulator.actions import HabitatSimActions
# --- my module ---
import kemono
from kemono.envs import train_env
from kemono.envs.wrap import (
  SemanticMapBuilderWrapper,
  SemanticWrapper
)

FORWARD_KEY = "w"
LEFT_KEY    = "a"
RIGHT_KEY   = "d"
FINISH      = "f"
LOOK_UP     = "x"
LOOK_DOWN   = "z"
QUIT        = "q"

CONFIG_PATH = '/src/configs/test/test_hm3d.train.rgbd.yaml'
OMEGACONF_PATH = '/src/configs/kemono/kemono_train_config.yaml'
ENV_ID = 'HabitatTrain-v0'

def example():
  habitat_config = kemono.get_config(CONFIG_PATH)
  env = train_env.make(ENV_ID, habitat_config, auto_stop=False)
  config = OmegaConf.load(OMEGACONF_PATH)
  env = SemanticWrapper(env, **config.envs.semantic_wrapper)
  env = SemanticMapBuilderWrapper(
    env,
    **config.envs.semantic_map_builder
  )

  print("Environment creation successful")
  while True:
    observations = env.reset()
    print('Episode id: {}, scene id: {}'.format(env.current_episode.episode_id, env.current_episode.scene_id))

    # --- show observations ---
    print('Observations:')
    print(f'  Object goal: {observations["objectgoal"]}')
    print(f"  GPS: {observations['gps']}")
    print(f"  compass: {observations['compass']}")
    env.render("interact")

    print("Agent stepping around inside environment.")

    count_steps = 0
    done = False
    while not done:
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
        print("INVALID KEY")
        continue
      
      observations, reward, done, info = env.step(action)
      count_steps += 1
      # --- show observations ---
      print('Observations:')
      print(f"  GPS: {observations['gps']}")
      print(f"  compass: {observations['compass']}")
      print(f"Rewards: {reward}")

      env.render("interact")

      metrics = info["metrics"]
      goal = info["goal"]
      print('Metrics:')
      print('  distance to goal: {:.3f}'.format(metrics['distance_to_goal']))
      print('  success: {}, spl: {:.3f}, softspl: {:.3f}'.format(
              metrics['success'] == 1.0,
              metrics['spl'], metrics['softspl']))
      print(f'  goal id: {goal["id"]}/{goal["name"]}')

    print("Episode finished after {} steps.".format(count_steps))


if __name__ == "__main__":
  example()