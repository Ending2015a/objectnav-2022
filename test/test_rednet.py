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
from kemono.envs import habitat_env
from kemono.envs.wrap import (
  SemanticMapBuilderWrapper,
  SemanticWrapper
)
from kemono.envs.control import ManualController

CONFIG_PATH = '/src/configs/test/test_hm3d.val_mini.rgbd.yaml'
OMEGACONF_PATH = '/src/configs/kemono/kemono_train_config.yaml'
ENV_ID = 'HabitatEnv-v0'

def example():
  habitat_config = kemono.get_config(CONFIG_PATH)
  env = habitat_env.make(
    ENV_ID,
    habitat_config,
    auto_stop = False,
    enable_pitch = True,
    enable_stop = True
  )
  config = OmegaConf.load(OMEGACONF_PATH)
  raise NotImplementedError("Please specified the restore path")
  env = SemanticWrapper(
    env,
    goal_mapping = config.envs.semantic_wrapper.goal_mapping,
    colorized = True,
    predictor_name = 'model',
    predictor_kwargs = dict(
      restore_path = '', #TODO
      device = 'cuda:0'
    )
  )
  # env = SemanticMapBuilderWrapper(
  #   env,
  #   **config.envs.semantic_map_builder
  # )
  controller = ManualController(env)

  print("Environment creation successful")
  while True:
    observations = env.reset()
    controller.reset()
    print('Episode id: {}, scene id: {}'.format(env.current_episode.episode_id, env.current_episode.scene_id))

    # --- show observations ---
    print('Observations:')
    print(f'  Object goal: {observations["objectgoal"]}')
    print(f"  GPS: {observations['gps']}")
    print(f"  compass: {observations['compass']}")
    env.render("human")

    print("Agent stepping around inside environment.")

    count_steps = 0
    done = False
    while not done:
      action = controller.act(observations)
      observations, reward, done, info = env.step(action)
      count_steps += 1
      # --- show observations ---
      print('Observations:')
      print(f"  GPS: {observations['gps']}")
      print(f"  compass: {observations['compass']}")
      print(f"Rewards: {reward}")

      env.render("human")

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