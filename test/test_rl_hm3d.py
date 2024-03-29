# --- built in ---
from dataclasses import dataclass
# --- 3rd party ---
import cv2
import gym
import habitat
from omegaconf import OmegaConf
import numpy as np
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.datasets import make_dataset
# --- my module ---
import kemono
from kemono.envs import habitat_env
from kemono.envs.wrap import (
  SemanticMapObserver,
  SemanticMapBuilderWrapper,
  SemanticWrapper,
  PlannerWrapper,
  CleanObsWrapper
)
from kemono.envs.control import ManualController

CONFIG_PATH = '/src/configs/test/test_hm3d.val_mini.rgbd.yaml'
OMEGACONF_PATH = '/src/configs/kemono/kemono_train_sac2.yaml'
ENV_ID = 'HabitatEnv-v0'


def render_scene(obs):
  small_map = obs['small_map']
  for i in range(len(small_map)):
    layer_map = np.stack((small_map[i],)*3, axis=-1)
    layer_map = 255 - (layer_map * 255).astype(np.uint8)
    cv2.imshow(f'layer {i} map', layer_map)


def example():
  config = kemono.utils.load_config(OMEGACONF_PATH, resolve=True)
  habitat_config = kemono.get_config(CONFIG_PATH)
  # habitat_config.defrost()
  # habitat_config.ENVIRONMENT.ITERATOR_OPTIONS.CYCLE = False
  # habitat_config.freeze()
  dataset = make_dataset(
    id_dataset = habitat_config.DATASET.TYPE,
    config = habitat_config.DATASET
  )
  # dataset.episodes = dataset.episodes[:3]
  env = habitat_env.make(
    ENV_ID,
    habitat_config,
    dataset,
    auto_stop = False,
    enable_pitch = True,
    enable_stop = True
  )
  env = SemanticWrapper(env, **config.envs.semantic_wrapper)
  env = SemanticMapBuilderWrapper(
    env,
    **config.envs.semantic_map_builder
  )
  env = SemanticMapObserver(
    env,
    **config.envs.semantic_map_observer
  )
  env = PlannerWrapper(
    env,
    **config.envs.planner
  )
  env = SemanticMapObserver(
    env,
    **config.envs.semantic_map_observer2
  )
  # env = CleanObsWrapper(
  #   env,
  #   **config.envs.clean_obs
  # )
  controller = ManualController(env)
  print("Environment creation successful")
  while True:
    observations = env.reset()


    controller.reset()
    print('Episode id: {}, scene id: {}'.format(env.current_episode.episode_id, env.current_episode.scene_id))

    # --- show observations ---
    print('Observations:')
    #print(f'  Object goal: {observations["objectgoal"]}')
    # print(f"  GPS: {observations['gps']}")
    # print(f"  compass: {observations['compass']}")
    env.render("interact")
    #render_scene(observations)

    print("Agent stepping around inside environment.")

    count_steps = 0
    done = False
    while not done:
      action = controller.act(observations)
      observations, reward, done, info = env.step(action)
      count_steps += 1
      # --- show observations ---
      print('Observations:')
      #print(f"  GPS: {observations['gps']}")
      #print(f"  compass: {observations['compass']}")
      # print(f"  plan: ({observations['plan_distance']}, "
      #       f"{observations['plan_angle']}, "
      #       f"{observations['plan_time']})")
      print(f"Rewards: {reward}")
      env.render("interact")
      #render_scene(observations)

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