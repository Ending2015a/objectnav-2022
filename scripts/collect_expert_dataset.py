# --- built in ---
import os
import argparse
import tqdm
import logging
# --- 3rd party ---
import cv2
import gym
import habitat
from omegaconf import OmegaConf
import numpy as np
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
import rlchemy
# --- my module ---
import kemono
from kemono.envs import habitat_env
from kemono.envs.wrap import (
  SemanticMapBuilderWrapper,
  SemanticWrapper,
  CleanObsWrapper
)
from kemono.envs.control import (
  ManualController,
  ShortestPathController
)

CONFIG_PATH = '/src/configs/test/test_hm3d.{}.rgbd.yaml'
OMEGACONF_PATH = '/src/configs/kemono/kemono_train_config.yaml'

def set_logging(rootpath):
  logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(rootpath, "logging.txt")),
        logging.StreamHandler()
    ]
  )

def set_seed(seed):
  import random
  random.seed(seed)
  np.random.seed(seed)

def get_shuffle_goals(episode):
  import random
  goals = list(episode.goals)
  random.shuffle(goals)
  return goals

def get_nearest_goals(episode):
  init_pos = np.asarray(episode.start_position)
  goal_pos = np.asarray([goal.position for goal in episode.goals])
  goal_dist = np.linalg.norm(goal_pos - init_pos, axis=-1)
  goal_y_dist = np.floor(np.abs(goal_pos - init_pos))[..., 1]
  # Note: first sort by goal_y_dist then sort by goal_dist
  goal_ids = np.lexsort((goal_dist, goal_y_dist))
  goals = [episode.goals[goal_id] for goal_id in goal_ids]
  return goals

def example(args):
  os.makedirs(args.save_path, exist_ok=True)
  set_logging(args.save_path)
  logging.info(f'{args}')
  config_path = CONFIG_PATH.format(args.split)
  habitat_config = kemono.get_config(config_path)
  habitat_config.defrost()
  habitat_config.TASK.SUCCESS.SUCCESS_DISTANCE = args.goal_radius
  habitat_config.freeze()
  config = OmegaConf.load(OMEGACONF_PATH)
  OmegaConf.resolve(config)
  logging.info(OmegaConf.to_yaml(config))
  env = habitat_env.make(
    config.envs.env_id,
    habitat_config,
    **config.envs.habitat_env
  )
  env = SemanticWrapper(env, **config.envs.semantic_wrapper)
  env = SemanticMapBuilderWrapper(
    env,
    **config.envs.semantic_map_builder
  )
  env = CleanObsWrapper(
    env,
    **config.envs.clean_obs
  )
  env = rlchemy.envs.Monitor(
    env,
    root_dir = args.save_path,
    video = True,
    video_kwargs = {'interval': 1, 'fps': 10}
  )
  env.add_tool(rlchemy.envs.TrajectoryRecorder(interval=1))

  if args.manual:
    controller = ManualController(env)
  else:
    controller = ShortestPathController(env, goal_radius=args.goal_radius)

  if args.manual and args.render is None:
    args.render = 'human'

  print("Start collecting data...")
  print("  Split:", args.split)
  print("  Config path:", config_path)
  print("  Save path:", args.save_path)
  print("  Episodes:", args.episodes)
  print("  Min ep timesteps:", args.min_ep_timesteps)
  print("  Goal radius:", args.goal_radius)
  print("  Seed:", args.seed)

  set_seed(args.seed)
  total_episodes = 0
  total_timesteps = 0
  episode_bar = tqdm.tqdm(total=args.episodes, position=0, leave=True)
  while True:
    episode_bar.update()
    if (total_episodes >= args.episodes):
      break
    total_episodes += 1
    observations = env.reset()
    ep_timesteps = 0
    goals = get_nearest_goals(env.habitat_env.current_episode)
    goal_iter = iter(goals)
    goal = next(goal_iter)
    controller.reset(goal=goal)
    print('Episode id: {}, scene id: {}'.format(env.current_episode.episode_id, env.current_episode.scene_id))

    # --- show observations ---
    if args.render is not None:
      env.render(args.render)
    if args.manual:
      print('Observations:')
      print(f'  Object goal: {observations["objectgoal"]}')

    done = False
    force_stop = False
    while not done:
      action = controller.act(observations)
      if action is None:
        if ep_timesteps < args.min_ep_timesteps:
          try:
            goal = next(goal_iter)
            controller.reset(goal=goal)
            continue
          except StopIteration:
            pass
        force_stop = True
      if not force_stop:
        observations, reward, done, info = env.step(action)
      else:
        observations, reward, done, info = env.force_stop()
        done = True
      ep_timesteps += 1
      total_timesteps += 1

      # rgb = np.transpose(observations['rgb'], (1, 2, 0))
      # cv2.imshow('rgb', rgb[...,::-1])
      # rgb = np.transpose(observations['small_map'], (1, 2, 0))
      # cv2.imshow('small_map', rgb[...,::-1])
      # rgb = np.transpose(observations['large_map'], (1, 2, 0))
      # cv2.imshow('large_map', rgb[...,::-1])

      if args.render is not None:
        env.render(args.render)
      if args.manual:
        metrics = info["metrics"]
        goal = info["goal"]
        print(f"Rewards: {reward:.6f}")
        print('Metrics:')
        print('  distance to goal: {:.3f}'.format(metrics['distance_to_goal']))
        print('  success: {}, spl: {:.3f}, softspl: {:.3f}'.format(
                metrics['success'] == 1.0,
                metrics['spl'], metrics['softspl']))
        print(f'  goal id: {goal["id"]}/{goal["name"]}')

  env.close()

  print("Data collected:")
  print("  Split:", args.split)
  print("  Config path:", config_path)
  print("  Save path:", args.save_path)
  print("  Episodes:", args.episodes)
  print("  Min ep timesteps:", args.min_ep_timesteps)
  print("  Goal radius:", args.goal_radius)
  print("  Seed:", args.seed)
  print("  Total samples:", total_timesteps)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-s', '--split', choices=['train', 'val', 'val_mini'], default='val_mini')
  parser.add_argument('--episodes', type=int, default=10)
  parser.add_argument('--min_ep_timesteps', type=int, default=0)
  parser.add_argument('--goal_radius', type=float, default=0.05)
  parser.add_argument('--save_path', type=str, default='/src/logs/kemono_expert/')
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--manual', action='store_true', default=False)
  parser.add_argument('--render', type=str, choices=['interact', 'human'], default=None)
  parser.add_argument('--interact', action='store_true', default=False)
  args = parser.parse_args()
  example(args)
