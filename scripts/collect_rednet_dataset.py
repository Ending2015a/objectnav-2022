# --- built in ---
import os
import argparse
# --- 3rd party ---
import cv2
import tqdm
import rlchemy
import numpy as np
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
# --- my module ---
import kemono
from kemono.envs import train_env
from kemono.envs import RedNetDatasetCollectTool
from kemono.envs.wrap import (
  SemanticWrapper
)

CONFIG_PATH = '/src/configs/test/test_hm3d.{split}.rgbd.yaml'
ENV_ID = 'HabitatTrain-v0'

GOAL_MAPPING = {
  0: 'chair',
  1: 'bed',
  2: 'plant',
  3: 'toilet',
  4: 'tv_monitor',
  5: 'sofa'
}

# def select_nearest_goal(episode):
#   init_pos = np.asarray(episode.start_position)
#   goal_pos = np.asarray([goal.position for goal in episode.goals])
#   goal_dist = np.linalg.norm(goal_pos - init_pos, axis=-1)
#   goal_y_dist = np.floor(np.abs(goal_pos - init_pos))[..., 1]
#   # Note: first sort by goal_y_dist then sort by goal_dist
#   goal_ids = np.lexsort((goal_dist, goal_y_dist))
#   return episode.goals[goal_ids[0]]

def set_seed(seed):
  import random
  random.seed(seed)
  np.random.seed(seed)

def get_shuffle_goals(episode):
  import random
  goals = list(episode.goals)
  random.shuffle(goals)
  return goals

def example(args):
  global CONFIG_PATH
  seed = args.seed
  split = args.split
  min_episodes = args.min_episodes
  min_timesteps = args.min_timesteps
  min_ep_timesteps = args.min_ep_timesteps
  interact = args.interact
  save_path = os.path.join(args.save_path, split)
  CONFIG_PATH = CONFIG_PATH.format(split=split)
  # create configurations
  config = kemono.get_config(CONFIG_PATH)
  # create environment
  env = train_env.make(
    ENV_ID,
    config,
    auto_stop = False,
    make_act_space = False
  )
  env = SemanticWrapper(
    env,
    goal_mapping = GOAL_MAPPING,
    predictor_type = 'gt',
    colorized = True
  )
  env = rlchemy.envs.Monitor(
    env,
    root_dir = save_path,
    video = True,
    video_kwargs = {'interval': 1, 'fps': 10}
  )
  env.add_tool(RedNetDatasetCollectTool())
  goal_radius = config.TASK.SUCCESS.SUCCESS_DISTANCE or 0.1
  follower = ShortestPathFollower(
    env.sim, goal_radius, False
  )

  print("Start collecting data...")
  print("  Split:", split)
  print("  Config path:", CONFIG_PATH)
  print("  Save path:", save_path)
  print("  Min episodes:", min_episodes)
  print("  Min timesteps:", min_timesteps)
  print("  Min ep timesteps:", min_ep_timesteps)
  print("  Seed:", seed)
  set_seed(seed)
  wait = 1
  total_episodes = 0
  total_timesteps = 0
  episode_bar = tqdm.tqdm(total=min_episodes, position=0, leave=True)
  timestep_bar = tqdm.tqdm(total=min_timesteps, position=1, leave=True)
  while True:
    episode_bar.update()
    if (total_episodes >= min_episodes
        and total_timesteps >= min_timesteps):
      break
    total_episodes += 1
    observations = env.reset()
    if interact:
      env.render('interact')
    ep_timesteps = 0
    done = False
    goals = get_shuffle_goals(env.habitat_env.current_episode)
    goal_iter = iter(goals)
    goal = next(goal_iter)
    while not done:
      if interact:
        key = cv2.waitKey(wait)
        if key == ord(' '):
          wait = (1 if wait == 0 else 0)
      action = follower.get_next_action(goal.position)
      if action is None:
        action = HabitatSimActions.STOP
      if action == HabitatSimActions.STOP:
        if ep_timesteps < min_ep_timesteps:
          try:
            # get next goal
            goal = next(goal_iter)
            continue
          except StopIteration:
            pass
      observations, reward, done, info = env.step(action)
      if interact:
        env.render('interact')
      total_timesteps += 1
      ep_timesteps += 1
      timestep_bar.update()
  episode_bar.close()
  timestep_bar.close()
  env.close()

  print("Data collected:")
  print("  Split:", split)
  print("  Config path:", CONFIG_PATH)
  print("  Save path:", save_path)
  print("  Min episodes:", min_episodes)
  print("  Min timesteps:", min_timesteps)
  print("  Total episodes:", total_episodes)
  print("  Total timesteps:", total_timesteps)
  print("  Min ep timesteps:", min_ep_timesteps)
  print("  Seed:", seed)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-s', '--split', choices=['train', 'val', 'val_mini'], default='val_mini')
  parser.add_argument('--min_episodes', type=int, default=10)
  parser.add_argument('--min_timesteps', type=int, default=1000)
  parser.add_argument('--min_ep_timesteps', type=int, default=0)
  parser.add_argument('--save_path', type=str, default='/src/logs/rednet_dataset/')
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--interact', action='store_true', default=False)
  args = parser.parse_args()
  example(args)