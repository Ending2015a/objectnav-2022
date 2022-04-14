# --- built in ---
import argparse
import tqdm
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
from kemono.envs import train_env
from kemono.envs.wrap import (
  SemanticMapBuilderWrapper,
  SemanticWrapper,
  CleanObsWrapper
)

FORWARD_KEY = "w"
LEFT_KEY    = "a"
RIGHT_KEY   = "d"
FINISH      = "f"
LOOK_UP     = "x"
LOOK_DOWN   = "z"
QUIT        = "q"

CONFIG_PATH = '/src/configs/test/test_hm3d.val_mini.rgbd.yaml'
OMEGACONF_PATH = '/src/configs/kemono/kemono_train_config.yaml'
ENV_ID = 'HabitatTrain-v0'
GOAL_RADIUS = 0.05

class ShortestPathController:
  def __init__(self, env, goal_radius=0.1):
    self.follower = ShortestPathFollower(env.sim, goal_radius, False)
    self.goal = None
  
  def reset(self, goal):
    self.goal = goal

  def step(self):
    keystroke = cv2.waitKey(1)
    if keystroke == ord(QUIT):
      print('Exit simulator')
      exit(0)
    action = self.follower.get_next_action(self.goal.position)
    if action is None:
      action = HabitatSimActions.STOP
    return action

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
  habitat_config = kemono.get_config(CONFIG_PATH)
  habitat_config.defrost()
  habitat_config.TASK.SUCCESS.SUCCESS_DISTANCE = GOAL_RADIUS
  habitat_config.freeze()
  env = train_env.make(ENV_ID, habitat_config, auto_stop=args.manual)
  config = OmegaConf.load(OMEGACONF_PATH)
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
    controller = ShortestPathController(env, GOAL_RADIUS)

  print("Start collecting data...")
  print("  Split:", args.split)
  print("  Config path:", CONFIG_PATH)
  print("  Save path:", args.save_path)
  print("  Episodes:", args.episodes)
  print("  Min ep timesteps:", args.min_ep_timesteps)
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
    goals = get_shuffle_goals(env.habitat_env.current_episode)
    goal_iter = iter(goals)
    goal = next(goal_iter)
    controller.reset(goal)
    print('Episode id: {}, scene id: {}'.format(env.current_episode.episode_id, env.current_episode.scene_id))

    # --- show observations ---
    if args.manual or args.interact:
      env.render("interact")
    if args.manual:
      print('Observations:')
      print(f'  Object goal: {observations["objectgoal"]}')

    done = False
    while not done:
      action = controller.step()
      if action == HabitatSimActions.STOP:
        if ep_timesteps < args.min_ep_timesteps:
          try:
            goal = next(goal_iter)
            controller.reset(goal)
            continue
          except StopIteration:
            pass
      observations, reward, done, info = env.step(action)
      ep_timesteps += 1
      total_timesteps += 1
        
      # rgb = np.transpose(observations['rgb'], (1, 2, 0))
      # cv2.imshow('rgb', rgb[...,::-1])
      # rgb = np.transpose(observations['small_map'], (1, 2, 0))
      # cv2.imshow('small_map', rgb[...,::-1])
      # rgb = np.transpose(observations['large_map'], (1, 2, 0))
      # cv2.imshow('large_map', rgb[...,::-1])

      if args.manual or args.interact:
        env.render("interact")
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
  print("  Config path:", CONFIG_PATH)
  print("  Save path:", args.save_path)
  print("  Episodes:", args.episodes)
  print("  Min ep timesteps:", args.min_ep_timesteps)
  print("  Seed:", args.seed)
  print("  Total samples:", total_timesteps)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-s', '--split', choices=['train', 'val', 'val_mini'], default='val_mini')
  parser.add_argument('--episodes', type=int, default=10)
  parser.add_argument('--min_ep_timesteps', type=int, default=0)
  parser.add_argument('--save_path', type=str, default='/src/logs/kemono_expert/')
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--manual', action='store_true', default=False)
  parser.add_argument('--interact', action='store_true', default=False)
  args = parser.parse_args()
  example(args)