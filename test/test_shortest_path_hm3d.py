# --- built in ---
# --- 3rd party ---
import cv2
import rlchemy
import numpy as np
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
# --- my module ---
import kemono
from kemono.envs import train_env
from kemono.envs import RedNetDatasetCollectTool
from kemono.envs.wrap import (
  MapBuilderWrapper,
  SemanticWrapper
)

FORWARD_KEY = "w"
LEFT_KEY    = "a"
RIGHT_KEY   = "d"
FINISH      = "f"
LOOK_UP     = "x"
LOOK_DOWN   = "z"
QUIT        = "q"

CONFIG_PATH = '/src/configs/test/test_hm3d.val_mini.rgbd.yaml'
ENV_ID = 'HabitatTrain-v0'

def select_nearest_goal(episode):
  init_pos = np.asarray(episode.start_position)
  goal_pos = np.asarray([goal.position for goal in episode.goals])
  goal_dist = np.linalg.norm(goal_pos - init_pos, axis=-1)
  goal_y_dist = np.floor(np.abs(goal_pos - init_pos))[..., 1]
  # Note: first sort by goal_y_dist then sort by goal_dist
  goal_ids = np.lexsort((goal_dist, goal_y_dist))
  return episode.goals[goal_ids[0]]

def example():
  config = kemono.get_config(CONFIG_PATH)
  env = train_env.make(ENV_ID, config, auto_stop=True)
  env = SemanticWrapper(env, predictor_type='gt')
  env = MapBuilderWrapper(env, draw_goals=True)
  goal_radius = config.TASK.SUCCESS.SUCCESS_DISTANCE or 0.1
  follower = ShortestPathFollower(
    env.sim, goal_radius, False
  )

  print("Environment creation successful")
  for episode_number in range(3):
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
    goal = select_nearest_goal(env.habitat_env.current_episode)
    while not done:
      keystroke = cv2.waitKey(1)

      if keystroke == ord(QUIT):
        print('Exit simulator')
        env.close()
        exit(0)
      
      action = follower.get_next_action(goal.position)
      if action is None:
        action = HabitatSimActions.STOP
      
      observations, reward, done, info = env.step(action)
      count_steps += 1
      # --- show observations ---
      print('Observations:')
      print(f"  GPS: {observations['gps']}")
      print(f"  compass: {observations['compass']}")
      print(f"Rewards: {reward}")

      env.render("interact")

      metrics = info["metrics"]
      goal_info = info["goal"]
      print('Metrics:')
      print('  distance to goal: {:.3f}'.format(metrics['distance_to_goal']))
      print('  success: {}, spl: {:.3f}, softspl: {:.3f}'.format(
              metrics['success'] == 1.0,
              metrics['spl'], metrics['softspl']))
      print(f'  goal id: {goal_info["id"]}/{goal_info["name"]}')

    print("Episode finished after {} steps.".format(count_steps))


if __name__ == "__main__":
  example()