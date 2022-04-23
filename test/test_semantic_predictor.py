# --- built in ---
# --- 3rd party ---
import numpy as np
from omegaconf import OmegaConf
# --- my module ---
import kemono
from kemono.envs import habitat_env
from kemono.envs.wrap import (
  SemanticWrapper
)
from kemono.envs.control import (
  ManualController,
  ShortestPathController
)

CONFIG_PATH = '/src/configs/challenge/challenge_objectnav2022.local.val_mini.rgbd.yaml'
OMEGACONF_PATH = '/src/configs/kemono/kemono_train_sac2.yaml'
ENV_ID = 'HabitatEnv-v0'
RESTORE_PATH = '/src/weights/rednet-50-epoch-10.ckpt'

def get_nearest_goals(episode):
  init_pos = np.asarray(episode.start_position)
  goal_pos = np.asarray([goal.position for goal in episode.goals])
  goal_dist = np.linalg.norm(goal_pos - init_pos, axis=-1)
  goal_y_dist = np.floor(np.abs(goal_pos - init_pos))[..., 1]
  # Note: first sort by goal_y_dist then sort by goal_dist
  goal_ids = np.lexsort((goal_dist, goal_y_dist))
  goals = [episode.goals[goal_id] for goal_id in goal_ids]
  return goals

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
  env = SemanticWrapper(
    env,
    goal_mapping = config.envs.semantic_wrapper.goal_mapping,
    colorized = True,
    predictor_name = 'rednet',
    predictor_kwargs = dict(
      restore_path = RESTORE_PATH,
      goal_logits_scale = 0.25,
      device = 'cuda'
    )
  )
  controller = ShortestPathController(env)

  while True:
    observations = env.reset()
    goals = get_nearest_goals(env.habitat_env.current_episode)
    goal_iter = iter(goals)
    goal = next(goal_iter)
    controller.reset(goal=goal)
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


if __name__ == '__main__':
  example()
