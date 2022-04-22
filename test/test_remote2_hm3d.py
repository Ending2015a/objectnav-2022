# --- built in ---
# --- 3rd party ---
import cv2
from omegaconf import OmegaConf
# --- my module ---
import kemono
from kemono.envs import habitat_env
from kemono.envs.wrap import (
  SemanticMapBuilderWrapper,
  SemanticWrapper,
  CleanObsWrapper,
  RayRemoteEnv
)
from kemono.envs.controller import ManualController

CONFIG_PATH = '/src/configs/test/test_hm3d.train.rgbd.yaml'
OMEGACONF_PATH = '/src/configs/kemono/kemono_train_config.yaml'
ENV_ID = 'HabitatTrain-v0'

def create_env(habitat_config, config):
  env = habitat_env.make(
    ENV_ID,
    habitat_config,
    auto_stop = True,
    enable_stop = True,
    enable_pitch = True
  )
  config = OmegaConf.create(config)
  env = SemanticWrapper(env, **config.envs.semantic_wrapper)
  env = SemanticMapBuilderWrapper(
    env,
    **config.envs.semantic_map_builder
  )
  env = CleanObsWrapper(
    env,
    **config.envs.clean_obs
  )
  return env

def example():
  habitat_config = kemono.get_config(CONFIG_PATH)
  config = OmegaConf.load(OMEGACONF_PATH)
  env = RayRemoteEnv(create_env, (habitat_config, config))

  controller = ManualController()

  while True:
    observations = env.reset()
    # --- show observations ---
    print('Observations:')
    print(f"  Objectgoal: {observations['objectgoal']}")

    env.render('human')

    count_steps = 0
    done = False
    while not done:
      action = controller.act()
      observations, reward, done, info = env.step(action)
      count_steps += 1
      # --- show observations ---
      print(f"Rewards: {reward}")

      env.render('human')

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
  kemono.utils.init_ray()
  example()