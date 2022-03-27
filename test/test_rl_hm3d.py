# --- built in ---
from dataclasses import dataclass
# --- 3rd party ---
import cv2
import gym
import habitat
import numpy as np
from habitat.sims.habitat_simulator.actions import HabitatSimActions
# --- my module ---
import kemono
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

@dataclass
class TrainEnvSpec():
  id: str = "Habitat_reward-v0"

class TrainEnv(habitat.RLEnv):
  metadata = {"render_modes": ['rgb_array', 'human', 'interact']}
  reward_range = {-float("inf"), float("inf")}
  spec = TrainEnvSpec()

  def __init__(
    self,
    config: habitat.Config,
    dataset: habitat.Dataset = None
  ):
    super().__init__(config=config, dataset=dataset)
    self.config = config
    self.tilt_angle = config.SIMULATOR.TILT_ANGLE
    self._agent_tilt_angle = 0
    self._cached_obs = None
    self.observation_space = self.make_observation_space()

  @property
  def dataset(self) -> habitat.Dataset:
    return self._env._dataset

  @property
  def sim(self) -> habitat.Simulator:
    return self._env.sim

  def step(self, action):
    obs = self._env.step(action)
    # augmenting compass with agent's pitch
    if action == HabitatSimActions.LOOK_UP:
      self._agent_tilt_angle += self.tilt_angle
    elif action == HabitatSimActions.LOOK_DOWN:
      self._agent_tilt_angle -= self.tilt_angle
    obs = self.get_observations(obs)
    # get done, rew, info, ...
    done = self.get_done(obs)
    info = self.get_info(obs)
    rew = self.get_reward(obs, done, info)
    return obs, rew, done, info

  def reset(self):
    obs = self._env.reset()
    self._agent_tilt_angle = 0
    obs = self.get_observations(obs)
    return obs

  def make_observation_space(self):
    # augmenting compass
    # compass [heading, pitch]
    obs_space = self._env.observation_space
    compass_space = obs_space['compass']
    new_compass_space = gym.spaces.Box(
      high = compass_space.high.max(),
      low = compass_space.low.min(),
      shape = (2,),
      dtype = compass_space.dtype
    )
    new_obs_spaces = {key: obs_space[key] for key in obs_space}
    new_obs_spaces['compass'] = new_compass_space
    new_obs_space = gym.spaces.Dict(new_obs_spaces)
    return new_obs_space

  def get_observations(self, obs):
    # compass: [heading, pitch]
    compass = obs['compass']
    pitch = np.radians(self._agent_tilt_angle)
    obs['compass'] = np.concatenate((compass, [pitch]), axis=0)
    self._cached_obs = obs
    return obs

  def get_done(self, obs):
    return self._env.episode_over

  def get_reward_range(self):
    return [-float('inf'), float('inf')]

  def get_reward(self, obs, done, info):
    return 0

  def get_info(self, obs):
    """Get environment info
    Available key:
    * metrics
      * distance_to_goal: nearest goal
      * success
      * spl
      * softspl

    Returns:
        _type_: _description_
    """
    metrics = self._env.get_metrics()
    return {'metrics': metrics}

  def render(self, mode="human"):
    if self._cached_obs is None:
      return
    scene = self.render_scene()
    if mode == 'rgb_array':
      return scene
    else:
      cv2.imshow("rgb + depth", scene)

  def render_scene(self):
    assert self._cached_obs is not None
    obs = self._cached_obs
    rgb = obs['rgb']
    depth = obs['depth']
    depth = (np.concatenate((depth,)*3, axis=-1) * 255.0).astype(np.uint8)
    bgr = rgb[...,::-1]
    scene = np.concatenate((bgr, depth), axis=1)
    return scene

def example():
  config = kemono.get_config(CONFIG_PATH)
  env = TrainEnv(config)
  env = SemanticWrapper(env, use_ground_truth=True)
  env = MapBuilderWrapper(env)

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