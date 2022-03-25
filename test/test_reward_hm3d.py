# --- built in ---
import os
import sys
from dataclasses import dataclass
# --- 3rd party ---
import cv2
import gym
import habitat
import numpy as np
from habitat.sims.habitat_simulator.actions import HabitatSimActions
# --- my module ---
import kemono
from kemono.semantics import SemanticMapping

FORWARD_KEY = "w"
LEFT_KEY    = "a"
RIGHT_KEY   = "d"
FINISH      = "f"
QUIT        = "q"

CONFIG_PATH = '/src/configs/test/test_hm3d.val_mini.rgbd.yaml'

semantic_mapping = None
semantic = None

@dataclass
class TrainEnvSpec():
  id: str = "Habitat_reward-v1"

class TrainEnv(habitat.RLEnv):
  metadata = {"render_modes": ['rgb_array', 'human', 'interact']}
  reward_range = {-float("inf"), float("inf")}
  spec = TrainEnvSpec()

  def __init__(
    self,
    config: habitat.Config,
    dataset: habitat.Dataset = None,
    enable_semantics: bool = False
  ):
    super().__init__(config=config, dataset=dataset)
    self._cached_obs = None
    self.semap = None
    self.setup_interact = False
    self.semap = SemanticMapping(
      dataset = self._env._dataset
    )
    if enable_semantics:
      self.semap.parse_semantics(
        self._env.sim.semantic_annotations()
      )

  def step(self, action):
    obs = self._env.step(action)
    done = self.get_done(obs)
    info = self.get_info(obs)
    rew = self.get_reward(obs, done, info)
    self._cached_obs = obs
    return obs, rew, done, info

  def reset(self):
    obs = self._env.reset()
    if self.semap.has_semantics:
      self.semap.parse_semantics(
        self._env.sim.semantic_annotations(),
        reset = True
      )
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
    info = {
      'metrics': metrics,
      'goal': {
        'id': self.semap.get_goal_category_id(obs['objectgoal']),
        'name': self.semap.get_goal_category_name(obs['objectgoal'])
      }
    }
    return info

  def render(self, mode="human"):
    if self._cached_obs is None:
      return

    if mode == 'rgb_array':
      scene = self.render_scene()
      return scene
    scene = self.render_scene()
    cv2.imshow("rgb+depth", scene)
    if mode == 'interact':
      seg = self.render_interact()
      cv2.imshow("semantic", seg)

  def render_scene(self):
    assert self._cached_obs is not None
    obs = self._cached_obs
    rgb = obs['rgb']
    depth = obs['depth']
    depth = (np.concatenate((depth,)*3, axis=-1) * 255.0).astype(np.uint8)
    bgr = rgb[...,::-1]
    scene = np.concatenate((bgr, depth), axis=1)
    return scene

  def render_interact(self):
    assert self.semap.has_semantics
    assert self._cached_obs is not None
    obs = self._cached_obs
    if 'semantic' not in obs:
      print("semantic observation does not exist")
      return
    if not self.setup_interact:
      cv2.namedWindow('semantic')
      cv2.setMouseCallback('semantic', self.on_click_semantic_info)
      self.setup_interact = True
    seg = self.semap.get_categorical_map(obs['semantic'], bgr=True)
    return seg

  def on_click_semantic_info(self, event, x, y, flags, param):
    obs = self._cached_obs
    if event == cv2.EVENT_LBUTTONDBLCLK:
      semantic = obs['semantic']
      if semantic is None or self.semap is None:
        return
      obj_id = semantic[y][x].item()
      print(f"On click (x, y) = ({x}, {y})")
      self.semap.print_object_info(obj_id, verbose=True)

def example():
  config = kemono.get_config(CONFIG_PATH)
  env = TrainEnv(config, enable_semantics=True)
  print("Environment creation successful")
  while True:
    observations = env.reset()
    print('Episode id: {}, scene id: {}'.format(env.current_episode.episode_id, env.current_episode.scene_id))

      # --- show observations ---
    print('Observations:')
    print(f'  Object goal: {observations["objectgoal"]}')
    print(f"  GPS: ({observations['gps'][1]:.5f}, {observations['gps'][0]:.5f}), compass: {observations['compass'][0]:.5f}")

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
      print(f"  GPS: ({observations['gps'][1]:.5f}, {observations['gps'][0]:.5f}), compass: {observations['compass'][0]:.5f}")
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