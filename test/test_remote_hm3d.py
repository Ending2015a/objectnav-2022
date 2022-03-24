# --- built in ---
import os
from re import L
import sys
import socket
# --- 3rd party ---
import cv2
import gym
import habitat
import numpy as np
import ray
from habitat.sims.habitat_simulator.actions import HabitatSimActions
# --- my module ---
sys.path.append('/src/')
import lib
from lib.semantic_mapping import SemanticMapping

FORWARD_KEY = "w"
LEFT_KEY    = "a"
RIGHT_KEY   = "d"
FINISH      = "f"
QUIT        = "q"

CONFIG_PATH = '/src/configs/test/test_hm3d.val_mini.rgbd.yaml'

semantic_mapping = None
semantic = None

@ray.remote(num_cpus=1, num_gpus=0.5)
class _RemoteWrapper():
  def __init__(self, config, show=True):
    self.show = show
    self.env = habitat.Env(config=config)
    self.semantic_mapping = None

  def hello(self):
    """Used to tes remote connection"""
    addr = socket.gethostbyname(socket.gethostname())
    return f"Hello from {addr}"

  def step(self, action):
    observations = self.env.step(action)
    if self.show:
      cv2.imshow("RGB", observations['rgb'][...,::-1])
      cv2.waitKey(1)
    observations = self.get_augmented_observations(observations)
    return observations

  def reset(self):
    observations = self.env.reset()
    self.init_semantic_mapping()
    if self.show:
      cv2.imshow("RGB", observations['rgb'][...,::-1])
      cv2.waitKey(1)
    observations = self.get_augmented_observations(observations)
    return observations

  def init_semantic_mapping(self):
    self.semantic_mapping = SemanticMapping(
      self.get_semantic_annotations(),
      bgr = True
    )
    self.semantic_mapping.print_semantic_meaning()

  def get_semantic_annotations(self):
    return self.env.sim.semantic_annotations()

  def get_augmented_observations(self, observations):
    seg = self.get_colored_seg_observation(observations['semantic'])
    observations['colored_seg'] = seg
    return observations

  def get_colored_seg_observation(self, semantic):
    return self.semantic_mapping.get_categorical_map(semantic)

  def call(self, func, *args, **kwargs):
    return getattr(self.env, func)(*args, **kwargs)

  def _getattr(self, name):
    return getattr(self.env, name)

  def _setattr(self, name, value):
    return setattr(self.env, name, value)


class RemoteHabitat():
  def __init__(self, config):
    self.env = _RemoteWrapper.remote(config)
  
  def hello(self):
    return ray.get(self.env.hello.remote())

  def step(self, action):
    return ray.get(self.env.step.remote(action))
  
  def reset(self):
    return ray.get(self.env.reset.remote())

  def get_metrics(self):
    return ray.get(self.env.call.remote('get_metrics'))

  @property
  def current_episode(self):
    return ray.get(self.env._getattr.remote('current_episode'))
  
  @property
  def episode_over(self):
    return ray.get(self.env._getattr.remote('episode_over'))
  
class VecRemoteHabitat():
  def __init__(self, config, n):
    self.n = n
    self.envs = [self.setup_env(config, i) for i in range(n)]
    self.env = self.envs[0]
  
  def setup_env(self, config, index):
    config = lib.configs.finetune_config(
      config,
      seed = index + 100,
      shuffle = True,
      max_scene_repeat_episodes = 10,
      width = 320,
      height = 240
    )
    return _RemoteWrapper.remote(config)

  def hello(self):
    return ray.get([self.envs[i].hello.remote() for i in range(self.n)])
  
  def step(self, action):
    return ray.get([self.envs[i].step.remote(action) for i in range(self.n)])
  
  def reset(self):
    return ray.get([self.envs[i].reset.remote() for i in range(self.n)])

  def get_metrics(self):
    return ray.get(self.env.call.remote('get_metrics'))

  @property
  def current_episode(self):
    return ray.get(self.env._getattr.remote('current_episode'))
  
  @property
  def episode_over(self):
    return ray.get(self.env._getattr.remote('episode_over'))

def show_sensors(observations, window_idx=0):
  rgb = observations['rgb']
  depth = observations['depth']
  semantic = observations['semantic']
  seg = observations['colored_seg']
  #print(semantic.dtype)
  # depth map (h, w, 1) -> (h, w, 3)
  depth = (np.concatenate((depth,)*3, axis=-1) * 255.0).astype(np.uint8)
  # color image rgb -> bgr
  bgr = rgb[:, :, [2,1,0]]
  # semantic
  #seg = seg[...,::-1]
  scene = np.concatenate((bgr, depth, seg), axis=1)
  cv2.imshow(f"rgb+depth_{window_idx}", scene)
  return scene

def example():
  print('creating environment')
  config = lib.get_config(CONFIG_PATH)
  remote_env = VecRemoteHabitat(config, n=4)
  #remote_env.hello()

  print("Environment creation successful")
  while True:
    observations = remote_env.reset()
    print('Episode id: {}, scene id: {}'.format(
      remote_env.current_episode.episode_id, remote_env.current_episode.scene_id))

      # --- show observations ---
    #print('Observations:')
    #print(f'  Object goal: {observations["objectgoal"]}')
    #print(f"  GPS: ({observations['gps'][1]:.5f}, {observations['gps'][0]:.5f}), compass: {observations['compass'][0]:.5f}")

    for i in range(4):
      show_sensors(observations[i], i)

    print("Agent stepping around inside environment.")

    count_steps = 0
    while not remote_env.episode_over:
      keystroke = cv2.waitKey(1)

      # if keystroke == ord(FORWARD_KEY):
      #   action = HabitatSimActions.MOVE_FORWARD
      #   print("action: FORWARD")
      # elif keystroke == ord(LEFT_KEY):
      #   action = HabitatSimActions.TURN_LEFT
      #   print("action: LEFT")
      # elif keystroke == ord(RIGHT_KEY):
      #   action = HabitatSimActions.TURN_RIGHT
      #   print("action: RIGHT")
      # elif keystroke == ord(FINISH):
      #   action = HabitatSimActions.STOP
      #   print("action: FINISH")
      if keystroke == ord(QUIT):
        print('Exit simulator')
        exit(0)
      # else:
      #   print("INVALID KEY")
      #   continue
      if count_steps % 10 < 5:
        action = HabitatSimActions.TURN_LEFT
      else:
        action = HabitatSimActions.TURN_RIGHT

      observations = remote_env.step(action)
      count_steps += 1
      # --- show observations ---
      #print('Observations:')
      #print(f"  GPS: ({observations['gps'][1]:.5f}, {observations['gps'][0]:.5f}), compass: {observations['compass'][0]:.5f}")

      
      for i in range(4):
        show_sensors(observations[i], i)

      metrics = remote_env.get_metrics()
      print('Metrics:')
      print('  distance to goal: {:.3f}'.format(metrics['distance_to_goal']))
      print('  success: {}, spl: {:.3f}, softspl: {:.3f}'.format(
              metrics['success'] == 1.0,
              metrics['spl'], metrics['softspl']))

    print("Episode finished after {} steps.".format(count_steps))


if __name__ == '__main__':
  ip = os.environ.get("RAY_SERVER_IP")
  port = os.environ.get("RAY_SERVER_PORT")
  ray.init(address=f"ray://{ip}:{port}")
  example()