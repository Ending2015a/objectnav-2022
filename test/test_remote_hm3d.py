# --- built in ---
import os
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

@ray.remote(num_cpus=0)
class _RemoteWrapper():
  def __init__(self, config, show=False):
    print(socket.gethostbyname(socket.gethostname()))
    self.show = show
    self.env = habitat.Env(config=config)

  def step(self, action):
    observations = self.env.step(action)
    if self.show:
      cv2.imshow("RGB", observations['rgb'][...,::-1])

  def reset(self):
    observations = self.env.reset()
    if self.show:
      cv2.imshow("RGB", observations['rgb'][...,::-1])
    return observations

  def call(self, func, *args, **kwargs):
    return getattr(self.env, func)(*args, **kwargs)

  def get_annotations(self):
    return self.env.sim.semantic_annotations()

  def getattr(self, name):
    return getattr(self.env, name)

  def setattr(self, name, value):
    return setattr(self.env, name, value)


class RemoteHabitat():
  def __init__(self, config):
    self.env = _RemoteWrapper.remote(config)
  
  def step(self, action):
    return ray.get(self.env.step.remote(action))
  
  def reset(self):
    return ray.get(self.env.reset.remote())

  def get_annotations(self):
    return ray.get(self.env.get_annotations.remote())

  @property
  def current_episode(self):
    return ray.get(self.env.getattr.remote('current_episode'))
  
  @property
  def episode_over(self):
    return ray.get(self.env.getattr.remote('episode_over'))
  
  @property
  def get_metrics(self):
    return ray.get(self.env.call.remote('get_metrics'))

def show_sensors(semantic_mapping, observations):
  global semantic
  rgb = observations['rgb']
  depth = observations['depth']
  semantic = observations['semantic']
  #print(semantic.dtype)
  # depth map (h, w, 1) -> (h, w, 3)
  depth = (np.concatenate((depth,)*3, axis=-1) * 255.0).astype(np.uint8)
  # color image rgb -> bgr
  bgr = rgb[:, :, [2,1,0]]
  # semantic
  seg = semantic_mapping.get_categorical_map(semantic)
  scene = np.concatenate((bgr, depth), axis=1)
  cv2.imshow("rgb+depth", scene)
  cv2.imshow("semantic", seg)
  return scene

def example():
  config = lib.get_config(CONFIG_PATH)
  remote_env = RemoteHabitat.remote(config)

  global semantic_mapping
  semantic_mapping = SemanticMapping(remote_env.get_annotations())
  semantic_mapping.print_semantic_meaning()

  print("Environment creation successful")
  while True:
    observations = remote_env.reset()
    exit(1)
    print('Episode id: {}, scene id: {}'.format(
      remote_env.current_episode.episode_id, remote_env.current_episode.scene_id))

      # --- show observations ---
    print('Observations:')
    print(f'  Object goal: {observations["objectgoal"]}')
    print(f"  GPS: ({observations['gps'][1]:.5f}, {observations['gps'][0]:.5f}), compass: {observations['compass'][0]:.5f}")

    show_sensors(semantic_mapping, observations)

    print("Agent stepping around inside environment.")

    count_steps = 0
    while not remote_env.episode_over:
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

      observations = remote_env.step(action)
      count_steps += 1
      # --- show observations ---
      print('Observations:')
      print(f"  GPS: ({observations['gps'][1]:.5f}, {observations['gps'][0]:.5f}), compass: {observations['compass'][0]:.5f}")

      show_sensors(semantic_mapping, observations)

      metrics = remote_env.get_metrics()
      print('Metrics:')
      print('  distance to goal: {:.3f}'.format(metrics['distance_to_goal']))
      print('  success: {}, spl: {:.3f}, softspl: {:.3f}'.format(
              metrics['success'] == 1.0,
              metrics['spl'], metrics['softspl']))

    print("Episode finished after {} steps.".format(count_steps))


if __name__ == '__main__':
  ray.init(TODO)
  example()