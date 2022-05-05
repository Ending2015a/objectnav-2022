# --- built in ---
import os
import sys
import time
import logging
# --- 3rd party ---
import cv2
import gym
import habitat
import numpy as np
from habitat.sims.habitat_simulator.actions import HabitatSimActions
import matplotlib.pyplot as plt
import torch
from torch import nn
# --- my module ---
import kemono

CONFIG_PATH = '/src/configs/test/test_hm3d.val_mini.rgbd.yaml'

FORWARD_KEY = "w"
LEFT_KEY    = "a"
RIGHT_KEY   = "d"
FINISH      = "f"
QUIT        = "q"
SAMPLE      = "e"



class Example():
  def __init__(self):
    self.topdown_map = None
    self.geodesic_map = None
    self.agent_height = None
    self.meters_per_pixel = 0.1
    self.sampled_point = None
    self.mesh = None
    self.vec_field = None

  def sample_goal(self, env):
    height, width = self.topdown_map.shape
    available_points = []
    for h in range(height):
      for w in range(width):
        if self.topdown_map[h, w]:
          available_points.append((w, h))

    point_idx = np.random.choice(np.arange(len(available_points)))
    self.sampled_point = np.asarray(available_points[point_idx])

  def get_geodesic_map(self, env):
    height, width = self.topdown_map.shape
    geodesic_map = np.zeros((height, width), dtype=np.float32)
    geodesic_map.fill(-1)
    vec_field = np.zeros((height, width, 2), dtype=np.float32)

    
    bounds = env.sim.pathfinder.get_bounds()
    goal_x = self.sampled_point[0] * self.meters_per_pixel + bounds[0][0]
    goal_y = self.sampled_point[1] * self.meters_per_pixel + bounds[0][2]
    agent_pos = env.sim.get_agent(0).state.position
    goal = np.asarray((goal_x, agent_pos[1], goal_y), dtype=np.float32)
    eps = 0.1
    eps_x = np.asarray((eps, 0., 0.), dtype=np.float32)
    eps_y = np.asarray((0., 0., eps), dtype=np.float32)

    for h in range(height):
      for w in range(width):
        if self.topdown_map[h, w]:
          x = w * self.meters_per_pixel + bounds[0][0]
          y = h * self.meters_per_pixel + bounds[0][2]
          xp = np.asarray((x, agent_pos[1], y), dtype=np.float32)
          gm = env.sim.geodesic_distance(xp, goal)
          ps_x = env.sim.geodesic_distance(xp+eps_x, goal)
          ng_x = env.sim.geodesic_distance(xp-eps_x, goal)
          ps_y = env.sim.geodesic_distance(xp+eps_y, goal)
          ng_y = env.sim.geodesic_distance(xp-eps_y, goal)

          geodesic_map[h, w] = gm
          dgm = np.asarray((ps_y-ng_y, ps_x-ng_x), dtype=np.float32)
          dgm[dgm == np.inf] = 0
          dgm[dgm == -np.inf] = 0
          dgm /= 2 * eps
          vec = -dgm * gm
          vec_field[h, w] = vec
    self.geodesic_map = geodesic_map
    self.vec_field = vec_field
    self.mesh = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    self.mesh = np.stack(self.mesh, axis=-1).astype(np.float32) # (y, x)
    self.mesh = self.mesh.reshape((-1, 2))
    self.vec_field = self.vec_field.reshape((-1, 2))
    mask = np.logical_and(self.vec_field[..., 0]==0, self.vec_field[..., 1]==0)
    self.mesh = self.mesh[~mask]
    self.vec_field = self.vec_field[~mask]

  def get_topdown_map(self, env):
    start_time = time.time()
    self.topdown_map = env.sim.pathfinder.get_topdown_view(self.meters_per_pixel, self.agent_height)
    print('time:', time.time() - start_time)

  def plot_scene(self, env, obs):
    topdown_map = self.topdown_map
    rgb = obs['rgb']
    depth = obs['depth']

    depth = (np.concatenate((depth,)*3, axis=-1) * 255.0).astype(np.uint8)
    cv2.imshow('rgb', rgb[...,::-1])
    cv2.imshow('depth', depth)

    topdown_im = np.ones((*topdown_map.shape, 3), dtype=np.float32)
    topdown_im[topdown_map == True] = np.asarray([0.5, 0.5, 0.5])
    topdown_im[topdown_map == False] = np.asarray([0.0, 0.0, 0.0])
    topdown_im = (topdown_im * 255.0).astype(np.uint8)
    
    agent_pos = env.sim.get_agent(0).state.position
    agent_rot = env.sim.get_agent(0).state.rotation

    bounds = env.sim.pathfinder.get_bounds()
    

    px = (agent_pos[0] - bounds[0][0]) / self.meters_per_pixel
    py = (agent_pos[2] - bounds[0][2]) / self.meters_per_pixel

    goal_x = self.sampled_point[0] * self.meters_per_pixel + bounds[0][0]
    goal_y = self.sampled_point[1] * self.meters_per_pixel + bounds[0][2]

    # plot agent
    topdown_im = cv2.circle(topdown_im, (int(px), int(py)),
        radius=2, color=(255, 0, 0), thickness=4)

    # plot random goal
    topdown_im = cv2.circle(topdown_im, (int(self.sampled_point[0]), int(self.sampled_point[1])),
        radius=2, color=(0, 0, 255), thickness=4)

    plt.close('all')
    plt.figure(figsize=(4, 6), dpi=300)
    plt.imshow(self.geodesic_map, cmap='hot', interpolation='nearest')
    plt.quiver(self.mesh[...,1], self.mesh[...,0], self.vec_field[...,1], self.vec_field[...,0], angles='xy', color='b')
    plt.scatter(self.sampled_point[0], self.sampled_point[1], color='yellow', s=1)
    plt.scatter(px, py, color='r', s=1)
    plt.tight_layout()
    plt.show()

    global_goal = np.asarray((goal_x, agent_pos[1], goal_y))
    from habitat.utils.geometry_utils import (
      quaternion_rotate_vector,
      quaternion_from_coeff
    )
    local_goal = quaternion_rotate_vector(
      agent_rot.inverse(), global_goal - agent_pos
    )
    local_goal = np.asarray((local_goal[0], local_goal[1], -local_goal[2]))
    print('agent pos:', agent_pos)
    print('goal global:', global_goal)
    print('goal local:', local_goal)
    print("is navigable?", env.sim.pathfinder.is_navigable(global_goal))

  def run(self):
    config = kemono.get_config(CONFIG_PATH)
    env = habitat.Env(config=config)
    observations = env.reset()


    self.agent_height = env.sim.get_agent(0).state.position[1]
    self.get_topdown_map(env)
    self.sample_goal(env)
    self.get_geodesic_map(env)
    self.plot_scene(env, observations)

    while not env.episode_over:
      keystroke = cv2.waitKey(0)
      if keystroke == ord(SAMPLE):
        print("Sample point")
        self.sample_goal(env)
        self.get_geodesic_map(env)
        self.plot_scene(env, observations)
        continue
      elif keystroke == ord(QUIT):
        print('Exit simulator')
        exit(0)
      else:
        print("INVALID KEY")
        continue



if __name__ == '__main__':
  Example().run()