# --- built in ---
import os
import sys
import time
# --- 3rd party ---
import cv2
import gym
import habitat
import numpy as np
from habitat.sims.habitat_simulator.actions import HabitatSimActions
# --- my module ---
import kemono

CONFIG_PATH = '/src/configs/test/test_hm3d.val_mini.rgbd.yaml'

FORWARD_KEY = "w"
LEFT_KEY    = "a"
RIGHT_KEY   = "d"
FINISH      = "f"
QUIT        = "q"
SAMPLE      = "e"

topdown_map = None
agent_height = None

meters_per_pixel = 0.1

sampled_point = None


def sample_goal(env):
  global topdown_map
  global sampled_point

  height, width = topdown_map.shape

  available_points = []
  for h in range(height):
    for w in range(width):
      if topdown_map[h, w]:
        available_points.append((w, h))

  point_idx = np.random.choice(np.arange(len(available_points)))
  sampled_point = np.asarray(available_points[point_idx])


def get_topdown_map(env):
  global topdown_map
  global agent_height

  start_time = time.time()
  topdown_map = env.sim.pathfinder.get_topdown_view(meters_per_pixel, agent_height)
  print('time:', time.time() - start_time)


def plot_scene(env, observations):
  global topdown_map
  rgb = observations['rgb']
  depth = observations['depth']

  depth = (np.concatenate((depth,)*3, axis=-1) * 255.0).astype(np.uint8)
  cv2.imshow("rgb", rgb[...,::-1])
  cv2.imshow("depth", depth)

  topdown_im = np.ones((*topdown_map.shape, 3), dtype=np.float32)

  topdown_im[topdown_map == True] = np.asarray([0.5, 0.5, 0.5])
  topdown_im[topdown_map == False] = np.asarray([0.0, 0.0, 0.0])

  topdown_im = (topdown_im * 255.0).astype(np.uint8)

  agent_pos = env.sim.get_agent(0).state.position
  agent_rot = env.sim.get_agent(0).state.rotation

  bounds = env.sim.pathfinder.get_bounds()
  px = (agent_pos[0] - bounds[0][0]) / meters_per_pixel
  py = (agent_pos[2] - bounds[0][2]) / meters_per_pixel

  goal_x = sampled_point[0] * meters_per_pixel + bounds[0][0]
  goal_y = sampled_point[1] * meters_per_pixel + bounds[0][2]

  # plot agent
  topdown_im = cv2.circle(topdown_im, (int(px), int(py)),
    radius=2, color=(255, 0, 0), thickness=4)

  # plot random goal
  topdown_im = cv2.circle(topdown_im, (int(sampled_point[0]), int(sampled_point[1])),
    radius=2, color=(0, 0, 255), thickness=4)

  cv2.imshow("topdown", topdown_im)

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

def example():
  global topdown_map
  global agent_height
  config = kemono.get_config(CONFIG_PATH)
  env = habitat.Env(config=config)
  while True:
    observations = env.reset()

    agent_height = env.sim.get_agent(0).state.position[1]
    get_topdown_map(env)
    sample_goal(env)
    plot_scene(env, observations)

    while not env.episode_over:
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
      elif keystroke == ord(SAMPLE):
        print("Sample point")
        sample_goal(env)
        continue
      elif keystroke == ord(QUIT):
        print('Exit simulator')
        exit(0)
      else:
        print("INVALID KEY")
        continue

      observations = env.step(action)
      height = env.sim.get_agent(0).state.position[1]
      # replot topdown map
      if abs(agent_height-height) > 1.0:
        agent_height = height
        get_topdown_map(env)

      plot_scene(env, observations)



if __name__ == '__main__':
  example()