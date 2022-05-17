# --- built in ---
import time
from dataclasses import dataclass
# --- 3rd party ---
import cv2
import gym
import habitat
from omegaconf import OmegaConf
import numpy as np
import dungeon_maps as dmap
from habitat.sims.habitat_simulator.actions import HabitatSimActions
import torch
import einops
from habitat.utils.geometry_utils import (
  quaternion_from_coeff,
  quaternion_rotate_vector,
)
from habitat.tasks.utils import cartesian_to_polar
import matplotlib.pyplot as plt
# --- my module ---
import kemono
from kemono.envs import habitat_env
from kemono.envs.wrap import (
  SemanticMapObserver,
  SemanticMapBuilderWrapper,
  SemanticWrapper,
  CleanObsWrapper
)
from kemono.envs.control import ManualController


CONFIG_PATH = '/src/configs/test/test_hm3d.val_mini.rgbd.yaml'
OMEGACONF_PATH = '/src/configs/envs/kemono_gsm_train.yaml'
ENV_ID = 'HabitatEnv-v0'

hex2rgb = lambda hex: [int(hex[i:i+2], 16) for i in (1, 3, 5)]
CAMERA_COLOR = hex2rgb('#EC5565')

def draw_mark(image, point, color=CAMERA_COLOR, size=6):
  radius = size
  thickness = radius + 2
  image = cv2.circle(image.copy(), (int(point[0]), int(point[1])),
      radius=radius, color=color, thickness=thickness)
  return image

def to_topdown_map_coords(

):
  pass

def from_topdown_map_coords(

):
  pass



def get_topdown_map(
  env,
  meters_per_pixel: float = 0.03,
  to_start: bool = True
):
  map_res = meters_per_pixel
  if to_start:
    episode = env.current_episode
    position = np.asarray(episode.start_position, dtype=np.float32)
    rotation = quaternion_from_coeff(episode.start_rotation)
  else:
    agent = env.sim.get_agent(0)
    position = agent.state.position
    rotation = agent.state.rotation
  topdown_view = env.sim.pathfinder.get_topdown_view(
    map_res, position[1]
  )
  bounds = env.sim.pathfinder.get_bounds()
  px = (position[0] - bounds[0][0]) / map_res
  pz = (position[2] - bounds[0][2]) / map_res
  # calculate world heading
  direction_vector = np.array([0., 0., -1])
  heading_vector = quaternion_rotate_vector(rotation, direction_vector)
  phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
  # create map projector
  # the map projector state is matched to the world state
  height, width = topdown_view.shape
  proj = dmap.MapProjector(
    width = width,
    height = height,
    hfov = env._config.SIMULATOR.DEPTH_SENSOR.HFOV,
    cam_pose = np.array([0., 0., phi]), # invert cam pose
    width_offset = px, # agent start pixel
    height_offset = height-1 - pz, # the height is inverted
    cam_pitch = 0.0,
    cam_height = 0.0,
    map_res = map_res,
    map_width = width,
    map_height = height,
    to_global = False, # local: agent location is the origin
    device = 'cuda'
  )
  # convert to tensor
  topdown_view_th = torch.tensor(topdown_view, dtype=torch.float32)
  topdown_view_th = einops.repeat(topdown_view_th, 'h w -> 1 1 h w')
  topdown_map = dmap.TopdownMap(
    topdown_map = topdown_view_th,
    height_map = topdown_view_th,
    mask = (topdown_view_th > 0.0),
    map_projector = proj
  )
  return topdown_map

def topdown_map_to_global(
  topdown_map: dmap.TopdownMap,
  meters_per_pixel: int
):
  # rotate
  topdown_map = dmap.maps.fuse_topdown_maps(
    topdown_map,
    map_projector = topdown_map.proj.clone(
      map_res = meters_per_pixel,
      cam_pose = np.array([0., 0., 0.]),
      to_global = True
    )
  )
  return topdown_map


def example():
  config = kemono.utils.load_config(OMEGACONF_PATH, resolve=True)
  habitat_config = kemono.get_config(CONFIG_PATH)
  env = habitat_env.make(
    ENV_ID,
    habitat_config,
    auto_stop = False,
    enable_pitch = True,
    enable_stop = True
  )
  env = SemanticWrapper(env, **config.envs.semantic_wrapper)
  env = SemanticMapBuilderWrapper(
    env,
    **config.envs.semantic_map_builder
  )
  env = SemanticMapObserver(
    env,
    **config.envs.semantic_map_observer
  )
  # env = CleanObsWrapper(
  #   env,
  #   **config.envs.clean_obs
  # )
  controller = ManualController(env)
  print("Environment creation successful")
  while True:
    observations = env.reset()
    agent_height = env.habitat_env.sim.get_agent(0).state.position[1]
    
    start_time = time.time()
    orig_topdown_map = get_topdown_map(
      env.habitat_env, meters_per_pixel = 0.015
    )
    topdown_map = topdown_map_to_global(
      orig_topdown_map,
      meters_per_pixel = 0.03
    )
    print(f"Took {time.time() - start_time} sec to rotate topdown map")

    topdown_view = topdown_map.height_map.cpu().numpy()[0, 0]
    topdown_view[~np.isinf(topdown_view)] = 1.0
    topdown_view[np.isinf(topdown_view)] = 0.0
    topdown_view = np.stack((topdown_view*255,)*3, axis=-1).astype(np.uint8)
    
    orig_topdown_view = orig_topdown_map.height_map.cpu().numpy()[0, 0]
    orig_topdown_view = np.stack((orig_topdown_view*255,)*3, axis=-1).astype(np.uint8)

    controller.reset()
    print('Episode id: {}, scene id: {}'.format(env.current_episode.episode_id, env.current_episode.scene_id))

    gps = observations['gps']
    camera_global_xyz = np.asarray((gps[1], agent_height, gps[0]), dtype=np.float32)
    camera_local = topdown_map.get_coords(
      camera_global_xyz,
      is_global = True
    )

    camera_local_pos = camera_local.cpu().numpy()[0, 0]
    td_view = draw_mark(topdown_view, (camera_local_pos[0], camera_local_pos[1]))

    camera_local = orig_topdown_map.get_coords(camera_global_xyz, is_global=True)
    camera_local_pos = camera_local.cpu().numpy()[0, 0]
    orig_td_view = draw_mark(orig_topdown_view, (camera_local_pos[0], camera_local_pos[1]))

    # --- show observations ---
    cv2.imshow('topdown view (original)', orig_td_view)
    cv2.imshow('topdown view (rotated)', td_view)
    env.render("interact")

    print("Agent stepping around inside environment.")

    count_steps = 0
    done = False
    while not done:
      action = controller.act(observations)
      observations, reward, done, info = env.step(action)
      count_steps += 1

      gps = observations['gps']
      camera_global_xyz = np.asarray((gps[1], agent_height, gps[0]), dtype=np.float32)
      camera_local = topdown_map.get_coords(
        camera_global_xyz,
        is_global = True
      )

      camera_local_pos = camera_local.cpu().numpy()[0, 0]
      td_view = draw_mark(topdown_view, (camera_local_pos[0], camera_local_pos[1]))

      camera_local = orig_topdown_map.get_coords(camera_global_xyz, is_global=True)
      camera_local_pos = camera_local.cpu().numpy()[0, 0]
      orig_td_view = draw_mark(orig_topdown_view, (camera_local_pos[0], camera_local_pos[1]))

      # --- show observations ---
      cv2.imshow('topdown view (original)', orig_td_view)
      cv2.imshow('topdown view (rotated)', td_view)
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