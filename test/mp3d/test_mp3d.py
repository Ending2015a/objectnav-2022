# --- built in ---
import os
import sys
# --- 3rd party ---
import cv2
import habitat
import numpy as np
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.utils.visualizations import maps
# --- my module ---
sys.path.append('/src/')
import lib

FORWARD_KEY = "w"
LEFT_KEY    = "a"
RIGHT_KEY   = "d"
FINISH      = "f"
QUIT        = "q"

CONFIG_PATH = '/src/configs/test/test_mp3d.val_mini.rgbd.yaml'

# bgr
semantic_colors = lib.utils.map3d_color_map(bgr=True)

def print_semantic_meaning(env):
  def print_scene_recur(scene, limit_output=10):
    count = 0
    for level in scene.levels:
      print(
        f"Level id:{level.id}, center:{level.aabb.center},"
        f" dims:{level.aabb.sizes}"
      )
      for region in level.regions:
        print(
          f"Region id:{region.id}, category:{region.category.name()},"
          f" center:{region.aabb.center}, dims:{region.aabb.sizes}"
        )
        for obj in region.objects:
          print(
            f"Object id:{obj.id}, category:{obj.category.index()}/{obj.category.name()},"
            f" center:{obj.aabb.center}, dims:{obj.aabb.sizes}"
          )
          count += 1
          if count >= limit_output:
            return None
  print_scene_recur(env.sim.semantic_annotations(), limit_output=15)

def get_object_category_mapping(env):
  scene = env.sim.semantic_annotations()
  cat_map = {}
  for level in scene.levels:
    for region in level.regions:
      for obj in region.objects:
        obj_id = int(obj.id.split('_')[-1])
        cat_map[obj_id] = obj.category.index()
  return cat_map

def get_category_task_mapping(dataset):
  task_map = {}
  id_map = {}
  for k, v in dataset.category_to_task_category_id.items():
    id_map[v] = k
  for k, v in dataset.category_to_scene_annotation_category_id.items():
    task_map[dataset.category_to_task_category_id[k]] = v
  return task_map, id_map

def object_id_to_category_id(semantic, cat_map):
  # semantic (h, w)
  semantic = np.vectorize(cat_map.get)(semantic)
  return semantic

def colorize_topdown_map(
  topdown_map: np.ndarray,
  fog_of_war_mask = None
):
  _map = maps.TOP_DOWN_MAP_COLORS[topdown_map]
  if fog_of_war_mask is not None:
    fog_of_war_desat_values = np.array([[0.0], [1.0]])
    desat_mask = topdown_map != maps.MAP_INVALID_POINT
    _map[desat_mask] = (
      _map * (1.0 - fog_of_war_desat_values[fog_of_war_mask]) +
      np.array([[0, 255, 255]]) * (fog_of_war_desat_values[fog_of_war_mask])
    ).astype(np.uint8)[desat_mask]
  return _map

def colorize_draw_agent_and_fit_to_height(
  topdown_map_info,
  output_height
):
  topdown_map = topdown_map_info['map']
  topdown_map = colorize_topdown_map(
    topdown_map, topdown_map_info['fog_of_war_mask']
  )
  map_agent_pos = topdown_map_info["agent_map_coord"]
  topdown_map = maps.draw_agent(
    image=topdown_map,
    agent_center_coord=map_agent_pos,
    agent_rotation=topdown_map_info["agent_angle"],
    agent_radius_px=min(topdown_map.shape[0:2]) // 32,
  )

  if topdown_map.shape[0] > topdown_map.shape[1]:
    topdown_map = np.rot90(topdown_map, 1)

  # scale top down map to align with rgb view
  old_h, old_w, _ = topdown_map.shape
  topdown_height = output_height
  topdown_width = int(float(topdown_height) / old_h * old_w)
  # cv2 resize (dsize is width first)
  topdown_map = cv2.resize(
    topdown_map,
    (topdown_width, topdown_height),
    interpolation=cv2.INTER_CUBIC,
  )
  return topdown_map

def make_graph(observations, metrics, cat_map, objectgoal):
  rgb = observations['rgb']
  depth = observations['depth']
  semantic = observations['semantic']
  # depth map (h, w, 1) -> (h, w, 3)
  depth = (np.concatenate((depth,)*3, axis=-1) * 255.0).astype(np.uint8)
  # color image rgb -> bgr
  bgr = rgb[:, :, [2,1,0]]
  # semantic (h, w)
  #semantic = semantic % len(semantic_colors)
  semantic = object_id_to_category_id(semantic, cat_map)
  #seg = np.stack((semantic,)*3, axis=-1).astype(np.uint8)
  seg = semantic_colors[semantic]
  #seg = seg[:, :, [2,1,0]]
  seg[semantic == objectgoal] = [0, 255, 255]
  # draw topdown map
  if metrics is not None:
    topdown_map = colorize_draw_agent_and_fit_to_height(
      metrics['top_down_map'],
      bgr.shape[0]
    )
    scene = np.concatenate((bgr, depth, seg, topdown_map), axis=1)
  else:
    scene = np.concatenate((bgr, depth, seg), axis=1)
  return scene

def example():
  config = lib.get_config(CONFIG_PATH)
  config.defrost()
  config.TASK.TOP_DOWN_MAP.DRAW_BORDER = False
  config.TASK.TOP_DOWN_MAP.DRAW_GOAL_AABBS = False
  config.TASK.TOP_DOWN_MAP.DRAW_VIEW_POINTS = False
  config.freeze()
  dataset = habitat.datasets.make_dataset(
    id_dataset=config.DATASET.TYPE, config=config.DATASET
  )
  env = habitat.Env(
    config=config, dataset=dataset
  )
  print("Environment creation successful")
  print_semantic_meaning(env)
  task_map, id_map = get_category_task_mapping(dataset)
  cat_map = get_object_category_mapping(env)
  while True:
    observations = env.reset()
    print('Episode id: {}, scene id: {}'.format(env.current_episode.episode_id, env.current_episode.scene_id))

    # --- show observations ---
    objectgoal = observations["objectgoal"].item()
    print('Observations:')
    print(f'  Object goal: {id_map[objectgoal]}')
    print(f"  GPS: ({observations['gps'][1]:.5f}, {observations['gps'][0]:.5f}), compass: {observations['compass'][0]:.5f}")

    cv2.imshow('Sensors', make_graph(observations, None, cat_map, task_map[objectgoal]))

    print("Agent stepping around inside environment.")

    count_steps = 0
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
      elif keystroke == ord(QUIT):
        print('Exit simulator')
        exit(0)
      else:
        print("INVALID KEY")
        continue

      observations = env.step(action)
      metrics = env.get_metrics()
      count_steps += 1
      # --- show observations ---
      print('Observations:')
      print(f"  GPS: ({observations['gps'][1]:.5f}, {observations['gps'][0]:.5f}), compass: {observations['compass'][0]:.5f}")

      cv2.imshow('Sensors', make_graph(observations, metrics, cat_map, task_map[objectgoal]))

      
      print('Metrics:')
      print('  distance to goal: {:.3f}'.format(metrics['distance_to_goal']))
      print('  success: {}, spl: {:.3f}, softspl: {:.3f}'.format(
            metrics['success'] == 1.0,
            metrics['spl'], metrics['softspl']))

    print("Episode finished after {} steps.".format(count_steps))


if __name__ == "__main__":
  example()