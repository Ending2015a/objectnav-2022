# --- built in ---
import os
import copy
from typing import (
  Any,
  Dict,
  Tuple,
  Optional
)
# --- 3rd party ---
import numpy as np
import cv2
import rlchemy
from rlchemy.lib.envs.monitor import Monitor, MonitorToolChain
# --- my module ---

__all__ = [
  'RedNetDatasetCollectTool'
]

class RedNetDatasetCollectTool(MonitorToolChain):
  dataset_path: str = 'dataset/{feature}/{scene_id}/episode_id-{episode_id}/'
  image_ext: str = 'png'
  semantic_ext: str = 'npy'
  dataset_suffix: str = '{episode_steps:03d}'
  def __init__(
    self,
    root_dir: Optional[str] = None,
    prefix: Optional[str] = None,
    force: bool = True,
    enabled: bool = True
  ):
    """Monitor tools for collecting rgb/depth/segmentation data for
    training RedNet.
    The data are saved to
    {root_dir}/{prefix}.{dataset_suffix}.{image_ext}

    In default, the data paths are:
    RGB:
      dataset/rgb/{scene_id}-epid{episode_id}/{steps:03d}.png

    Depth:
      dataset/depth/{scene_id}-epid{episode_id}/{steps:03d}.png

    Segmentation:
      dataset/rgb/{scene_id}-epid{episode_id}/{steps:03d}.npy

    Note that depth images are saved in 16bit grayscale format.
    segmentations in default are the mpcat40 category ids of the scene.

    Args:
      root_dir (str, optional): _description_. Defaults to None.
      prefix (str, optional): _description_. Defaults to None.
      force (bool, optional): _description_. Defaults to True.
      enabled (bool, optional): _description_. Defaults to True.
    """    
    super().__init__()

    self._root_dir = root_dir
    self._prefix = prefix
    self._suffix = self.dataset_suffix
    self._image_ext = self.image_ext
    self._semantic_ext = self.semantic_ext
    self._force = force
    self._enabled = enabled
    self._episodic_steps = 0

  def set_monitor(self, monitor: Monitor):
    super().set_monitor(monitor)
    # Default save root: {monitor.root_dir}/{dataset_path}
    self._root_dir = (self._root_dir
      or os.path.join(self._monitor.root_dir, self.dataset_path))
    # Default prefix: {monitor.prefix}
    self._prefix = (self._prefix or self._monitor.prefix)
    # Raise an exception if the path is not empty
    if not self._force:
      if not self._path_is_empty(self._root_dir):
        raise RuntimeError("Try to write to non-empty dataset "
          f"directory {self._root_dir}")

  def _create_path(
    self,
    dirpath: Optional[str] = None,
    filepath: Optional[str] = None
  ):
    rlchemy.utils.safe_makedirs(dirpath=dirpath, filepath=filepath)

  def _before_step(self, act: Any) -> Any:
    self._episodic_steps += 1
    return act

  def _after_step(self, obs: Any, rew: Any, done: Any, info: Any) -> Any:
    if not self._enabled:
      return (obs, rew, done, info)
    self._save_data(obs)
    return (obs, rew, done, info)

  def _before_reset(self, **kwargs) -> Any:
    self._episodic_steps = 0
    return kwargs

  def _after_reset(self, obs) -> Any:
    """The episode info (stats) updates here"""
    if not self._enabled:
      return obs
    self._save_data(obs)
    return obs

  def close(self):
    if not self._enabled:
      return
    self._enabled = False
  
  def _habitat_episode_info(self):
    episode = self._env.habitat_env.current_episode
    scene_id = episode.scene_id
    scene_id = os.path.basename(scene_id).split('.')[0]
    episode_id = episode.episode_id
    object_category = episode.object_category
    return {
      'scene_id': scene_id,
      'episode_id': episode_id,
      'object_category': object_category
    }

  def _save_data(self, obs):
    if 'rgb' in obs:
      rgb_path = self._save_rgb_data(obs['rgb'])
    if 'depth' in obs:
      depth_path = self._save_depth_data(obs['depth'])
    if 'seg' in obs:
      seg_path = self._save_semantic_data(obs['seg'])

  def _save_rgb_data(self, rgb):
    path = self._make_path(
      root_dir = self._root_dir,
      prefix = self._prefix,
      suffix = self._suffix,
      ext = self._image_ext,
      macro_dict = dict(
        feature = 'rgb',
        episode_steps = self._episodic_steps
      )
    )
    self._create_path(filepath=path)
    # to bgr
    bgr = rgb[...,::-1]
    cv2.imwrite(path, bgr)
    return path

  def _save_depth_data(self, depth):
    path = self._make_path(
      root_dir = self._root_dir,
      prefix = self._prefix,
      suffix = self._suffix,
      ext = self._image_ext,
      macro_dict = dict(
        feature = 'depth',
        episode_steps = self._episodic_steps
      )
    )
    self._create_path(filepath=path)
    # to 16bit depth
    # to read the data:
    #   depth = cv2.imread(path, -1)
    #   depth = depth.astype(np.float32)/65536
    # note that the shape of loaded depth is in (height, width)
    depth = np.clip(depth, 0.0, 1.0)
    depth = (depth * 65536).astype(np.uint16)
    cv2.imwrite(path, depth)
    return path

  def _save_semantic_data(self, seg):
    path = self._make_path(
      root_dir = self._root_dir,
      prefix = self._prefix,
      suffix = self._suffix,
      ext = self._semantic_ext,
      macro_dict = dict(
        feature = 'seg',
        episode_steps = self._episodic_steps
      )
    )
    self._create_path(filepath=path)
    # to npy
    np.save(path, seg)
    return path
  
  def _make_path(
    self,
    root_dir: str,
    prefix: Optional[str],
    suffix: Optional[str],
    ext: str,
    macro_dict: Dict[str, str] = {}
  ) -> str:
    paths = []
    if prefix:
      paths.append(str(prefix))
    if suffix:
      paths.append(str(suffix))
    paths.append(str(ext))

    filename = '.'.join(paths)
    path = os.path.join(root_dir, filename)
    abspath = os.path.abspath(path)
    episode_info = self._habitat_episode_info()
    stats_dict = copy.deepcopy(self.stats)
    stats_dict.update(episode_info)
    stats_dict.update(macro_dict)
    return abspath.format(**stats_dict)

  def _path_is_empty(self, path: str) -> bool:
    """Return True if the path (dir) does not exist,
    or it's empty.
    """
    if not path:
      return True
    return ((not os.path.isdir(path))
      or (len(os.listdir(path)) == 0))