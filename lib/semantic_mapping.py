# --- built in ---
from typing import Dict, Optional, Union
# --- 3rd party ---
import numpy as np
import habitat
from habitat_sim.scene import (
  SemanticLevel,
  SemanticRegion,
  SemanticObject
)
# --- my module ---
from lib import utils

class SemanticMapping():
  def __init__(
    self,
    env: habitat.Env,
    bgr: bool = True,
    default_category: str = 'unknown'
  ):
    """SemanticMapping provides an interface for mapping
    HM3D dataset's semantic meanings to MP3D dataset's
    semantic meanings

    Args:
      env (habitat.Env): habitat environments
      bgr (bool, optional): channel order of the image. BGR or RGB.
        Defaults to True.
      default_category (str, optional): default mpcat40 category for
        unlabeled/unknown objects in HM3D. Defaults to 'unknown'.
    """
    self.semantics = env.sim.semantic_annotations()
    self.object_id_to_object: Dict[int, SemanticObject] = {}
    # ---
    self.parse_semantics()
    self.hm3d_mapping = HM3DMapping(
      self,
      bgr = bgr,
      default_category = default_category
    )

  def parse_semantics(self):
    for level in self.semantics.levels:
      self.parse_level(level)
    for region in self.semantics.regions:
      self.parse_region(region)
    for object in self.semantics.objects:
      self.parse_object(object)

  def parse_level(self, level: SemanticLevel):
    for region in level.regions:
      self.parse_region(region)

  def parse_region(self, region: SemanticRegion):
    for object in region.objects:
      self.parse_object(object)
  
  def parse_object(self, object: SemanticObject):
    obj_id = int(object.id.split('_')[-1])
    self.object_id_to_object[obj_id] = object
  
  def print_semantic_meaning(self, top_n: int=3):
    """Print top N semantic meanings

    Args:
      top_n (int, optional): number of semantic meanings to print.
        Defaults to 3.
    """
    def _print_scene(scene, top_n=10):
      count = 0
      for region in scene.regions:
        for obj in region.objects:
          self.print_object_info(obj)
          count += 1
          if count >= top_n:
            return None
    _print_scene(self.semantics, top_n=top_n)

  def get_categorical_map(
    self,
    hm3d_obj_id_map: np.ndarray
  ) -> np.ndarray:
    """Create colorized mpcat40 categorical map

    Args:
      hm3d_obj_id_map (np.ndarray): expecting a 2D image (h, w),
        where each element is the object id in HM3D dataset. Usually
        this is the output from the semantic sensors.

    Returns:
        np.ndarray: colorized mpcat40 semantic segmentation image
          (h, w, 3).
    """
    # semantic_map: (h, w)
    return self.hm3d_mapping.get_mpcat40_category_map(hm3d_obj_id_map)

  def get(self, object_id: int) -> Optional[SemanticObject]:
    """Get SemanticObject by object ID

    Args:
      object_id (int): object's ID in HM3D dataset

    Returns:
      SemanticObject: the corresponding object's semantic meanings
        if the ID does not exist, then return None.
    """
    return self.object_id_to_object.get(object_id, None)

  def print_object_info(
    self,
    object: Union[int, SemanticObject],
    verbose=False
  ):
    """Print object's semantic meanings in HM3D dataset
    set `verbose`=True for its corresponding mpcat40 semantic
    meanings.

    Args:
      object (Union[int, SemanticObject]): object id or object.
      verbose (bool, optional): whether to print mpcat40 semantic
        meanings. Defaults to False.
    """
    # verbose: print mpcat40 info
    if not isinstance(object, SemanticObject):
      _object = self.get(object)
      assert _object is not None, f"object not found: {object}"
      object = _object
    print("==================")
    print(
      f"Object ID: {object.id}\n"
      f"  * category: {object.category.index()}/{object.category.name()}\n"
      f"  * center: {object.aabb.center}\n"
      f"  * dims: {object.aabb.sizes}"
    )
    if verbose:
      obj_id = int(object.id.split('_')[-1])
      mpcat40cat = self.hm3d_mapping.get_mpcat40cat(obj_id)
      print(f"  * mpcat40 category: {mpcat40cat.mpcat40index}/{mpcat40cat.mpcat40}")
      print(f"  * color: {mpcat40cat.hex}")


class HM3DMapping():
  def __init__(
    self,
    semantic_mapping: SemanticMapping,
    bgr: bool = True,
    default_category: str = 'unknown',
  ):
    """HM3DMapping helps mapping the HM3D dataset ID to
    mpcat40 category id, name, definitoins.

    Args:
      semantic_mapping (SemanticMapping): parsed semantic meaning.
      bgr (bool, optional): channel order of the image. BGR or RGB.
        Defaults to True.
      default_category (str, optional): default mpcat40 category for
        unlabeled/unknown objects in HM3D. Defaults to 'unknown'.
    """
    self.semantic_mapping = semantic_mapping
    self.hm3d_to_mpcat40_category_map = \
      utils.hm3d_to_mpcat40_category_map()
    self.mpcat40_color_map = utils.mpcat40_color_map(bgr=bgr)
    self.default_category = default_category
    # HM3D category name to MP3D raw category name
    self.manual_mapping = {
      'kitchen tools': 'kitchen utencil',
      'bedside cabinet': 'cabinet',
      'shoes on shelf': 'shoes'
    }
    # ---
    self.hm3d_obj_id_to_mpcat40_category_id = {}
    self.parse_hm3d_category_id()
    self._get_mpcat40_category_id_map = \
      np.vectorize(self.hm3d_obj_id_to_mpcat40_category_id.get)

  def parse_hm3d_category_id(self):
    """Create hm3d_obj_id_to_mpcat40_category_id mapping"""
    for obj_id, object in self.semantic_mapping.object_id_to_object.items():
      category_name = object.category.name().lower()
      # map category name by user defined mapping
      if category_name in self.manual_mapping.keys():
        category_name = self.manual_mapping[category_name]
      # if the category name does not exists in the mp3d categories
      # set it to `unknown`.
      if category_name not in self.hm3d_to_mpcat40_category_map:
        category_name = self.default_category
      # get mpcat40 category definitions
      mpcat40cat = self.hm3d_to_mpcat40_category_map[category_name]
      self.hm3d_obj_id_to_mpcat40_category_id[obj_id] = mpcat40cat.mpcat40index

  def get_mpcat40cat(
    self,
    hm3d_obj_id: int
  ) -> utils.MPCat40Category:
    """Get mpcat40 category definitions by giving HM3D object ID"""
    mpcat40cat_id = self.hm3d_obj_id_to_mpcat40_category_id[hm3d_obj_id]
    return utils.mpcat40categories[mpcat40cat_id]

  def get_mpcat40_category_id_map(
    self,
    hm3d_obj_id_map: np.ndarray
  ) -> np.ndarray:
    """Raw category map (index)"""
    return self._get_mpcat40_category_id_map(hm3d_obj_id_map)

  def get_mpcat40_category_map(
    self,
    hm3d_obj_id_map: np.ndarray
  ) -> np.ndarray:
    """Colorized category map (rgb or bgr)"""
    return self.mpcat40_color_map[
      self.get_mpcat40_category_id_map(hm3d_obj_id_map)
    ]
