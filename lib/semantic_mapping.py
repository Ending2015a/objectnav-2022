# --- built in ---
from typing import Dict
# --- 3rd party ---
import numpy as np
# --- my module ---
from lib import utils

class SemanticMapping():
  def __init__(
    self,
    env,
    bgr: bool = True,
    default_category: str = 'unknown'
  ):
    self.semantics = env.sim.semantic_annotations()
    self.object_id_to_object = {}
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

  def parse_level(self, level):
    for region in level.regions:
      self.parse_region(region)

  def parse_region(self, region):
    for object in region.objects:
      self.parse_object(object)
  
  def parse_object(self, object):
    obj_id = int(object.id.split('_')[-1])
    self.object_id_to_object[obj_id] = object
  
  def print_semantic_meaning(self, limit_output=3):
    def _print_scene(scene, limit_output=10):
      count = 0
      for region in scene.regions:
        for obj in region.objects:
          self.print_object_info(obj)
          count += 1
          if count >= limit_output:
            return None
    _print_scene(self.semantics, limit_output=limit_output)

  def get_categorical_map(self, hm3d_obj_id_map: np.ndarray):
    # semantic_map: (h, w)
    return self.hm3d_mapping.get_mpcat40_category_map(hm3d_obj_id_map)

  def get(self, object_id):
    return self.object_id_to_object.get(object_id)

  def print_object_info(self, object, verbose=False):
    # verbose: print mpcat40 info
    if isinstance(object, int):
      object = self.get(object)
    assert object is not None
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
    """This class helps to mapping the hm3d object id to
    mpcat40 cateogry id, name, colors
    """
    self.semantic_mapping = semantic_mapping
    self.hm3d_to_mpcat40_category_map = utils.hm3d_to_mpcat40_category_map()
    self.mpcat40_color_map = utils.mpcat40_color_map(bgr=bgr)
    self.default_category = default_category
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

  def get_mpcat40cat(self, hm3d_obj_id):
    mpcat40cat_id = self.hm3d_obj_id_to_mpcat40_category_id[hm3d_obj_id]
    return utils.mpcat40categories[mpcat40cat_id]

  def get_mpcat40_category_id_map(self, hm3d_obj_id_map):
    """Raw category map (index)"""
    return self._get_mpcat40_category_id_map(hm3d_obj_id_map)

  def get_mpcat40_category_map(self, hm3d_obj_id_map):
    """Colorized category map (rgb or bgr)"""
    return self.mpcat40_color_map[
      self.get_mpcat40_category_id_map(hm3d_obj_id_map)
    ]
