# --- bulit in ---
import os
import sys
import time
import logging
import math
from typing import Any, Tuple, Dict
import dataclasses
# --- 3rd party ---
import numpy as np
import torch
from torch import nn
import pandas as pd
# --- my module ---

# === Loading mp3d hm3d semantic labels ===
LIB_PATH = os.path.dirname(os.path.abspath(__file__))
# See https://github.com/niessner/Matterport/blob/master/metadata/mpcat40.tsv
MPCAT40_PATH = os.path.join(LIB_PATH, 'mpcat40.tsv')
CATEGORY_MAPPING_PATH = os.path.join(LIB_PATH, 'category_mapping.tsv')
MPCAT40_DF = pd.read_csv(MPCAT40_PATH, sep='\t')
CATEGORY_MAPPING_DF = pd.read_csv(CATEGORY_MAPPING_PATH, sep='\t')

def dataclass_factory(df, name):
  _class = dataclasses.make_dataclass(
    name,
    [
      (column_name, column_dtype)
      for column_name, column_dtype in
      zip(df.dtypes.index, df.dtypes.values)
    ]
  )
  cat_insts = []
  for index, row in df.iterrows():
    cat_inst = _class(**{
      column: row[column]
      for column in df.columns
    })
    cat_insts.append(cat_inst)
  return _class, cat_insts

MPCat40Category, mpcat40categories = \
  dataclass_factory(MPCAT40_DF, 'MPCat40Category')
HM3DCategory, hm3dcategories = \
  dataclass_factory(CATEGORY_MAPPING_DF, 'HM3DCategory')

hex2rgb = lambda hex: [int(hex[i:i+2], 16) for i in (1, 3, 5)]
hex2bgr = lambda hex: hex2rgb(hex)[::-1]


def hm3d_to_mpcat40_category_map() -> Dict[str, MPCat40Category]:
  # mapping by either category name
  # default: category name
  mpcat40_name_to_category = {
    mpcat40cat.mpcat40: mpcat40cat
    for mpcat40cat in mpcat40categories
  }
  hm3d_name_to_mpcat40 = {}
  for hm3dcat in hm3dcategories:
    hm3d_name_to_mpcat40[hm3dcat.category] = \
      mpcat40_name_to_category[hm3dcat.mpcat40]
    hm3d_name_to_mpcat40[hm3dcat.raw_category] = \
      mpcat40_name_to_category[hm3dcat.mpcat40]
  return hm3d_name_to_mpcat40

def mpcat40_color_map(bgr=False) -> np.ndarray:
  colors = [
    hex2rgb(mpcat40cat.hex) if not bgr else hex2bgr(mpcat40cat.hex)
    for mpcat40cat in mpcat40categories
  ]
  colors = np.asarray(colors, dtype=np.uint8)
  return colors

def hm3d_color_map(bgr=False) -> Dict[str, np.ndarray]:
  """Deprecated"""
  cat_map = hm3d_to_mpcat40_category_map()
  colors = {
    cat_name: hex2rgb(mpcat40cat.hex) if not bgr else
      hex2bgr(mpcat40cat.hex)
    for cat_name, mpcat40cat in cat_map.items()
  }
  return colors