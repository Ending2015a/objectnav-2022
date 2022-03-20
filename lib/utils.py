# --- bulit in ---
import os
import sys
import time
import logging
import math
from typing import Any
# --- 3rd party ---
import numpy as np
import torch
from torch import nn
import pandas as pd
# --- my module ---

# === initialize data ===
LIB_PATH = os.path.abspath(__file__)
# See https://github.com/niessner/Matterport/blob/master/metadata/mpcat40.tsv
TSV_PATH = os.path.join(os.path.dirname(LIB_PATH), 'mpcat40.tsv')
MP3D_DATA = pd.read_csv(TSV_PATH, sep = '\t')

hex2rgb = lambda hex: [int(hex[i:i+2], 16) for i in (1, 3, 5)]
hex2bgr = lambda hex: hex2rgb(hex)[::-1]

def mp3d_color_map(bgr=False):
  colors = [
    hex2rgb(hex_color) if not bgr else hex2bgr(hex_color)
    for hex_color in MP3D_DATA['hex']
  ]
  colors = np.asarray(colors, dtype=np.uint8)
  return colors