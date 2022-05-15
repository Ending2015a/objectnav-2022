# --- built in ---
import os
import sys
import math
import time
import logging
# --- 3rd party ---
import habitat
import numpy as np
from torch import nn
from contextlib import contextmanager
# --- my module ---

def split_tiles(
  image: np.ndarray,
  tile_size: int,
  stride: int
):
  """Split image into small tiles

  Args:
    image (np.ndarray): image to split, in shape (h, w, c)
    tile_size (int): _description_
    stride (int): _description_
  
  Returns:
    List[np.ndarray]: splitted tiles
    List[Tuple[int, int]]: center coordinates of each tile [w, h]
  """
  height, width = image.shape
  nrows = int(math.ceil(height/stride))
  ncols = int(math.ceil(width/stride))
  h_pad = int((nrows-1) * stride + tile_size - height)
  w_pad = int((ncols-1) * stride + tile_size - width)
  tiles = []
  centers = []
  padded = np.pad(image, ((0, h_pad), (0, w_pad)))
  for h_idx in range(nrows):
    for w_idx in range(ncols):
      start_h = h_idx * stride
      start_w = w_idx * stride
      center_h = start_h + tile_size//2
      center_w = start_w + tile_size//2
      end_h = start_h + tile_size
      end_w = start_w + tile_size
      tile = padded[start_h:end_h, start_w:end_w]
      tiles.append(tile)
      centers.append((center_w, center_h))
  return tiles, centers

@contextmanager
def evaluate(model: nn.Module):
  """Temporary switch model to evaluation mode"""
  training = model.training
  try:
    model.eval()
    yield model
  finally:
    if training:
      model.train()