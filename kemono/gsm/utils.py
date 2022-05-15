# --- built in ---
import os
import sys
import math
import time
import logging
from typing import Union, Tuple
# --- 3rd party ---
import habitat
import numpy as np
import torch
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


def to_4D_tensor(t: torch.Tensor) -> torch.Tensor:
  """Convert `t` to 4D tensors (b, c, h, w)
  Args:
    t (torch.Tensor): 0/1/2/3/4D tensor
      0D: ()
      1D: (c,)
      2D: (h, w)
      3D: (c, h, w)
      4D: (b, c, h, w)
  
  Returns:
    torch.Tensor: 4D image tensor
  """
  ndims = len(t.shape)
  if ndims == 0:
    # () -> (b, c, h, w)
    return torch.broadcast_to(t, (1, 1, 1, 1))
  elif ndims == 1:
    # (c,) -> (b, c, h, w)
    return t[None, :, None, None]
  elif ndims == 2:
    # (h, w) -> (b, c, h, w)
    return t[None, None, :, :] # b, c
  elif ndims == 3:
    # (c, h, w) -> (b, c, h, w)
    return t[None, :, :, :] # b
  else:
    return t

def from_4D_tensor(t: torch.Tensor, ndims: int) -> torch.Tensor:
  """Convert `t` to `ndims`-D tensors
  Args:
    t (torch.Tensor): 4D image tensors in shape (b, c, h, w).
    ndims (int): the original rank of the image.
  Returns:
    torch.Tensor: `ndims`-D tensors
  """
  _ndims = len(t.shape)
  assert _ndims == 4, f"`t` must be a 4D tensor, but {_ndims}-D are given."
  if ndims == 0:
    return t[0, 0, 0, 0]
  elif ndims == 1:
    return t[0, :, 0, 0]
  elif ndims == 2:
    return t[0, 0, :, :] # -b, -c
  elif ndims == 3:
    return t[0, :, :, :] # -b
  else:
    return t

def resize_image(
  image: Union[np.ndarray, torch.Tensor],
  size: Tuple[int, int],
  mode: str = 'nearest',
  **kwargs
) -> Union[np.ndarray, torch.Tensor]:
  size = torch.Size(size)
  is_tensor = torch.is_tensor(image)
  t = torch.as_tensor(image)
  orig_shape = t.shape
  orig_dtype = t.dtype
  orig_ndims = len(orig_shape)
  # convert to 4D tensor (b, c, h, w)
  t = to_4D_tensor(t)
  # if the size is already the desired size
  if t.shape[-len(size):] == size:
    return image
  t = t.to(dtype=torch.float32)
  t = nn.functional.interpolate(t, size=size, mode=mode, **kwargs)
  t = from_4D_tensor(t, orig_ndims)
  t = t.to(dtype=orig_dtype)
  if not is_tensor:
    return t.cpu().numpy()
  return t
