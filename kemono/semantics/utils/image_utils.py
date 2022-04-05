# --- built in ---
from re import I
from typing import (
  Any,
  Tuple,
  Union
)
# --- 3rd party ---
import torch
from torch import nn
import numpy as np
# --- my module ---

__all__ = [
  'to_4D_tensor',
  'from_4D_tensor',
  'ImageNormalize',
  'resize_image'
]

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

class ImageNormalize(nn.Module):
  def __init__(
    self,
    mean: Union[np.ndarray, torch.Tensor],
    std: Union[np.ndarray, torch.Tensor],
  ):
    super().__init__()
    mean = to_4D_tensor(torch.as_tensor(mean))
    std = to_4D_tensor(torch.as_tensor(std))
    self.register_buffer('mean', mean, persistent=False)
    self.register_buffer('std', std, persistent=False)

  def forward(self, x):
    return (x - self.mean) / self.std

def resize_image(
  image: Union[np.ndarray, torch.Tensor],
  size: Tuple[int, int],
  mode: str = 'nearest',
  **kwargs
) -> torch.Tensor:
  """Resize image tensor

  Args:
      image (Union[np.ndarray, torch.Tensor]): expecting
        (h, w), (c, h, w), (b, c, h, w)
      size (Tuple[int, int]): target size in (h, w)
      mode (str, optional): resize mode. Defaults to 'nearest'.
  """
  size = torch.Size(size)
  t = torch.as_tensor(image)
  orig_shape = t.shape
  orig_dtype = t.dtype
  orig_ndims = len(orig_shape)
  _t = to_4D_tensor(t) # (b, c, h, w)
  # do nothing if the tensor shape is already matched to the target size
  if _t.shape[-len(size):] == size:
    return t
  t = _t
  t = t.to(dtype=torch.float32)
  t = nn.functional.interpolate(t, size=size, mode=mode, **kwargs)
  t = from_4D_tensor(t, orig_ndims)
  return t.to(dtype=orig_dtype)
