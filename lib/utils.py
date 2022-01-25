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
import torch_scatter
import unstable_baselines as ub
# --- my module ---

# ========= Constant =========
ANGLE_EPS = 0.001
# ========= Linalg utils =========
def deg2rad(deg):
  """Convert angle from degree to radian"""
  return deg * (math.pi / 180.0)

def rad2deg(rad):
  """Convert angle from radian to degree"""
  return rad / (math.pi/ 180.0)

def polar2cart(r: float, th: float):
  """Convert polar to cartesian"""
  x = r * np.cos(th)
  y = r * np.sin(th)
  return x, y

def distance(a, b):
  """Calculate distance in Euclidean"""
  return np.linalg.norm(a-b, ord=2)

def get_camera_params(width, height, hfov, vfov=None):
  """Calculate camera intrinsic parameters from image size and fov

  Args:
      width (float): image width
      height (float): image height
      hfov (float): camera horizontal field of view
      vfov (float, optional): vertical field of view. Defaults to None.
  
  Returns:
      ub.utils.StateObject: a StateObject with the following keys
          cx: x-coordinates of center of the image
          cy: y-coordinates of center of the image
          fx: focal distance on x-axis
          fy: focal distance on y-axis
  """
  cx = width / 2.0
  cy = height / 2.0
  fx = cx / np.tan(hfov/2.0)
  fy = cy / np.tan(vfov/2.0) if vfov is not None else fx
  return ub.utils.StateObject(cx=cx, cy=cy, fx=fx, fy=fy)

# ========== PyTorch utils ===========
def validate_tensors(*tensors, same_device=None, same_dtype=None):
  """Validate tensors, convert all tensors to torch.tensor
  and ensure they are on the same device if `same_device` is specified
  or is set to True.

  Args:
      same_device (bool, torch.device, optional): device that all tensors to place on.
          Defaults to None.
      same_dtype (bool, torch.dtype, optional): data type. Defaults to None.

  Returns:
      list of validated tensors
  """
  if len(args) == 0:
    return
  # convert the first tensor
  first_tensor = ub.utils.to_tensor(tensors[0])
  # Get device
  if same_device is True:
    same_device = first_tensor.device
  else:
    same_device = None
  # Get dtype
  if same_dtype is True:
    same_dtype = first_tensor.dtype
  else:
    same_dtype = None
  cvt_tensors = []
  for tensor in tensors:
    cvt_tensors.append(
      ub.utils.to_tensor(tensor, device=device, dtype=dtype)
    )
  return *cvt_tensors

def rotate(points, axis, angle):
  """Rotating 3D `points` along the given `axis`
    with Rodringues' Rotation formula:
      R = I + S * sin(angle) + S^2 * (1-cos(angle))
    where
      R: rotation matrics
      S: [
              0,  -axis_x,  axis_y,
         axis_z,        0, -axis_x,
        -axis_y,   axis_z,       0
      ]
      S^2: matmul(S, S)
  
    Note that the coordinate system follows the rule
      X: right
      Y: up
      Z: forward

  Args:
      points (torch.tensor): points to rotate, in shape (b, ..., 3)
      axis (torch.tensor): axis the points rotated along, in shape (b, 3)
      angle (torch.tensor): rotated angle in radian with shape (b,) or (b, 1)
  
  Returns:
      torch.tensor: rotated points
  """
  points = ub.utils.to_tensor(points, dtype=torch.float32)
  axis = ub.utils.to_tensor(axis, dtype=torch.float32)
  angle = ub.utils.to_tensor(angle, dtype=torch.float32)
  device = points.device
  batch = points.shape[0]
  # Shape axis and angle to batch shape
  axis = axis.view(-1, 3)
  axis = torch.broadcast_to(axis, (batch, 3))
  angle = angle.view(-1, 3)
  angle = torch.broadcast_to(angle, (batch, 3))
  # Create batch rotation matrices from axis
  ax = axis / torch.linalg.norm(axis, dim=-1, keepdim=True) # (b, 3)
  ax_x = ax[..., 0] # (b,)
  ax_y = ax[..., 1]
  ax_z = ax[..., 2]
  zeros = torch.zeros((batch,), dtype=torch.float32, device=device)
  S_flat = torch.stack((
    zeros, -ax_z, ax_y,
    ax_z, zeros, -ax_x,
    -ax_y, ax_x, zeros
  ), dim=-1) # flat rotation matrices (b, 9)
  S = S_flat.view(-1, 3, 3)
  S2 = torch.einsum('bij,bjk->bik', S, S) # matmul(S, S)
  S2_flat = S2.view(-1, 9)
  eye_flat = torch.eye(3).view(-1, 9) # flat eye matrices
  # Clamp angle if it is near to 0.0
  angle = torch.where(torch.abs(angle) > ANGLE_EPS, angle, 0.)
  # Create rotation matrices
  R_flat = eye_flat + torch.sin(angle) * S_flat + (1 - torch.cos(angle)) * S2_flat
  R = R_flat.view(R, (-1, 3, 3)) # (b, 3, 3)
  # apply rotation
  points = torch.einsum('bji,b...j->b...i', R, points)
  return points

def translate(points, offsets):
  """Translating 3D points (move along XYZ)

    Note that the coordinate system follows the rule
      X: right
      Y: up
      Z: forward

  Args:
      points (torch.tensor): points to translate, (b, ..., 3)
      offsets (torch.tensor): XYZ offsets (b, 3)
  """
  points = ub.utils.to_tensor(points, dtype=torch.float32)
  offsets = ub.utils.to_tensor(offsets, dtype=torch.float32)
  batch = points.shape[0]
  offsets = offsets.view(-1, 1, 3)
  offsets = torch.broadcast_to(offsets, (batch, 1, 3))
  # apply translation
  points = torch.reshape(
    points.view(batch, -1, 3) + offsets,
    points.shape
  )
  return points

def ravel_index(index, shape, keepdim=False):
  """Ravel multi-dimensional indices to 1D index
  similar to np.ravel_multi_index

  For example:
  ```python
  indices = [[3, 2, 3], [0, 2, 1]]
  shape = (6, 5, 4)
  print(ravel_index(indices, shape))
  ```
  will output:
  ```
  tensor([71,  9])
  # 71 = 3 * (5*4) + 2 * (4) + 3
  # 9 = 0 * (5*4) + 2 * (4) + 1
  ```

  Args:
      index (torch.tensor): indices in reversed order dn, ..., d1,
          with shape (..., n)
      shape (tuple): shape of each dimension dn, ..., d1
      keepdim (bool, optional): keep dimensions. Defaults to False.

  Returns:
      torch.tensor: Raveled indices in shape (...,), if `keepdim` is False,
          otherwise in shape (..., 1).
  """
  index = ub.utils.to_tensor(index, dtype=torch.int64)
  shape = ub.utils.to_tensor((1,) + shape[::-1], dtype=torch.int64) # [1, d1, ..., dn]
  shape = torch.cumprod(shape, dim=0)[:-1].flip(0) # [d1*...*dn-1, ..., d1*d2, d1, 1]
  index = (index * shape).sum(dim=-1, keepdim=keepdim) # (..., 1) or (...,)
  return index

def scatter_max(
  canvas: torch.tensor,
  indices: torch.tensor,
  values: torch.tensor,
  masks: torch.tensor = None,
  fill_value: float = None,
  _validate_args: bool = True
):
  """Scattering values over an `n`-dimensional canvas

  In the case of projecting values to an image-type canvas (`n`=2), i.e.
  projecting height values to a top-down height map, the shape of the canvas is
  (..., d2, d1) or we say (..., h, w), where `...` is the batch dimensions.
  `values`, in this case, is the height values and has the shape (..., N) for a
  point cloud containing `N` points. `indices` is the coordinates of the top-
  down map corresponds to each point that being projected to and has the shape
  (..., N, 2). Note that the last dimension of `indices` stores in a reversed
  order, i.e. [d2, d1] or [h, w]. For `n`-dimensional canvas (..., dn, ..., d1),
  it stores [dn, ..., d1].

  Args:
      canvas (torch.tensor): canvas in shape (..., dn, ..., d1)
      indices (torch.tensor): (dn, ..., d1) coordinates in shape (..., N, n).
      values (torch.tensor): values to scatter in shape (..., N).
      masks (torch.tensor, optional): boolean masks. True for valid values,
          while False for invalid values. shape (..., N). Defaults to None.
      fill_value (float, optional): default values to fill in with. Defaults
          to None.
  
  Returns:
      torch.tensor: scattered canvas, (..., dn, ..., d1)
      torch.tensor: mask, (..., dn, ..., d1)
      torch.tensor: indices of scattered points, (..., dn, ..., d1)
  """
  if _validate_args:
    # Create default masks
    if masks is None:
      masks = torch.ones(values.shape)
    (canvas, indices, values, masks) = validate_tensors(
      canvas, indices, values, masks, same_device=True
    )
    # Ensure dtypes
    canvas = canvas.to(dtype=torch.float32)
    indices = indices.to(dtype=torch.int64)
    values = values.to(dtype=torch.float32)
    masks = masks.to(dtype=torch.bool)
  assert masks is not None
  # Get dimensions
  n = indices.shape[-1]
  N = values.shape[-1]
  assert len(canvas.shape) > n, \
    f"The rank of `canvas` must be greater than {n}, got {len(canvas.shape)}"
  dn_d1 = canvas.shape[-n:] # (dn, ..., d1)
  batch_dims = canvas.shape[:-n] # (...,)
  # Mark the out-of-bound points as invalid
  valid_areas = [masks]
  for i in reversed(range(n)):
    di = indices[..., i] # (..., N)
    valid_areas.extend((di < dn_d1[i], di >= 0))
  masks = torch.stack(valid_areas, dim=0).all(dim=0) # (..., N)
  # Set dummy indices for invalid points (0, ..., 0, -1)
  indices[..., :][~masks] = 0
  indices[..., -1][~masks] = -1
  # Flatten all things to 1D
  flat_canvas = canvas.view(*batch_dims, -1) # (..., d1*...*dn)
  flat_indices = ravel_index(indices, dn_d1) # convert nd indices to 1d indices (..., N)
  flat_masks = masks # (..., N)
  flat_values = values # (..., N)
  # Create dummy channel to store invalid values
  dummy_channel = torch.zeros_like(flat_canvas[..., 0:1]) # (..., 1)
  dummy_shift = 1
  # Shifting dummy index from (0, ..., 0, -1) to (0, ..., 0, 0)
  flat_canvas = torch.cat((dummy_channel, flat_canvas), dim=-1) # (..., 1 + d1*...*dn)
  flat_indices = flat_indices + dummy_shift
  # Initialize canvas with -np.inf if `fill_value` is provided
  if fill_value is not None:
    flat_canvas.fill_(fill_value)
  _, flat_indices = torch_scatter.scatter_max(
    flat_values, flat_indices, dim=-1, out=flat_canvas
  )
  # Slice out dummy channel
  flat_canvas = flat_canvas[..., 1:] # (..., d1*...*dn)
  canvas = flat_canvas.view(canvas.shape) # (..., dn, ..., d1)
  flat_indices = flat_indices[..., 1:] # (..., d1*...*dn)
  indices = flat_indices.view(canvas.shape) % N - 1 # (..., dn, ..., d1)
  masks = ~(indices == -1) # (..., dn, ..., d1)
  return canvas, masks, indices

def smooth_image(image, smooth_iter=2, smooth_kernel=3, smooth_op='max'):
  """Smoothing image tensor

  Args:
      image (torch.tensor): 2/3/4D image tensor
      smooth_iter (int, optional): number of iterations to smooth. Defaults to 2.
      smooth_kernel (int, optional): size of smoothing kernel. Defaults to 3.
      smooth_op (str, optional): type of smoothing operations, must be one of
          ['max', 'avg']. Defaults to 'max'.
  
  Returns:
      torch.tensor: smoothed image tensor
  """
  op = None
  if smooth_op = 'max':
    op = nn.functional.max_pool2d
  elif smooth_op = 'avg':
    op = nn.functional.avg_pool2d
  else:
    raise NotImplementedError(
      f"Unknown op `{smooth_op}` must be one of ['max', 'avg']")
  # Convert to 4D image
  image = ub.utils.to_tensor(image)
  dtype = image.dtype
  ndims = len(image.shape)
  image = to_4D_image(image)
  image = image.to(dtype=torch.float32)
  padding = smooth_kernel//2
  for _ in range(smooth_iter):
    image = op(image, smooth_kernel, 1, padding)
  # Convert back to the original rank
  image = from_4D_image(image, ndims)
  return image

def to_4D_image(image):
  """Convert `image` to 4D tensors (b, c, h, w)

  Args:
      image (torch.tensor): 2/3/4D image tensor
          2D: (h, w)
          3D: (c, h, w)
          4D: (b, c, h, w)
  
  Returns:
      torch.tensor: 4D image tensor
  """
  ndims = len(image.shape)
  assert ndims in [2, 3, 4], \
    f"only supports 2/3/4D images while {ndims}-D are given."
  if ndims == 2:
    return image[None, None, :, :]
  elif ndims == 3:
    return image[None, :, :, :]
  else:
    return image

def from_4D_image(image, ndims):
  """Convert `image` to `ndims`-D tensors

  Args:
      image (torch.tensor): 4D image tensors in shape (b, c, h, w).
      ndims (int): the original rank of the tensor.

  Returns:
      torch.tensor: `ndims`-D tensors
  """
  _ndims = len(image.shape)
  assert _ndims == 4, f"`image` must be a 4D tensor, while {_ndims}-D are given."
  if ndims == 2:
    return image[0, 0, :, :]
  elif ndims == 3:
    return image[0, :, :, :]
  else:
    return image