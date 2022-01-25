# --- built in ---
import os
import abc
import sys
import time
import logging
from typing import Tuple
# --- 3rd party ---
import numpy as np
import torch
from torch import nn
import einops
import unstable_baselines as ub
# --- my module ---
import lib
from lib import utils

class DepthProjection():
  """DepthProjection provides utility functions for projecting depth maps
  onto various spaces."""
  @classmethod
  def project(
    cls,
    depth_map: torch.tensor,
    cam_pose: torch.tensor,
    width_offset: torch.tensor,
    height_offset: torch.tensor,
    cam_pitch: torch.tensor,
    cam_height: torch.tensor,
    map_res: float,
    map_width: int,
    map_height: int,
    focal_x: float,
    focal_y: float,
    center_x: float,
    center_y: float,
    trunc_depth_min: float,
    trunc_depth_max: float,
    clip_border: int,
    to_global: bool,
    get_point_cloud: bool = False,
    _validate_args: bool = True
  ):
    """Project depth maps to top-down height maps

    Args:
        depth_map (torch.tensor): UNNORMALIZED depth map, which means it has the
            range of values [min_depth, max_depth]. The rank must be at least 4D
            (b, c, h, w), (b, ..., h, w) for higher dimensions. torch.float32.
        cam_pose (torch.tensor): camera pose [x, z, yaw] with shape (b, 3), where
            yaw in rad. torch.float32
        width_offset (torch.tensor): batch offset along map width. torch.int64
        height_offset (torch.tensor): batch offset along map height. torch.int64
        cam_pitch (torch.tensor): batch camera pitch. torch.float32
        cam_height (torch.tensor): batch camera height. torch.float32
        map_res (float): map resolution, meter per cell.
        map_width (int): map width, pixel.
        map_height (int): map height, pixel.
        focal_x (float): focal distance along x-axis.
        focal_y (float): focal distance along y-axis.
        center_x (float): center position.
        center_y (float): center position.
        trunc_depth_min (float): minimal depth to truncate. Set to -1 to disable.
        trunc_depth_max (float): maximal depth to truncate. Set to -1 to disable.
        clip_border (int): number of pixels to clip the depth map's left/right/top/
            down borders.
        to_global (bool): convert to global space by `cam_pose`.
        get_point_cloud (bool, optional): return transformed point cloud.
            Defaults to False.

    Returns:
        torch.tensor: top-down height maps. the value is set to -inf for invalid
            regions. One can use `np.isinf` or `mask` to check if it's valid.
        torch.tensor: mask, True for valid regions. torch.bool
        torch.tensor, optional: point cloud in global coordinates, if
            `get_point_cloud` is set to True. Shape (..., 3), where `...` refers
            to the shape of input depth map. torch.float32
    """
    # Validate tensors:
    if _validate_args:
      # Convert to tensors and ensure they are on the same device
      (depth_map, cam_pose, width_offset, height_offset,
        cam_pitch, cam_height) = utils.validate_tensors(
          depth_map, cam_pose, width_offset, height_offset,
          cam_pitch, cam_height,
          same_device = True
      )
      # Ensure tensor shape at least 4D (b, ..., h, w)
      depth_map = utils.to_4D_image(depth_map) # (b, c, h, w)
      # Note that we don't broadcast shapes here since we do
      # not sure the actual size of the batch dimensions, so
      # we remain such broadcasting parts into each sub-
      # method.
      cam_pose = cam_pose.view(-1, 3) # (b, 3)
      cam_pitch = cam_pitch.view(-1) # (b,)
      cam_height = cam_height.view(-1) # (b,)
      width_offset = width_offset.view(-1) # (b,)
      height_offset = height_offset.view(-1) # (b,)
      # Ensure dtypes
      depth_map = depth_map.to(dtype=torch.float32)
      cam_pose = cam_pose.to(dtype=torch.float32)
      width_offset = width_offset.to(dtype=torch.float32)
      height_offset = height_offset.to(dtype=torch.float32)
      cam_pitch = can_pitch.to(dtype=torch.float32)
      cam_height = cam_height.to(dtype=torch.float32)
    # TODO: Denormalize depth map [0, 1] -> [min, max]
    #depth_map = depth_map * (max_depth - min_depth) + min_depth
    # Convert depth map to point cloud
    point_cloud, valid_area = cls.depth_to_point_cloud(
      depth_map = depth_map,
      focal_x = focal_x,
      focal_y = focal_y,
      center_x = center_x,
      center_y = center_y,
      trunc_depth_max = trunc_depth_max,
      trunc_depth_min = trunc_depth_min,
      _validate_args = False
    ) # (b, ..., h, w, 3)
    # Truncate depth map border pixels
    if (clip_border is not None) and (clip_border > 0):
      valid_area = cls._clip_borders(
        valid_area = valid_area,
        clip_border = clip_border
      )
    # Transform space from camera space to local space
    point_cloud = cls.cam_to_local_space(
      point_cloud = point_cloud,
      cam_pitch = cam_pitch,
      cam_height = cam_height,
      _validate_args = False
    )
    # Transform space from local space to global space
    if to_global:
      point_cloud = cls.local_to_global_space(
        point_cloud = point_cloud,
        cam_pose = cam_pose,
        _validate_args = False
      )
    # Flatten point cloud
    # (b, ..., h, w, 3) -> (b, ..., h*w, 3)
    flat_point_cloud = torch.flatten(point_cloud, -3, -2)
    # (b, ..., h, w) -> (b, ..., h*w)
    flat_valid_area = torch.flatten(valid_area, -2, -1)
    # Project point cloud to height map
    height_map, valid_area = cls._orth_project(
      flat_point_cloud = flat_point_cloud,
      flat_valid_area = flat_valid_area,
      width_offset = width_offset,
      height_offset = height_offset,
      map_res = map_res,
      map_width = map_width,
      map_height = map_height,
      smooth_iter = smooth_iter,
      smooth_window = smooth_window
    )
    if get_point_cloud:
      return height_map, valid_area, point_cloud
    else:
      return height_map, valid_area  
  
  @classmethod
  def depth_to_point_cloud(
    cls,
    depth_map: torch.tensor,
    focal_x: float,
    focal_y: float,
    center_x: float,
    center_y: float,
    trunc_depth_min: float,
    trunc_depth_max: float,
    _validate_args: bool = True
  ):
    """Generates point cloud from `depth_map`
    
        X: positive to Right
        Y: positive to Up
        Z: positive to Forward (Depth)
    
    Args:
        depth_map (torch.tensor): unnormalized depth map, 2/3/4D image with c=1, shape
            (h, w), (c, h, w) or (b, c, h, w). torch.float32
        focal_x (float): focal distance along x-axis.
        focal_y (float): focal distance along y-axis.
        center_x (float): center position.
        center_y (float): center position.
        trunc_depth_min (float): minimal depth to truncate. Set to -1 to disable truncation.
        trunc_depth_max (float): maximal depth to truncate. Set to -1 to disable truncation.
      
    Returns:
        torch.tensor: point cloud in shape (..., 3).
        torch.tensor: mask in shape (..., h, w) indicating the valid area.
    """
    if _validate_args:
      # Convert to tensors and ensure they are on the same device
      depth_map = utils.validate_tensors(depth_map, same_device=True)
      # Ensure tensor shape at least 4D (b, ..., h, w)
      depth_map = utils.to_4D_image(depth_map) # (b, c, h, w)
      # Ensure dtypes
      depth_map = depth_map.to(dtype=torch.float32)
    device = depth_map.device
    ndims = len(depth_map.shape) # (..., h, w)
    h = depth_map.shape[-2]
    w = depth_map.shape[-1]
    # Generate x, y coordinates
    x = torch.arange(w, dtype=torch.float32, device=device)
    # (0,0) at upper left corner of the depth map
    y = torch.arange(h-1, -1, -1, dtype=torch.float32, device=device)
    # Expand dims to match depth map
    x = x.view((1,)*(ndims-2) + (1, -1)) # (..., 1, w)
    y = y.view((1,)*(ndims-2) + (-1, 1)) # (..., h, 1)
    # Cast dtypes
    cx = ub.utils.to_tensor_like(center_x, depth_map)
    cy = ub.utils.to_tensor_like(center_y, depth_map)
    fx = ub.utils.to_tensor_like(focal_x, depth_map)
    fy = ub.utils.to_tensor_like(focal_y, depth_map)
    # Convert to point cloud
    z = depth_map # (..., h, w)
    x = (x-cx)/fx * depth_map # (..., h, w)
    y = (y-cy)/fy * depth_map # (..., h, w)
    point_cloud = torch.stack((x, y, z), dim=-1) # (..., h, w, 3)
    valid_area = torch.ones_like(z, dtype=torch.bool) # (..., h, w)
    # Truncate invalid values
    if trunc_depth_max is not None:
      valid_area = torch.logical_and(z <= trunc_depth_max, valid_area)
    if trunc_depth_min is not None:
      valid_area = torch.logical_and(z >= trunc_depth_min, valid_area)
    return point_cloud, valid_area
  
  @classmethod
  def _clip_borders(
    cls,
    valid_area: torch.tensor,
    clip_border: int
  ):
    """Clip depth map left/right/top/down borders
    by setting the corresponding `valid_area` to False

    Args:
        valid_area (torch.tensor): mask in shape (..., h, w). torch.bool
        clip_border (int): number of border pixels to clip.

    Returns:
        torch.tensor: clipped valid area in shape (..., h, w). torch.bool
    """
    device = valid_area.device
    *batch_dims, h, w = valid_area.shape # (..., h, w)
    batch_ndims = len(batch_dims)
    # Compute number of pixels to clip
    clip_size = clip_border * 2
    clipped_shape = (*batch_dims, int(h-clip_size), int(w-clip_size))
    clipped_area = torch.ones(clipped_shape, dtype=torch.bool, device=device)
    # Create paddings for l/r/t/d sides
    pad_size = [clip_border] * 4
    padded_area = nn.functional.pad(
      clipped_area,
      pad_size,
      mode='constant',
      False
    )
    return torch.logical_and(valid_area, padded_area)
  
  @classmethod
  def cam_to_local_space(
    cls,
    point_cloud: torch.tensor,
    cam_pitch: torch.tensor,
    cam_height: torch.tensor,
    _validate_args: bool = True
  ):
    """Transform point cloud from camera space to local space

    Args:
        point_cloud (torch.tensor): point cloud in shape (b, ..., 3), torch.float32
        cam_pitch (torch.tensor): camera pitch in rad. (b,) or float.
        cam_height (torch.tensor): camera height (b,) or float.

    Returns:
        torch.tensor: transformed point cloud in shape (b, ..., 3), torch.float32
    """
    if _validate_args:
      (point_cloud, cam_pitch, cam_height) = utils.validate_tensors(
        point_cloud, cam_pitch, cam_height, same_device=True
      )
      # Ensure tensor shapes
      orig_shape = point_cloud.shape
      orig_ndims = len(orig_shape)
      if orig_ndims < 2:
        # pad batch dim, the rank of point cloud should be at least 2
        point_cloud = point_cloud.view(-1, 3)
      batch = point_cloud.shape[0]
      point_cloud = point_cloud.view(batch, -1, 3) # (b, ..., 3)
      cam_pitch = torch.broadcast_to(cam_pitch, (batch,))
      cam_height = torch.broadcast_to(cam_height, (batch,))
      # Ensure dtypes
      point_cloud = point_cloud.to(dtype=torch.float32)
      cam_pitch = cam_pitch.to(dtype=torch.float32)
      cam_height = cam_height.to(dtype=torch.float32)
    # Rotate `cam_pitch` angle along x-axis
    point_cloud = utils.rotate(point_cloud, [1., 0., 0.], cam_pitch)
    zeros = torch.zeros_like(cam_height)
    x = zeros
    y = cam_height
    z = zeros # (b,)
    # Apply translations
    pos = torch.stack((x, y, z), dim=-1) # (b, 3)
    point_cloud = utils.translate(point_cloud, pos)
    if _validate_args:
      point_cloud = point_cloud.view(orig_shape)
    return point_cloud

  @classmethod
  def local_to_cam_space(
    cls,
    point_cloud: torch.tensor,
    cam_pitch: torch.tensor,
    cam_height: torch.tensor,
    _validate_args: bool = True
  ):
    """Transform point cloud from local space to camera space

    Args:
        point_cloud (torch.tensor): point cloud in shape (b, ..., 3), torch.float32
        cam_pitch (torch.tensor): camera pitch in rad. (b,) or float.
        cam_height (torch.tensor): camera height (b,) or float.

    Returns:
        torch.tensor: transformed point cloud in shape (b, ..., 3), torch.float32
    """
    if _validate_args:
      (point_cloud, cam_pitch, cam_height) = utils.validate_tensors(
        point_cloud, cam_pitch, cam_height, same_device=True
      )
      # Ensure tensor shapes
      orig_shape = point_cloud.shape
      orig_ndims = len(orig_shape)
      if orig_ndims < 2:
        # pad batch dim, the rank of point cloud should be at least 2
        point_cloud = point_cloud.view(-1, 3)
      batch = point_cloud.shape[0]
      point_cloud = point_cloud.view(batch, -1, 3) # (b, ..., 3)
      cam_pitch = torch.broadcast_to(cam_pitch, (batch,))
      cam_height = torch.broadcast_to(cam_height, (batch,))
      # Ensure tensor dtypes
      point_cloud = point_cloud.to(dtype=torch.float32)
      cam_pitch = cam_pitch.to(dtype=torch.float32)
      cam_height = cam_height.to(dtype=torch.float32)
    zeros = torch.zeros_like(cam_height)
    x = zeros
    y = -cam_height
    z = zeros
    # Apply translations
    pos = torch.stack((x, y, z), dim=-1)
    point_cloud = utils.translate(point_cloud, pos)
    point_cloud = utils.rotate(point_cloud, [1., 0., 0.], -cam_pitch)
    if _validate_args:
      point_cloud = point_cloud.view(orig_shape)
    return point_cloud

  @classmethod
  def local_to_global_space(
    cls,
    point_cloud: torch.tensor,
    cam_pose: torch.tensor,
    _validate_args: bool = True
  ):
    """Transform point cloud from local space to global space

    Args:
        point_cloud (torch.tensor): point cloud in shape (b, ..., 3), torch.float32
        cam_pose (torch.tensor): camera pose [x, z, yaw] in shape (b, 3). yaw's
            unit is rad.
    
    Returns:
        torch.tensor: transformed point cloud (b, ..., 3), torch.float32
    """
    if _validate_args:
      (point_cloud, cam_pose) = utils.validate_tensors(
        point_cloud, cam_pose, same_device=True
      )
      # Ensure tensor shapes
      orig_shape = point_cloud.shape
      orig_ndims = len(orig_shape)
      if orig_ndims < 2:
        # pad batch dim, the rank of point cloud should be at least 2
        point_cloud = point_cloud.view(-1, 3)
      batch = point_cloud.shape[0]
      point_cloud = point_cloud.view(batch, -1, 3) # (b, ..., 3)
      cam_pose = torch.broadcast_to(cam_pose, (batch, 3))
      # Ensure tensor dtypes
      point_cloud = point_cloud.to(dtype=torch.float32)
      cam_pose = cam_pose.to(dtype=torch.float32)
    # Rotate yaw along y-axis
    yaw = cam_pose[..., 2] # (b,)
    point_cloud = utils.rotate(point_cloud, [0., 1., 0.], yaw)
    zeros = torch.zeros_like(yaw)
    x = cam_pose[..., 0]
    y = zeros
    z = cam_pose[..., 1] # (b,)
    # Apply translations
    pos = torch.stack((x, y, z), dim=-1) # (b, 3)
    point_cloud = utils.translate(point_cloud, pos)
    if _validate_args:
      point_cloud = point_cloud.view(orig_shape)
    return point_cloud

  @classmethod
  def global_to_local_space(
    cls,
    point_cloud: torch.tensor,
    cam_pose: torch.tensor,
    _validate_args: bool = True
  ):
    """Transform point cloud from global space to local space

    Args:
        point_cloud (torch.tensor): point cloud in shape (b, ..., 3), torch.float32
        cam_pose (torch.tensor): camera pose [x, z, yaw] in shape (b, 3). yaw's
            unit is rad.

    Returns:
        torch.tensor: transformed point cloud (b, ..., 3), torch.float32
    """
    if _validate_args:
      (point_cloud, cam_pose) = utils.validate_tensors(
        point_cloud, cam_pose, same_device=True
      )
      # Ensure tensor shapes
      orig_shape = point_cloud.shape
      orig_ndims = len(orig_shape)
      if orig_ndims < 2:
        # pad batch dim, the rank of point cloud should be at least 2
        point_cloud = point_cloud.view(-1, 3)
      batch = point_cloud.shape[0]
      point_cloud = point_cloud.view(batch, -1, 3) # (b, ..., 3)
      cam_pose = torch.broadcast_to(cam_pose, (batch, 3))
      # Ensure tensor dtypes
      point_cloud = point_cloud.to(dtype=torch.float32)
      cam_pose = cam_pose.to(dtype=torch.float32)
    yaw = cam_pose[..., 2] # (b,)
    zeros = torch.zeros_like(yaw)
    x = cam_pose[..., 0]
    y = zeros
    z = cam_pose[..., 1] # (b,)
    # Apply translations
    pos = torch.stack((x, y, z), dim=-1) # (b, 3)
    point_cloud = utils.translate(point_cloud, -pos)
    # Rotate `yaw` angle along y-aixs
    point_cloud = utils.rotate(point_cloud, [0., 1., 0.], -yaw)
    if _validate_args:
      point_cloud = point_cloud.view(orig_shape)
    return point_cloud

  @classmethod
  def point_cloud_to_map_space(
    cls,
    point_cloud: torch.tensor,
    width_offset: torch.tensor,
    height_offset: torch.tensor,
    map_res: float,
    map_width: int,
    map_height: int,
    _validate_args: bool = True
  ):
    """Project point cloud to map space
    The difference with `_orth_project` is that this method does not project
    point cloud onto an image (orthogonal projection). The returned tensors
    are still a point cloud, but in map coordinates.

    Args:
        point_cloud (torch.tensor): point cloud in shape (b, ..., 3), torch.float32
        width_offset (torch.tensor): batch offset along map width. torch.int64
        height_offset (torch.tensor): batch offset along map height. torch.int64
        map_res (float): map resolution, meter per cell.
        map_width (int): map width, pixel.
        map_height (int): map height, pixel.

    Returns:
        torch.tensor: transformed point cloud (b, ..., 3), torch.float32
    """
    if _validate_args:
      (point_cloud, width_offset, height_offset) = utils.validate_tensors(
        point_cloud, width_offset, height_offset, same_device=True
      )
      # Ensure tensor shapes
      orig_shape = point_cloud.shape
      orig_ndims = len(orig_shape)
      if orig_ndims < 2:
        # pad batch dim, the rank of point cloud should be at least 2
        point_cloud = point_cloud.view(-1, 3)
      batch = point_cloud.shape[0]
      point_cloud = point_cloud.view(batch, -1, 3) # (b, ..., 3)
      width_offset = torch.broadcast_to(width_offset, (batch,))
      height_offset = torch.broadcast_to(height_offset, (batch,))
      # Ensure tensor dtypes
      point_cloud = point_cloud.to(dtype=torch.float32)
      width_offset = width_offset.to(dtype=torch.float32)
      height_offset = height_offset.to(dtype=torch.float32)
    # Quantize indices
    x_bin, z_bin = cls._bin_quantize(
      point_cloud = point_cloud,
      width_offset = width_offset,
      height_offset = height_offset,
      map_res = map_res,
      map_width = map_width,
      map_height = map_height
    )
    x = x_bin.to(dtype=torch.float32)
    y = point_cloud[..., 1] # y
    z = z_bin.to(dtype=torch.float32) # (b, ..., 3)
    indices = torch.stack((x, y, z), axis=-1)
    if _validate_args:
      indices = indices.view(orig_shape)
    return indices

  @classmethod
  def map_space_to_point_cloud(
    cls,
    indices: torch.tensor,
    width_offset: torch.tensor,
    height_offset: torch.tensor,
    map_res: float,
    map_width: int,
    map_height: int,
    _validate_args: bool = True
  ):
    """Project back from map space to point cloud

    Args:
        indices (torch.tensor): indices in shape (b, ..., 3), torch.float32
        width_offset (torch.tensor): batch offset along map width. torch.int64
        height_offset (torch.tensor): batch offset along map height. torch.int64
        map_res (float): map resolution, meter per cell.
        map_width (int): map width, pixel.
        map_height (int): map height, pixel.

    Returns:
        torch.tensor: transformed point cloud (b, ..., 3), torch.float32
    """
    if _validate_args:
      (indices, width_offset, height_offset) = utils.validate_tensors(
        indices, width_offset, height_offset, same_device=True
      )
      # Ensure tensor shapes
      orig_shape = indices.shape
      orig_ndims = len(orig_shape)
      if orig_ndims < 2:
        # pad batch dim, the rank of point cloud should be at least 2
        indices = indices.view(-1, 3)
      batch = indices.shape[0]
      indices = indices.view(batch, -1, 3) # (b, ..., 3)
      width_offset = torch.broadcast_to(width_offset, (batch,))
      height_offset = torch.broadcast_to(height_offset, (batch,))
      # Ensure tensor dtypes
      indices = indices.to(dtype=torch.float32)
      width_offset = width_offset.to(dtype=torch.float32)
      height_offset = height_offset.to(dtype=torch.float32)
    # Reverse quantiation
    x, z = cls._rev_bin_quantize(
      indices = indices,
      width_offset = width_offset,
      height_offset = height_offset,
      map_res = map_res,
      map_width = map_width,
      map_height = map_height
    )
    y = indices[..., 1]
    point_cloud = torch.stack((x, y, z), dim=-1) # (b, ..., 3)
    if _validate_args:
      point_cloud = point_cloud.view(orig_shape)
    return point_cloud

  @classmethod
  def _bin_quantize(
    cls,
    points: torch.tensor,
    width_offset: torch.tensor,
    height_offset: torch.tensor,
    map_res: float,
    map_width: int,
    map_height: int
  ):
    """Quantize x, z coordinates of each point in a point cloud
    to the specified bins.

    Args:
        points (torch.tensor): points in shape (b, ..., 3), torch.float32
        width_offset (torch.tensor): bin offset on width orientation (usually
            horizontal), shape (b,), torch.float32.
        height_offset (torch.tensor): bin offsets on height orientation (usually
            vertical), shape (b,), torch.float32.
        map_res (float): map resolutions.
        map_width (int): map width.
        map_height (int): map height.

    Returns:
        torch.tensor: indices of bins on x-axis for each point, torch.int64
        torch.tensor: indices of bins on z-axis for each point, torch.int64
    """
    # Ensure tensor shape
    batch_dims = points.shape[:-1]
    batch_ndims = len(batch_dims)
    # (...,)
    width_offset = width_offset.view((-1,)+(1,)*(batch_ndims-1))
    height_offset = height_offset.view((-1,)+(1,)*(batch_ndims-1))
    width_offset = width_offset.to(dtpye=torch.float32)
    height_offset = height_offset.to(dtype=torch.float32)
    map_width = map_width.to(dtype=torch.float32)
    map_height = map_height.to(dtype=torch.float32)
    x = points[..., 0]
    y = points[..., 1]
    z = points[..., 2]
    # Z: (0, z) -> (far, near) + shift, X: (0, x) -> (left, right) + shift
    z_bin = torch.round( -z/map_res + (map_height-1) - height_offset).to(dtype=torch.int64)
    x_bin = torch.round(  x/map_res + width_offset).to(dtype=torch.int64)
    return x_bin, z_bin

  @classmethod
  def _rev_bin_quantize(
    cls,
    indices: torch.tensor,
    width_offset: torch.tensor,
    height_offset: torch.tensor,
    map_res: float,
    map_width: int,
    map_height: int
  ):
    """Revert `_bin_quantize` operation

    Args:
        indices (torch.tensor): (x, y, z) indices in map space with shape (b, ..., 3).
            torch.float32
        width_offset (torch.tensor): bin offset on width orientation (usually horizontal).
        height_offset (torch.tensor): bin offsets on height orientation (usually vertical).
        map_res (float): map resolutions.
        map_width (int): map width.
        map_height (int): map height.

    Returns:
        torch.tensor: x coordinate of each point.
        torch.tensor: z coordinate of each point.
    """
    # Ensure tensor shape
    batch_dims = indices.shape[:-1]
    batch_ndims = len(batch_dims)
    # (...,)
    width_offset = width_offset.view((-1,)+(1,)*(batch_ndims-1))
    height_offset = height_offset.view((-1,)+(1,)*(batch_ndims-1))
    width_offset = width_offset.to(dtpye=torch.float32)
    height_offset = height_offset.to(dtype=torch.float32)
    map_width = map_width.to(dtype=torch.float32)
    map_height = map_height.to(dtype=torch.float32)
    ind_x = indices[..., 0]
    ind_y = indices[..., 1]
    ind_z = indices[..., 2]
    z = -(ind_z - (map_height-1) + height_offset) * map_res
    x = (ind_x - width_offset) * map_res
    return x, z

  @classmethod
  def orth_project(
    cls,
    points: torch.tensor,
    values: torch.tensor,
    masks torch.tensor,
    width_offset: torch.tensor,
    height_offset: torch.tensor,
    map_res: float,
    map_width: int,
    map_height: int,
    fill_value: float = -np.inf,
    _validate_args: bool = True
  ):
    """Project values to top-down map.
    The shape of `points` is (b, ..., n, 3), where `n` is the number of points.
    `...` is arbitrary batch dimensions. The resulting shape is (b, ..., mh, mw,
    v), where `mh`, `mw` are `map_height` and `map_width`, respectively.

    Args:
        points (torch.tensor): flattened points, shape (b, ..., n, 3). torch.float32.
        values (torch.tensor): flattened values for each point. shape (b, ..., n).
            torch.float32.
        masks (torch.tensor): flattened mask for each point, True for valid points.
            shape (b, ..., n). torch.float32.
        width_offset (torch.tensor): image width offset, shape (b,). torch.float32.
        height_offset (torch.tensor): image height offset, shape (b,). torch.float32.
        map_res (float): map resolution unit: meter/cell.
        map_width (int): number of width pixels
        map_height (int): number of height pixels
        fill_value (float, optional): default values to fill in the top-down map.
            Defaults to -np.inf.
    
    Returns:
        torch.tensor: top-down map in shape (b, ..., mh, mw)
        torch.tensor: mask in shape (b, ..., mh, mw), False if the value is invalid,
            True otherwise.
    """
    if _validate_args:
      # Convert to tensors and ensure they are on the same device
      (points, values, masks, width_offset, height_offset
        ) = utils.validate_tensors(
          points, values, masks, width_offset, height_offset,
          same_device = True
      )
      # Ensure tensor shapes
      orig_shape = points.shape
      orig_ndims = len(orig_shape)
      if orig_ndims < 3:
        points = points.view(1, -1, 3) # at least 3 dims (b, n, 3)
      batch = points.shape[0]
      b___n = torch.broadcast_shapes(value.shape, masks.shape, points.shape[:-1])
      points = torch.broadcast_to(points, b___n+(3,)) # (b, ..., n, 3)
      values = torch.broadcast_to(values, b___n) # (b, ..., n)
      masks = torch.broadcast_to(masks, b___n) # (b, ..., n)
      width_offset = torch.broadcast_to(width_offset, (batch,))
      height_offset = torch.broadcast_to(height_offset, (batch,))
      # Ensure dtypes
      points = points.to(dtype=torch.float32)
      values = values.to(dtype=torch.float32)
      masks = masks.to(dtype=torch.bool)
      width_offset = width_offset.to(dtype=torch.float32)
      height_offset = height_offset.to(dtype=torch.float32)
    device = points.device
    valid_area = masks
    batch_dims = points.shape[:-2] # (b, ...)
    x_bin, z_bin = cls._bin_quantize(
      points = points,
      width_offset = width_offset,
      height_offset = height_offset,
      map_res = map_res,
      map_width = map_width,
      map_height = map_height
    )
    # (x, z) coordinates on 2D top-down map
    indices = torch.stack((z_bin, x_bin), dim=-1) # (b, ..., n, 2)
    # Filtering invalid area (b, ..., n)
    valid_area = torch.stack((
      x_bin >= 0, x_bin < map_width,
      z_bin >= 0, z_bin < map_height, valid_area
    ), dim=0).all(dim=0) # reduce all
    # Create canvas (b, ..., mh, mw)
    canvas_dims = (*batch_dims, map_height, map_width)
    canvas = torch.zeros(canvas_dims, device=device)
    # Perform orthogonal projection
    # canvas: (b, ..., mh, mw)
    # indices: (b, ..., n, 2)
    # values: (b, ..., n)
    # valid_area: (b, ..., n)
    # Ensure dtypes
    canvas = canvas.to(dtype=torch.float32)
    indices = indices.to(dtype=torch.int64)
    values = values.to(dtype=torch.float32)
    valid_area = valid_area.to(dtype=torch.bool)
    topdown_map, masks, _ = utils.scatter_max(
      canvas = canvas,
      indices = indices,
      values = values,
      masks = valid_area,
      fill_value = fill_value,
      _validate_args = False
    ) # (b, ..., mh, mw)
    return topdown_map, masks

  # @classmethod
  # def _orth_project(
  #   cls,
  #   flat_point_cloud: torch.tensor,
  #   flat_valid_area: torch.tensor,
  #   width_offset: torch.tensor,
  #   height_offset: torch.tensor,
  #   map_res: float,
  #   map_width: int,
  #   map_height: int,
  #   smooth_iter: int,
  #   smooth_window: int
  # ):
  #   """Project point cloud to top-down height map.
  #   The shape changes from (b, ..., n, 3) to (b, ..., mh, mw, 1), where `n` is an
  #   arbitrary number of points in the point cloud, `...` is the batch dimensions,
  #   `mh`, `mw` are `map_height` and `map_width`, respectively.

  #   Args:
  #       flat_point_cloud (torch.tensor): flattened point cloud, shape (b, ..., n, 3).
  #           torch.float32.
  #       flat_valid_area (torch.tensor): flattened mask, shape (b, ..., n).
  #           torch.float32
  #       width_offset (torch.tensor): image width offset, shape (b,). torch.float32.
  #       height_offset (torch.tensor): image height offset, shape (b,). torch.float32.
  #       map_res (float): map resolution unit: meter/cell.
  #       map_width (int): number of width pixels
  #       map_height (int): number of height pixels
  #       smooth_iter (int): number of iterations to smooth height maps.
  #       smooth_window (int): type of smoothing window.
    
  #   Returns:
  #       torch.tensor: height map in shape (b, ..., mh, mw, 1)
  #       torch.tensor: mask in shape (b, ..., mh, mw)
  #   """
  #   point_cloud = flat_point_cloud
  #   valid_area = flat_valid_area
  #   batch_dims = point_cloud.shape[:-2] # len((b, ...,))
  #   value_dim = 1 # value depth (used in scatter_max)
  #   x_bin, z_bin = cls._bin_quantize(
  #     point_cloud = point_cloud,
  #     width_offset = width_offset,
  #     height_offset = height_offset,
  #     map_res = map_res,
  #     map_width = map_width,
  #     map_height = map_height
  #   )
  #   # (x, z) coordinates on 2D height map
  #   indices = torch.stack((z_bin, x_bin), dim=-1) # (b, ..., n, 2)
  #   # Filtering invalid area (b, ..., n)
  #   valid_area = torch.stack((
  #     x_bin >= 0, x_bin < map_width,
  #     z_bin >= 0, z_bin < map_height, valid_area
  #   ), dim=0).all(dim=0) # reduce all
  #   # Create canvas
  #   canvas_dims = (*batch_dims, map_height, map_width, value_dim)
  #   canvas = torch.fill(canvas_dims, -np.inf) # (b, ..., mh, mw, 1)
  #   values = point_cloud[..., 1].unsqueeze(dim=-1) # (b, ..., n, 1)
  #   # Perform orthogonal projection
  #   # canvas: (b, ..., mh, mw, 1)
  #   # indices: (b, ..., n, 2)
  #   # values: (b, ..., n)
  #   # valid_area: (b, ..., n)
  #   height_map, _ = utils.scatter_max(
  #     canvas = canvas,
  #     indices = indices,
  #     values = values,
  #     mask = valid_area,
  #     fill_value = -np.inf
  #   ) # (b, ..., mh, mw, 1)
  #   height_map = height_map.squeeze(dim=-1) # (b, ..., mh, mw)
  #   # Smoothing height map
  #   height_map = utils.smooth_image(
  #     height_map,
  #     smooth_iter,
  #     smooth_window,
  #     smooth_op='max'
  #   )
  #   # update `valid_area`
  #   mask = ~torch.isinf(height_map) # (b, ..., mh, mw)
  #   return height_map, mask