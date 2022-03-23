# --- built in ---
import collections
from typing import (
  Any,
  Optional,
  List,
  Callable
)
# --- 3rd party ---
import torch
import rlchemy
# --- my module ---

__all__ = [
  'Cache',
  'NestedCaches'
]

class NestedCaches():
  def __init__(
    self,
    buffer_size: int,
    batch_dim: int = 0
  ):
    """This class is used to cache the trajectory-wise RNN states
    when training.
    The caches are saved as torch.Tensor in the following format
    {
      'obs': [obs_A, obs_B, obs_C, obs_D],
      'act': [act_A, act_B, act_C, act_D]
    }
    where A, B, C, D indicates the different indices of data, e.g.
    different trajectories.

    Args:
      buffer_size (int): usually the maximum number of trajectories
        this buffer can cache. Please set this to `num_workers`
        of stream dataset.
      batch_dim (int, optional): dimension to stack the sampled batch
        data. Defaults to 0.
    """
    self.buffer_size: int = buffer_size
    self.batch_dim: int = batch_dim
    self._cached_inds: Optional[List[int]] = None
    self.data: Optional[Any] = None
  
  def update(
    self,
    data: Any,
    indices: Optional[List[int]] = None
  ):
    """Update cache contents"""
    if indices is None:
      indices = self._cached_inds
    self._set_data(data, indices=indices)

  def melloc(self, data: Any):
    """Create cache spaces
    The spaces are initialized as
    {
      'obs': [torch.zeros, torch.zeros, torch.zeros, torch.zeros],
      'act': [torch.zeros, torch.zeros, torch.zeros, torch.zeros]
    }
    """
    def _melloc_op(v):
      return [
        torch.zeros(
          v.shape[1:],
          dtype=v.dtype,
          device=v.device
        )
      ] * self.buffer_size
    self.data = rlchemy.utils.map_nested(data, op=_melloc_op)

  @property
  def isnull(self) -> bool:
    """Return True if the cache spaces are not created"""
    return self.data is None

  def __len__(self) -> int:
    """Cache size"""
    return self.buffer_size

  def __getitem__(self, key: List[int]) -> Any:
    """Get caches

    Args:
      key (List[int]): indices.

    Returns:
      Any: sliced data.
    """
    return self._get_data(key)

  def __setitem__(self, key: List[int], value: Any):
    """Set caches

    Args:
      key (List[int]): indices.
      value (Any): sliced data.
    """
    self._set_data(value, indices=key)

  def _get_data(self, indices: List[int]):
    if self.isnull:
      raise RuntimeError("Buffer space not created, please `add` data, "
        " or calling `melloc` first.")
    # get data
    def _slice_op(v):
      vs = [v[ind] for ind in indices]
      return torch.stack(vs, dim=self.batch_dim)
    return rlchemy.utils.map_nested(self.data, op=_slice_op)

  def _set_data(self, data: Any, indices: List[int]):
    if self.isnull:
      self.melloc(data)
    # assign data to buffer
    def _assign_op(data_tuple, inds):
      vs, data = data_tuple
      for i, ind in enumerate(inds):
        v = vs[i]
        if torch.is_tensor(v):
          v = v.detach()
        else:
          v = rlchemy.utils.to_tensor_like(v, data[ind])
        data[ind] = v
    rlchemy.utils.map_nested_tuple(
      (data, self.data),
      op = _assign_op,
      inds = indices
    )

class Caches(NestedCaches):
  def keys(self):
    return (
      collections.abc.KeysView([])
      if self.isnull else self.data.keys()
    )

  def update(self, indices=None, **kwargs):
    super().update(kwargs, indices=indices)
