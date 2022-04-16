# --- built in ---
from typing import Any, Tuple
# --- 3rd party ---
import torch
import numpy as np
import rlchemy
# --- my module ---
from kemono.data.dataset import StreamDataset
from kemono.data.stream_producer import BaseStreamProducer
from kemono.data.cache import Caches

__all__ = [
  "StreamDatasetSampler"
]

class StreamDatasetSampler():
  def __init__(
    self,
    dataset: StreamDataset,
    stream_producer: BaseStreamProducer,
    num_workers: int,
    batch_size: int = None,
    time_first: bool = True,
    drop_remainder: bool = False,
    return_indices: bool = False
  ):
    assert isinstance(dataset, StreamDataset)
    self.dataset = dataset
    self.stream_producer = stream_producer
    self.num_workers = num_workers
    self.batch_size = batch_size
    self.time_first = time_first
    self.return_indices = return_indices
    if batch_size is not None:
      self.dataset = dataset.batch(
        batch_size,
        drop_remainder = drop_remainder
      )
    # create data iterator
    self._dataset_it = iter(self.dataset)
    # caches
    self._cached_inds = None
    self._caches = Caches(self.num_workers)

  @staticmethod
  def _swap_time_batch(data):
    def _swap_op(data):
      if len(data.shape) >= 2:
        if torch.is_tensor(data):
          return torch.swapaxes(data, 0, 1)
        else:
          return np.swapaxes(data, 0, 1)
      return data
    return rlchemy.utils.map_nested(data, _swap_op)

  def sample(self) -> Tuple[Any, Any]:
    ind, data = next(self._dataset_it)
    ind = rlchemy.utils.to_numpy(ind)
    self._cached_inds = ind
    
    if self.time_first:
      # in default the first and second dimensions of the data
      # are `batch` and `time`. If `time_first` is True, we
      # swap the first two dimensions
      data = self._swap_time_batch(data)
    if not self._caches.isnull:
      caches = self._caches[ind]
    else:
      caches = None
    if self.return_indices:
      return data, caches, ind
    else:
      return data, caches
  
  def cache(self, **kwargs):
    if self._cached_inds is None:
      raise RuntimeError("`_cached_inds` is empty")
    self._caches.update(indices=self._cached_inds, **kwargs)
  
  def close(self):
    del self._dataset_it
    del self._caches
    del self.dataset
    self.stream_producer.close()





