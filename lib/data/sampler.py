# --- built in ---
# --- 3rd party ---
import numpy as np
import rlchemy
# --- my module ---
from lib.data.dataset import SequentialDataset
from lib.data.stream_producer import BaseStreamProducer
from lib.data.cache import Cache

__all__ = [
  "SequantialDatasetSampler"
]

class SequantialDatasetSampler():
  def __init__(
    self,
    dataset: SequentialDataset,
    stream_producer: BaseStreamProducer,
    num_workers: int,
    batch_size: int = None,
    time_first: bool = True,
    drop_remainder: bool = False,
  ):
    assert isinstance(dataset, SequentialDataset)
    self.dataset = dataset
    self.stream_producer = stream_producer
    self.num_workers = num_workers
    self.batch_size = batch_size
    self.time_first = time_first
    self._batch_dim = 0 if not self.time_first else 1
    if batch_size is not None:
      dataset = dataset.batch(
        batch_size,
        drop_remainder = drop_remainder,
        batch_dim = self._batch_dim
      )
    # create data iterator
    self._dataset_it = iter(self.dataset)
    # caches
    self._cached_inds = None
    self._caches = Cache(self.num_workers)
  
  def sample(self):
    ind, data = next(self._dataset_it)
    ind = rlchemy.utils.to_numpy(ind)
    self._cached_inds = ind
    if not self._caches.isnull:
      caches = self._caches[ind]
    else:
      caches = None
    return data, caches
  
  def cached(self, **kwargs):
    if self._cached_inds is None:
      raise RuntimeError("`_cached_inds` is empty")
    self._caches.update(indices=self._cached_inds, **kwargs)
  
  def close(self):
    del self._dataset_it
    del self._caches
    del self.dataset
    self.stream_producer.close()





