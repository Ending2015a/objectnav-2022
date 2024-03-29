# --- built in ---
import queue
import threading
from typing import (
  Any,
  Optional,
  Callable,
  List,
  Union,
  Tuple,
  Iterable,
  Iterator
)
from distutils.version import StrictVersion as Version
# --- 3rd party ---
import numpy as np
import rlchemy
# --- my module ---
from kemono.data.stream_producer import BaseStreamProducer, StreamInfo
from kemono.data.wrap.stream_producer_wrapper import StreamProducerWrapper

__all__ = [
  'CombinedStreamProducer',
  'ZeroPadChunkStreamWrapper',
  'ResetMaskStreamWrapper'
]

# custom stream producer wrappers

class CombinedStreamProducer(StreamProducerWrapper):
  def __init__(
    self,
    stream_producers: List[BaseStreamProducer],
    shuffle: bool = False
  ):
    super().__init__(stream_producers[0])
    self.stream_producers = stream_producers
    for producer in stream_producers:
      producer.require(BaseStreamProducer)
    self.shuffle = shuffle
    self._lock = threading.Lock()
    self.buffer = None
    self.recharge()
  
  def recharge(self):
    indices = []
    for idx, producer in enumerate(self.stream_producers):
      producer.maybe_recharge()
      indices.extend([idx] * producer.buffer.qsize())
    # generate permutations
    num_items = len(indices)
    if self.shuffle:
      np.random.shuffle(indices)
    self.buffer = queue.Queue(maxsize=num_items)
    try:
      for ind in indices:
        self.buffer.put(ind, block=False)
    except queue.Full:
      pass

  def maybe_recharge(self):
    with self._lock:
      if self.buffer.empty():
        self.recharge()

  def get_stream_info(
    self,
    ind: Optional[Tuple[int, StreamInfo]]=None
  ) -> StreamInfo:
    if ind is None:
      info = self.stream_producers[0].get_stream_info()
      producer_id = 0
    else:
      producer_id, info = ind
    return StreamInfo(info=info, producer_id=producer_id)

  def read_stream(self, stream_info: StreamInfo) -> Any:
    producer = self.stream_producers[stream_info.producer_id]
    data = producer.read_stream(stream_info.info)
    return data

  def get_sample(self) -> StreamInfo:
    with self._lock:
      producer_id = self.buffer.get(block=False)
      stream_info = self.stream_producers[producer_id].get_sample()
    return self.get_stream_info((producer_id, stream_info))
  
  def __str__(self) -> str:
    wrapped = ', '.join([
      str(producer) for producer in self.stream_producers
    ])
    return f"<{type(self).__name__}({wrapped})>"
  
  def __len__(self) -> int:
    return self.buffer.qsize()

  @property
  def unwrapped(self) -> BaseStreamProducer:
    # first stream_producer
    return self.stream_producer.unwrapped


class ZeroPadChunkStreamWrapper(StreamProducerWrapper):
  def __init__(
    self,
    stream_producer: BaseStreamProducer,
    chunk_size: Optional[int] = 32,
    mask_key: str = 'mask'
  ):
    super().__init__(stream_producer)
    self.chunk_size = chunk_size
    self.mask_key = mask_key

  def read_stream(self, stream_info: StreamInfo) -> Any:
    data = super().read_stream(stream_info)
    # get the length of data from the first field
    data_len = len(next(iter(rlchemy.utils.iter_nested(data))))
    mask = np.ones((data_len,), dtype=bool)
    data[self.mask_key] = mask
    if self.chunk_size is not None:
      pad_size = self.chunk_size - (data_len % self.chunk_size)
      data = rlchemy.utils.map_nested(data, op=self._pad_op, pad_size=pad_size)
    return data

  @staticmethod
  def _pad_op(data, pad_size):
    pad_size = [(0, pad_size)]
    for _ in range(1, len(data.shape)):
      pad_size.append((0, 0))
    return np.pad(data, pad_size)


class ResetMaskStreamWrapper(StreamProducerWrapper):
  def __init__(
    self,
    stream_producer: BaseStreamProducer,
    reset_key: str = 'reset'
  ):
    super().__init__(stream_producer)
    self.reset_key = reset_key
  
  def read_stream(self, stream_info: StreamInfo) -> Any:
    data = super().read_stream(stream_info)
    # get the length of data from the first field
    data_len = len(next(iter(rlchemy.utils.iter_nested(data))))
    reset = np.zeros((data_len,), dtype=bool)
    reset[0] = True
    data[self.reset_key] = reset
    return data
