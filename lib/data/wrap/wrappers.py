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
# --- my module ---
from lib.data import dataspec as lib_dataspec
from lib.data.stream_producer import BaseStreamProducer, StreamInfo
from lib.data.wrap.stream_producer_wrapper import StreamProducerWrapper

__all__ = [
    'CombinedStreamProducer'
]

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
  
  def recharge(self):
    indices = []
    for idx, producer in enumerate(self.stream_producers):
      self.producer.maybe_recharge()
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

  def read_stream(self, stream_info: StreamInfo):
    producer = self.stream_producer[stream_info.producer_id]
    data = producer.read_stream(stream_info.info)
    return data

  def get_sample(self):
    with self._lock:
      producer_id = self.buffer.get(block=False)
      stream_info = self.stream_producers[producer_id].get_sample()
    return self.get_stream_info((producer_id, stream_info))
  
  def __str__(self):
    wrapped = ', '.join([
      str(producer) for producer in self.stream_producers
    ])
    return '<{}({})>'.format(type(self).__name__, wrapped)
  
  def __len__(self):
    return self.buffer.qsize()

  @property
  def unwrapped(self):
    # first stream_producer
    return self.stream_producer.unwrapped