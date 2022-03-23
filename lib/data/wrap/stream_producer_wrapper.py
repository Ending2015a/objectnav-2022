# --- built in ---
import abc
import queue
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
# --- 3rd party ---
# --- my module ---
from lib.data import dataspec as lib_dataspec
from lib.data.stream_producer import BaseStreamProducer, StreamInfo

class StreamProducerWrapper():
  def __init__(self, stream_producer: BaseStreamProducer):
    self.stream_producer = stream_producer
    self._dataspec = None
    self.require(BaseStreamProducer)

  def __getattr__(self, name):
    return getattr(self.stream_producer, name)

  @property
  def dataspec(self):
    if self._dataspec is None:
      self._dataspec = self.gen_dataspec()
    return self._dataspec

  @property
  def unwrapped(self):
    return self.stream_producer.unwrapped

  def gen_dataspec(self):
    # read one stream
    data = self.read_stream(self.get_stream_info())
    sigs = lib_dataspec._generate_dataspec(data)
    return sigs

  def closed(self):
    return self.stream_producer.closed()
  
  def close(self):
    return self.stream_producer.close()

  def maybe_recharge(self):
    self.stream_producer.maybe_recharge()

  def get_stream_info(self, ind=None):
    return self.stream_producer.get_stream_info(ind)
  
  @abc.abstractmethod
  def read_stream(self, stream_info: StreamInfo):
    return self.stream_producer.read_stream(stream_info)
  
  def get_sample(self):
    return self.stream_producer.get_sample()
  
  def get_stream(self):
    try:
      stream_info = self.get_sample()
      data = self.read_stream(stream_info)
      stream_info.data = data
    except queue.Empty:
      return None
    return stream_info
  
  def iswrapped(self, _class):
    if not isinstance(self, _class):
      if (hasattr(self.stream_producer, 'iswrapped') and
          callable(self.stream_producer.iswrapped)):
        return self.stream_producer.iswrapped(_class)
      else:
        return isinstance(self.stream_producer, _class)
    return True

  def require(self, _class):
    if not self.iswrapped(_class):
      raise RuntimeError(f"Wrapper `{type(self).__name__}` requires "
        f"`{_class.__name__}`: {self}".format)
  
  def iterator(self, *args, **kwargs):
    while True:
      data = self.get_stream(*args, **kwargs)
      if data is None:
        break
      yield data

  def __call__(self, *args, iterate=False, **kwargs):
        if not iterate:
            return self.get_stream(*args, **kwargs)
        else:
            return self.iterator(*args, **kwargs)
    
  def __len__(self):
    return len(self.stream_producer)

  def __str__(self):
    return '<{}{}>'.format(type(self).__name__, self.stream_producer)

  def __repr__(self):
    return str(self)

  def __del__(self):
    if not self.closed():
      self.close()