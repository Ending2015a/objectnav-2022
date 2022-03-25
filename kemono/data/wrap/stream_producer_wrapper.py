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
from kemono.data.dataspec import DataSpec, _generate_dataspec
from kemono.data.stream_producer import BaseStreamProducer, StreamInfo

class StreamProducerWrapper():
  def __init__(self, stream_producer: BaseStreamProducer):
    self.stream_producer = stream_producer
    self._dataspec = None
    self.require(BaseStreamProducer)

  def __getattr__(self, name: str) -> Any:
    return getattr(self.stream_producer, name)

  @property
  def dataspec(self) -> DataSpec:
    if self._dataspec is None:
      self._dataspec = self.gen_dataspec()
    return self._dataspec

  @property
  def unwrapped(self) -> BaseStreamProducer:
    return self.stream_producer.unwrapped

  def gen_dataspec(self) -> DataSpec:
    # read one stream
    data = self.read_stream(self.get_stream_info())
    sigs = _generate_dataspec(data)
    return sigs

  def closed(self) -> bool:
    return self.stream_producer.closed()
  
  def close(self) -> None:
    return self.stream_producer.close()

  def maybe_recharge(self):
    self.stream_producer.maybe_recharge()

  def get_stream_info(self, ind: Any=None) -> StreamInfo:
    return self.stream_producer.get_stream_info(ind)
  
  @abc.abstractmethod
  def read_stream(self, stream_info: StreamInfo) -> Any:
    return self.stream_producer.read_stream(stream_info)
  
  def get_sample(self) -> StreamInfo:
    """Sample one stream info from the buffer"""
    return self.stream_producer.get_sample()
  
  def get_stream(self) -> StreamInfo:
    """Get stream info with loaded stream data"""
    try:
      stream_info = self.get_sample()
      data = self.read_stream(stream_info)
      stream_info.data = data
    except queue.Empty:
      return None
    return stream_info
  
  def iswrapped(self, _class: type) -> bool:
    if not isinstance(self, _class):
      if (hasattr(self.stream_producer, 'iswrapped') and
          callable(self.stream_producer.iswrapped)):
        return self.stream_producer.iswrapped(_class)
      else:
        return isinstance(self.stream_producer, _class)
    return True

  def require(self, _class: type):
    if not self.iswrapped(_class):
      raise RuntimeError(f"Wrapper `{type(self).__name__}` requires "
        f"`{_class.__name__}`: {self}".format)
  
  def iterator(self, *args, **kwargs) -> Iterator:
    while True:
      data = self.get_stream(*args, **kwargs)
      if data is None:
        break
      yield data

  def __call__(
    self,
    *args,
    iterate: bool = False,
    **kwargs
  ) -> Union[StreamInfo, Iterator]:
    if not iterate:
        return self.get_stream(*args, **kwargs)
    else:
        return self.iterator(*args, **kwargs)
    
  def __len__(self) -> int:
    return len(self.stream_producer)

  def __str__(self) -> str:
    return f"<{type(self).__name__}{self.stream_producer}>"

  def __repr__(self) -> str:
    return str(self)

  def __del__(self):
    if not self.closed():
      self.close()