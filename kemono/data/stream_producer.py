# --- built in ---
import os
import abc
import glob
import copy
import queue
import logging
import threading
from dataclasses import dataclass
from typing import (
  Any,
  Optional,
  List,
  Dict,
  Union,
  Tuple,
  Iterable,
  Iterator,
)
# --- 3rd party ---
import numpy as np
import rlchemy
# --- my module ---
from kemono.data import dataspec as km_dataspec

__all__ = [
  'BaseStreamProducer',
  'BaseDynamicStreamProducer',
  'RlchemyStreamProducer',
  'RlchemyDynamicStreamProducer'
]

TRAIN_DATA_ROOT = 'TRAIN_DATA_ROOT'
IGNORE_ERRORS = os.environ.get('KEMONO_SP_IGNORE_ERRORS', True)

def get_data_root(
  data_root: Optional[str] = None,
  default: Optional[str] = None
) -> Optional[str]:
  return data_root or os.environ.get(TRAIN_DATA_ROOT, default)

def _convert_to_string_list(
  items: Union[str, Iterable[str], Any]
) -> List[str]:
  """Convert items to a list of strings

  Args:
    items (Union[str, List[str], Any]): expecintg a string, or a list/iterable
      of string

  Returns:
    List[str]: a list of string
  """
  if isinstance(items, str):
    items = [items]
  else:
    # if items is an iterable, iter it and convert to str
    items = [str(item) for item in items]
  return items

@dataclass
class StreamInfo:
  index: Optional[int] = None
  path: Optional[str] = None
  data: Optional[Any] = None
  # for nested Stream Producer
  info: Optional["StreamInfo"] = None
  producer_id: Optional[int] = None

class BaseStreamProducer(metaclass=abc.ABCMeta):
  """StreamProducer responsible for producing random stream data
  loaded from dick
  {data_root}/{globa_path}
  {data_root}/{source_id}/{stream_id}/{stream_data}.npy
  """
  def __init__(
    self,
    data_root: Optional[str] = None,
    glob_paths: Optional[List[str]] = None,
    stream_paths: Optional[List[str]] = None,
    shuffle: bool = False,
    callbacks: List[Any] = []
  ):
    """The base class of stream producer

    Args:
        data_root (Optional[str], optional): _description_. Defaults to None.
        glob_paths (Optional[List[str]], optional): _description_. Defaults to None.
        stream_paths (Optional[List[str]], optional): _description_. Defaults to None.
        shuffle (bool, optional): _description_. Defaults to False.
        callbacks (List[Any], optional): _description_. Defaults to [].
    """
    self.shuffle = shuffle
    # initialize state variables
    self._stream_paths = None
    self._dataspec = None
    self._lock = threading.Lock()
    self._is_closed = False
    self._on_before_recharge_callbacks = []
    self._on_after_recharge_callbacks = []
    self._on_load_stream_paths_callbacks = []
    self.buffer = None
    callbacks = callbacks or []
    for callback in callbacks:
      self.register_callback(callback)
    self.load_stream_paths(data_root, glob_paths, stream_paths)
    self.recharge()

  @property
  def dataspec(self):
    if self._dataspec is None:
      self._dataspec = self.gen_dataspec()
    return self._dataspec

  @property
  def unwrapped(self) -> "BaseStreamProducer":
    return self

  @property
  def stream_paths(self) -> List[str]:
    return self._stream_paths

  @stream_paths.setter
  def stream_paths(self, values: List[str]):
    self._stream_paths = values

  def load_stream_paths(
    self,
    data_root: Optional[str] = None,
    glob_paths: Optional[List[str]] = None,
    stream_paths: Optional[List[str]] = None
  ):
    if stream_paths is None:
      stream_paths = []
    else:

      try:
        stream_paths = _convert_to_string_list(stream_paths)
      except:
        raise ValueError("Unknown stream_paths, stream_paths must be a `str` "
          f"or a list of `str`, got {type(stream_paths)}")
        
    if glob_paths is not None:
      # get data root, raise exception if not specified
      data_root = get_data_root(data_root)
      if data_root is None:
        raise ValueError("Please specify `data_root` or set the environment "
          f"variable `TRAIN_DATA_ROOT`, got {data_root}")

      try:
        glob_paths = _convert_to_string_list(glob_paths)
      except:
        raise ValueError("Unknown source_ids, source_ids must be a `str` or a "
          f"list of `str`, got {type(stream_paths)}")
    
      for glob_path in glob_paths:
        glob_path = os.path.join(data_root, glob_path)

        # append streams to stream_paths
        stream_names = glob.glob(glob_path, recursive=True)
        if len(stream_names) == 0:
          print('WARNING:Stream Producer: no stream were found in:')
          print(f'WARNING:Stream Producer: {glob_path}')
        stream_names = sorted(stream_names)
        stream_paths.extend(stream_names)

    if len(stream_paths) == 0:
      print('WARNING:Stream Producer: no stream were found')

    self._stream_paths = stream_paths
    self.on_load_stream_paths()
  
  def gen_dataspec(self):
    # read one stream
    data = self.read_stream(self.get_stream_info())
    sigs = km_dataspec.generate_dataspec(data)
    return sigs

  def register_callback(self, callback: Any):
    if callback is None:
      return
    if hasattr(callback, "on_before_recharge"):
      self._on_before_recharge_callbacks.append(callback)
    if hasattr(callback, "on_after_recharge"):
      self._on_after_recharge_callbacks.append(callback)
    if hasattr(callback, "on_load_stream_paths"):
      self._on_load_stream_paths_callbacks.append(callback)

  def recharge(self):
    if (self._stream_paths is None
        or len(self._stream_paths) == 0):
      print('WARNING:Stream Producer: no stream were found')
    self.on_before_recharge()
    num_items = len(self._stream_paths)
    if self.shuffle:
      indices = np.random.permutation(num_items)
    else:
      indices = np.arange(num_items)
    # create queue
    self.buffer = queue.Queue(maxsize=num_items)
    try:
      for ind in indices:
        self.buffer.put(ind, block=False)
    except queue.Full:
      pass
    self.on_after_recharge()
  
  def maybe_recharge(self):
    # if buffer is empty, recharge it.
    with self._lock:
      if self.buffer.empty():
        self.recharge()
  
  def get_stream_info(self, ind: Optional[Any]=None):
    if ind is None:
      ind = 0 # dummy info
    return StreamInfo(index=ind, path=self._stream_paths[ind])

  @abc.abstractmethod
  def read_stream(self, stream_info: StreamInfo) -> Tuple[Any, ...]:
    """Customize the data loading method

    Args:
      stream_info (StreamInfo): stream info contains the index of stream
        and the path to the stream data.

    Returns:
      Tuple[Any, ...]: stream data

    Raises:
        NotImplementedError:
    """
    stream_path = stream_info.path
    raise NotImplementedError
    # load data
    # return obs, act, ...
  
  def get_sample(self):
    """Sample one stream (info)
    Raise:
      queue.Empty: if self.buffer is empty
    """
    with self._lock:
      ind = self.buffer.get(block=False)
    return self.get_stream_info(ind)

  def _safe_read_stream(
    self,
    stream_info: StreamInfo
  ) -> Tuple[bool, StreamInfo]:
    """The first bool denotes wherther the data is correctly loaded"""
    try:
      data = self.read_stream(stream_info)
      return True, data
    except:
      if IGNORE_ERRORS:
        return False, None
      raise

  def get_stream(self) -> StreamInfo:
    """Sample one stream info and read the stream data

    Returns:
      StreamInfo: loaded stream info. By accessing the
        `stream.data` to get the loaded data
    """
    try:
      res = False
      while not res:
        stream_info = self.get_sample()
        res, data = self._safe_read_stream(stream_info)
      stream_info.data = data
    except queue.Empty:
      return None
    return stream_info

  def iterator(self, *args, **kwargs) -> Iterator:
    """Iterate over all stream data"""
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
  ) -> Union[Any, Iterator]:
    """Sample one stream or iterate over all streams

    Args:
      iterate (bool, optional): return an iterator. Defaults to False.
    
    Returns:
      Union[Any, Iterator]: one sample or an iterator
    """
    if not iterate:
      return self.get_stream(*args, **kwargs)
    else:
      return self.iterator(*args, **kwargs)

  def require(self, _class: type):
    return isinstance(self, _class)

  def __len__(self) -> int:
    return self.buffer.qsize()
  
  def __str__(self) -> str:
    return f"<{type(self).__name__}({len(self._stream_paths)})>"
    
  def __repr__(self) -> str:
      return str(self)

  def __del__(self):
    if not self.closed():
      self.close()

  def closed(self) -> bool:
    return self._is_closed
  
  def close(self):
    self._is_closed = True

  # === events ===
  def on_before_recharge(self):
    for callback in self._on_before_recharge_callbacks:
      callback(self)

  def on_after_recharge(self):
    for callback in self._on_after_recharge_callbacks:
      callback(self)
  
  def on_load_stream_paths(self):
    for callback in self._on_load_stream_paths_callbacks:
      callback(self)

class BaseDynamicStreamProducer(BaseStreamProducer):
  def __init__(
    self,
    data_root: Optional[str] = None,
    glob_paths: Optional[List[str]] = None,
    stream_paths: Optional[List[str]] = None,
    shuffle: bool = False,
    callbacks: List[Any] = []
  ):
    """BaseDynamicStreamProducer checks for updates in each source path
    on every buffer recharge.

    Args:
        data_root (str, optional): _description_. Defaults to None.
        source_ids (List[str], optional): _description_. Defaults to None.
        glob_paths (List[str], optional): _description_. Defaults to None.
        shuffle (bool, optional): _description_. Defaults to False.
        filter_fn (Callable, optional): _description_. Defaults to None.
    """
    self._load_kwargs = {
      'data_root': copy.deepcopy(data_root),
      'glob_paths': copy.deepcopy(glob_paths),
      'stream_paths': copy.deepcopy(stream_paths)
    }

    super().__init__(
      data_root = data_root,
      glob_paths = glob_paths,
      stream_paths = stream_paths,
      shuffle = shuffle,
      callbacks = callbacks
    )

  def on_before_recharge(self):
    # reload stream paths
    self.load_stream_paths(**self._load_kwargs)
    # callbacks
    super().on_before_recharge()

class RlchemyStreamProducer(BaseStreamProducer):
  def read_stream(self, stream_info: StreamInfo) -> Dict[str, Any]:
    traj = rlchemy.envs.load_trajectory(stream_info.path)
    traj.pop('info', None)
    return traj

class RlchemyDynamicStreamProducer(BaseDynamicStreamProducer):
  def read_stream(self, stream_info: StreamInfo) -> Dict[str, Any]:
    traj = rlchemy.envs.load_trajectory(stream_info.path)
    traj.pop('info', None)
    return traj
