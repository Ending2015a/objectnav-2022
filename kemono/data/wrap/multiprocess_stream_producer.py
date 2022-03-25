# --- built in ---
import gc
import os
import copy
import queue
import threading
import multiprocessing
from typing import (
  Any,
  Dict,
  Optional,
  Callable
)
# --- 3rd party ---
import cloudpickle
import numpy as np
# --- my module ---
from kemono.data.stream_producer import BaseStreamProducer, StreamInfo
from kemono.data.wrap.stream_producer_wrapper import StreamProducerWrapper

def debug_info(v):
  """print data infos"""
  return (f"{v.dtype}, {v.shape}, {np.max(v)}, {np.min(v)}, "
    f"{np.sum(np.isnan(v))}, {np.sum(np.isinf(v))}")

def debug_data(data):
  """Nested print data infos"""
  try:
    for k, vs in data.items():
      if isinstance(vs, tuple):
        print(f"  {k}:")
        for idx, v in enumerate(vs):
          print(f"    [{idx}]: {debug_info(v)}")
      elif isinstance(vs, dict):
        print(f"  {k}:")
        for _k, v in vs.items():
          print(f"    {_k}: {debug_info(v)}")
      else:
        print(f"  {k}: {debug_info(vs)}")
  except Exception as e:
    print(f"Failed to print debug info: {type(e).__name__}: {str(e)}")

class TaskCounter():
  def __init__(self, initial_value: int=0):
    self.value: int = initial_value
    self.lock = threading.Lock()

  def increase(self):
    """Increase counter"""
    with self.lock:
      self.value += 1
  
  def decrease(self):
    """Decrease counter"""
    with self.lock:
      assert self.value > 0, "No pending task, but decrease was called"
      self.value -= 1

  def reset(self, initial_value: int=0):
    """Reset counter"""
    with self.lock:
      self.value = initial_value

  def is_zero(self):
    """Check counter is zero"""
    return self.value == 0

class CloudpickleWrapper():
  def __init__(self, obj):
    self.packed_obj = obj
  
  def unpack(self):
    return self.packed_obj
  
  def __getstate__(self):
    return cloudpickle.dumps(self.packed_obj)
  
  def __setstate__(self, obj):
    self.packed_obj = cloudpickle.loads(obj)

def stream_process_worker(
  id: int,
  result_queue: multiprocessing.Queue,
  task_queue: multiprocessing.Queue,
  wrapped_fn: CloudpickleWrapper,
  stop_signal: multiprocessing.Event
):
  stream_producer = wrapped_fn.unpack()()
  while not stop_signal.is_set():
    gc.collect() # do something meaningful than sleep
    try:
      stream_info = task_queue.get(timeout=1)
      data = stream_producer.read_stream(stream_info)
      stream_info.data = data
      result_queue.put(stream_info)
    except queue.Empty:
      pass
    except Exception as e:
      print(f'{type(e).__name__}: {str(e)}')
      debug_data(data)
  # print process finished
  print(f"Process {id} finished")

class MultiprocessStreamProducer(StreamProducerWrapper):
  def __init__(
    self,
    stream_producer: BaseStreamProducer,
    stream_producer_fn: Callable,
    num_threads: Optional[int] = None,
    env_vars: Optional[Dict[str, str]] = None
  ):
    """Faster stream producer using multi processing.
    Usually this wrapper is wrapped at the final layer of your
    stream producer.

    Args:
        stream_producer (BaseStreamProducer): main stream producer for generating
          data specs and stream samples.
        stream_producer_fn (Callable): function for generating stream producers.
        num_threads (int, optional): number of processes. Defaults to None.
    """
    super().__init__(stream_producer)
    if num_threads is None:
      num_threads = multiprocessing.cpu_count()
    self.num_threads = num_threads
    self.env_vars = env_vars
    
    self.threads = []
    self.task_count = TaskCounter()
    self.result_queue: multiprocessing.Queue = None
    self.task_queue: multiprocessing.Queue = None
    self.stream_producer_fn = stream_producer_fn
    self.stop_signal: multiprocessing.Event = None # to stop processes

    self.steup_threads()

  def setup_threads(self):
    mp_methods = multiprocessing.get_all_start_methods()
    forkserver_available = 'forkserver' in mp_methods
    start_method = 'forkserver' if forkserver_available else 'spawn'
    ctx = multiprocessing.get_context(start_method)

    self.task_queue = ctx.Queue()
    self.result_queue = ctx.Queue()
    self.stop_signal = ctx.Event()

    # Set environment variables for spawned processes
    spawn_ev = self.env_vars or {}
    # Cached original environment variables
    orig_ev = {}
    for ev_key in spawn_ev.keys():
      orig_ev[ev_key] = copy.deepcopy(os.environ.get(ev_key))
      os.environ[ev_key] = spawn_ev[ev_key]
    
    # Spawn subprocesses
    self.threads = []
    for i in range(self.num_threads):
      remote_pipe, worker_pipe = ctx.Pipe(duplex=True) # Create connection pipe
      args = (
        i,
        self.result_queue,
        self.task_queue,
        CloudpickleWrapper(self.stream_producer_fn),
        self.stop_signal
      )
      thread = ctx.Process(target=stream_process_worker, args=args, daemon=True)
      thread.start()
      self.threads.append(thread)

    # Reset to original environment variables
    for ev_key in orig_ev.keys():
      if orig_ev[ev_key] is None:
        del os.environ[ev_key]
      else:
        os.environ[ev_key] = orig_ev[ev_key]

  def closed(self) -> bool:
    return self.stream_producer.closed()

  def close(self):
    self.stop_signal.set()
    for thread in self.threads:
      thread.terminate()
    # join all threads
    for thread in self.threads:
      thread.join(1)
    # close stream producer
    self.stream_producer.close()

  def prefetch(
    self,
    num_samples: Optional[int] = None
  ) -> "MultiprocessStreamProducer":
    """Prefetch streams.
    
    Args:
      num_samples (int, optional): number of streams to prefetch.
        if None or -1, all streams are placed into pending queue.
    """
    if num_samples is None or num_samples < 0:
      num_samples = self.buffer.qsize()
    for i in range(num_samples):
      self.maybe_recharge()
      self.sample_once()
    return self
  
  def sample_once(self):
    """Sample once from the stream producer buffer
    and put it into the stream queue
    Raises:
      queue.Empty: if buffer is empty
    """
    stream = self.get_sample()
    self.task_queue.put(stream)
    self.task_counter.increase()
  
  def read_stream(self, stream_info: StreamInfo) -> Any:
    return self.stream_producer.read_stream(stream_info)

  def get_stream(self, block: bool=True) -> Any:
    """Get one loaded stream and push a new tasks into stream pending
    queue.

    Args:
      block (bool, optional): whether to block threads until at least
        one stream is loaded. Defaults to True.

    Returns:
        Any: loaded stream
    """
    # push new tasks into pending queue if
    # sample buffer is not empty
    try:
      self.sample_once()
    except queue.Empty:
      pass
    # If task counter is not zero, that means some
    # tasks are still being processed.
    while not (self.task_counter.is_zero() or self.closed()):
      gc.collect() # do something meaningful than sleep
      try:
        # Try to get buffered data in result_queue
        data = self.result_queue.get(block=block, timeout=1)
        self.task_counter.decrease()
        return data
      except queue.Empty:
        if not block:
          break