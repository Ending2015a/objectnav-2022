# --- built in ---
import os
import shutil
from typing import Optional
# --- 3rd party ---
# --- my module ---
from kemono.data.stream_producer import BaseStreamProducer

class StreamManager():
  def __init__(
    self,
    max_to_keep: Optional[int] = None,
    clean_unused: bool = False
  ):
    self.max_to_keep = max_to_keep
    self.clean_unused = clean_unused

  def on_load_stream_paths(self, stream_producer: BaseStreamProducer):
    if self.max_to_keep is not None:
      self._sweep(stream_producer)

  def _sweep(self, stream_producer: BaseStreamProducer):
    num_streams = len(stream_producer.stream_paths)
    unused_streams = []
    if num_streams > self.max_to_keep:
      stream_paths = stream_producer.stream_paths[-self.max_to_keep:]
      unused_streams = stream_producer.stream_paths[:self.max_to_keep]
      stream_producer.stream_paths = stream_paths
    if self.clean_unused:
      for unused in unused_streams:
        if os.path.isfile(unused):
          os.remove(unused)
        if os.path.isdir(unused):
          shutil.rmtree(unused)



