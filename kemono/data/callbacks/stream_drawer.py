# --- built in ---
import random
from typing import Optional
# --- 3rd party ---
# --- my module ---
from kemono.data.stream_producer import BaseStreamProducer

class StreamDrawer():
  def __init__(
    self,
    max_to_draw: Optional[int] = None
  ):
    self.max_to_draw = max_to_draw

  def on_load_stream_paths(self, stream_producer: BaseStreamProducer):
    if self.max_to_draw is not None:
      self._draw(stream_producer)
  
  def _draw(self, stream_producer: BaseStreamProducer):
    num_streams = len(stream_producer.stream_paths)
    if num_streams > self.max_to_draw:
      stream_paths = stream_producer.stream_paths
      random.shuffle(stream_paths)
      stream_producer.stream_paths = stream_paths[:self.max_to_draw]