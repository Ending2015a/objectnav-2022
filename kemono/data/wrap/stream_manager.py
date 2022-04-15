# --- built in ---
# --- 3rd party ---
# --- my module ---
from kemono.data.stream_producer import BaseStreamProducer, StreamInfo
from kemono.data.wrap.stream_producer_wrapper import StreamProducerWrapper

class StreamManager(StreamProducerWrapper):
  def __init__(
    self,
    stream_producer: BaseStreamProducer,
    max_to_draw: int = None,
    clean_unused: bool = False
  ):
    super().__init__(stream_producer)
    self.max_to_draw = max_to_draw
    self.clean_unused = clean_unused
  