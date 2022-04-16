# --- built in ---
import time
# --- 3rd party ---
import cv2
import numpy as np
import rlchemy
# --- my module ---
from kemono.data import RlchemyDynamicStreamProducer
from kemono.data.wrap import (
  ZeroPadChunkStreamWrapper,
  ResetMaskStreamWrapper,
  CombinedStreamProducer,
  MultiprocessStreamProducer,
  StreamProducerWrapper
)
from kemono.data.callbacks import StreamManager

class SlowProducer(StreamProducerWrapper):
  def read_stream(self, stream_info):
    data = super().read_stream(stream_info)
    time.sleep(1)
    return data

def make_producer():
  producer = RlchemyDynamicStreamProducer(
    data_root = '/src/logs/kemono_expert/trajs/',
    glob_paths = '*.trajectory.npz'
  )
  producer = SlowProducer(producer)
  return producer

def example():
  producer = MultiprocessStreamProducer(
    stream_producer = make_producer(),
    stream_producer_fn = make_producer,
    num_threads = 2
  )
  print(producer.task_queue.qsize())
  producer.prefetch(5)
  print(producer.task_queue.qsize())
  print(producer.result_queue.qsize())
  time.sleep(5)
  print(producer.result_queue.qsize())
  while True:
    stream_info = producer.get_stream(block=False)
    if stream_info is None:
      break
    print(stream_info.path)
    print(producer.task_queue.qsize(), producer.result_queue.qsize())
  time.sleep(5)
  print(producer.task_queue.qsize(), producer.result_queue.qsize())
  producer.close()


if __name__ == '__main__':
  example()
