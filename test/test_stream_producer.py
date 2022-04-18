# --- built in ---
import time
# --- 3rd party ---
import cv2
import numpy as np
import rlchemy
from omegaconf import OmegaConf
# --- my module ---
from kemono.data import RlchemyDynamicStreamProducer
from kemono.data.wrap import (
  ZeroPadChunkStreamWrapper,
  ResetMaskStreamWrapper,
  CombinedStreamProducer,
  MultiprocessStreamProducer,
  StreamProducerWrapper
)
from kemono.data.dataset import Dataset, StreamDataset
from kemono.data.sampler import StreamDatasetSampler
from kemono.data.callbacks import StreamManager

CONFIG_PATH = '/src/configs/kemono/kemono_train_config.yaml'

def make_stream_producer(configs):
  producers = []
  for key, config in configs.items():
    producer = RlchemyDynamicStreamProducer(
      **config.producer
    )
    if 'manager' in config:
      producer.register_callback(StreamManager(
        **config.manager
      ))
    producer = ResetMaskStreamWrapper(producer)
    producer = ZeroPadChunkStreamWrapper(
      producer,
      **config.zero_pad
    )
    producers.append(producer)
  producer = CombinedStreamProducer(producers, shuffle=True)
  return producer

def example():
  config = OmegaConf.load(CONFIG_PATH)
  OmegaConf.resolve(config)
  producer = make_stream_producer(config.train_dataset.stream_producer)
  dataset = StreamDataset(
    producer,
    **config.train_dataset.dataset
  )
  sampler = StreamDatasetSampler(
    dataset,
    producer,
    **config.train_dataset.sampler
  )
  for i in range(100):
    batch, caches = sampler.sample()
    print(i, batch['obs']['rgb'].shape)

if __name__ == '__main__':
  example()


# class SlowProducer(StreamProducerWrapper):
#   def read_stream(self, stream_info):
#     data = super().read_stream(stream_info)
#     time.sleep(1)
#     return data

# def make_producer():
#   producer = RlchemyDynamicStreamProducer(
#     data_root = '/src/logs/kemono_expert/trajs/',
#     glob_paths = '*.trajectory.npz'
#   )
#   producer = SlowProducer(producer)
#   return producer

# def example():
#   producer = MultiprocessStreamProducer(
#     stream_producer = make_producer(),
#     stream_producer_fn = make_producer,
#     num_threads = 2
#   )
#   print(producer.task_queue.qsize())
#   producer.prefetch(5)
#   print(producer.task_queue.qsize())
#   print(producer.result_queue.qsize())
#   time.sleep(5)
#   print(producer.result_queue.qsize())
#   while True:
#     stream_info = producer.get_stream(block=False)
#     if stream_info is None:
#       break
#     print(stream_info.path)
#     print(producer.task_queue.qsize(), producer.result_queue.qsize())
#   time.sleep(5)
#   print(producer.task_queue.qsize(), producer.result_queue.qsize())
#   producer.close()


if __name__ == '__main__':
  example()
