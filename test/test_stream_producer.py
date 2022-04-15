# --- built in ---
import os
import argparse
# --- 3rd party ---
import cv2
import numpy as np
import rlchemy
# --- my module ---
from kemono.data import RlchemyDynamicStreamProducer
from kemono.data.wrap import CombinedStreamProducer

def example(args):
  producer_1 = RlchemyDynamicStreamProducer(
    data_root = '/src/logs/kemono_expert/trajs/',
    glob_paths = '*.trajectory.npz'
  )
  producer_2 = RlchemyDynamicStreamProducer(
    data_root = '/src/logs/kemono_expert_val/trajs/',
    glob_paths = '*.trajectory.npz'
  )
  producer = CombinedStreamProducer(
    [producer_1, producer_2],
  )
  producer_len = len(producer)
  while True:
    stream_info = producer.get_stream()
    if stream_info is None:
      break
    print(stream_info.info.path)
  assert producer.get_stream() == None
  assert len(producer) == 0
  producer.maybe_recharge()
  assert len(producer) == producer_len
  for stream_info in producer(iterate=True):
    print(stream_info.info.path)
  assert len(producer) == 0


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--root', type=str, default='/src/logs/kemono_expert/trajs')
  parser.add_argument('--glob', type=str, default='*.trajectory.npz')
  args = parser.parse_args()
  example(args)
