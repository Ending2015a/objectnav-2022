# --- built in ---
# --- 3rd party ---
import numpy as np
import torch
# --- my module ---
from kemono.data import RlchemyDynamicStreamProducer
from kemono.data.wrap import ZeroPadChunkStreamWrapper, ResetMaskStreamWrapper
from kemono.data.dataset import StreamDataset
from kemono.data import StreamDatasetSampler


def example():
  producer = RlchemyDynamicStreamProducer(
    data_root = '/src/logs/kemono_expert/trajs/',
    glob_paths = '*.trajectory.npz'
  )
  producer = ZeroPadChunkStreamWrapper(producer, chunk_size=10)
  producer = ResetMaskStreamWrapper(producer)
  dataset = StreamDataset(
    producer,
    num_workers = 6,
    group_size = 3,
    seq_len = 10,
    drop_remainder = False
  )
  sampler = StreamDatasetSampler(
    dataset,
    producer,
    num_workers = 6,
    batch_size = 3,
    drop_remainder = False,
    return_indices = True
  )
  counter = np.zeros((6,), dtype=np.int64)
  data, caches, ind = sampler.sample()
  print('obs.rgb', data['obs']['rgb'].shape)
  print('obs.large_map', data['obs']['large_map'].shape)
  print('obs.small_map', data['obs']['small_map'].shape)
  print('obs.objectgoal', data['obs']['objectgoal'].shape)
  print('next_obs.rgb', data['next_obs']['rgb'].shape)
  print('next_obs.large_map', data['next_obs']['large_map'].shape)
  print('next_obs.small_map', data['next_obs']['small_map'].shape)
  print('next_obs.objectgoal', data['next_obs']['objectgoal'].shape)
  print('act', data['act'].shape)
  print('rew', data['rew'].shape)
  print('done', data['done'].shape)
  print('reset', data['reset'].shape)
  print('mask', data['mask'].shape)
  assert caches is None
  counter[ind] += 1
  
  caches = {
    'policy': np.ones((3, 1), dtype=np.int64),
    'value': np.ones((3, 1), dtype=np.int64)
  }
  sampler.cache(**caches)

  policy_cache = torch.as_tensor(sampler._caches.data['policy'])
  value_cache = torch.as_tensor(sampler._caches.data['value'])
  # Note that the cache is saved as a list
  assert len(sampler._caches.data['policy']) == 6
  assert len(sampler._caches.data['value']) == 6
  for i in range(6):
    assert sampler._caches.data['policy'][i].shape == (1,)
    assert sampler._caches.data['value'][i].shape == (1,)

  for i in range(100):
    data, caches, ind = sampler.sample()
    counter[ind] += 1
    caches['policy'] += 1
    caches['value'] += 1
    sampler.cache(**caches)

  for i in range(6):
    assert np.all(sampler._caches.data['policy'][i].numpy() == counter[i])


if __name__ == '__main__':
  example()

