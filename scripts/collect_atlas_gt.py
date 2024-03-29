# --- built in ---
import copy
import argparse
from typing import (
  List,
  Union
)
# --- 3rd party ---
import habitat
import gym
import numpy as np
from omegaconf import OmegaConf
from habitat.datasets import make_dataset
# --- my module ---
import kemono
from kemono.envs.wrap import (
  SubprocVecEnv
)

class GTSamplerWrapper(gym.Wrapper):
  def __init__(self, env, config, index=0):
    self.env = env
    self._config = config
    self._index = index
    self.sampler = kemono.atlas.AtlasSampler(
      **config.sampler
    )

  def start_sample(self):
    # start sampling
    import tqdm
    try:
      total_episodes = len(self.env._dataset.episodes)
      pbar = tqdm.tqdm(
        range(total_episodes),
        desc = f'Process {self._index}',
        position = self._index,
        leave = True
      )
      for ep in pbar:
        self.env.reset()
        self.sampler.create_atlas(
          self.env,
          masking = True,
          find_gt_from_path = self._config.gt_dirpath
        )
        self.sampler.save_atlas(self._config.gt_dirpath)
    except StopIteration:
      print(f'Process {self._index} end.')

  def reset(self):
    return 0

  def step(self, action):
    self.start_sample()
    return 0, 0, 0, 0

class Env(habitat.Env):
  metadata = {"render_modes": []}
  reward_range = (-float('inf'), float('inf'))
  spec = None
  action_space = None
  observation_space = None
  def _setup_episode_iterator(self):
    assert self._dataset is not None
    self._episode_iterator = iter(self._dataset.episodes)

def get_gpus(
  processes: int,
  gpus: Union[int, List[int]]
):
  if isinstance(gpus, int):
    gpus = [gpus] * processes
  else:
    gpus = list(gpus)
  if len(gpus) < processes:
    gpus = gpus * processes
  return gpus[:processes]


def slice_dataset(
  habitat_config,
  config,
  num_splits,
  split_index,
):
  dataset = make_dataset(
    id_dataset = habitat_config.DATASET.TYPE,
    config = habitat_config.DATASET
  )
  num_episodes = config.num_episodes
  episodes = dataset.episodes[:num_episodes]
  num_episodes = len(episodes)
  # get chunks (every rep_ep size)
  iterator_opts = habitat_config.ENVIRONMENT.ITERATOR_OPTIONS
  rep_ep = iterator_opts.MAX_SCENE_REPEAT_EPISODES
  start = split_index * rep_ep
  episode_chunks = []
  for chunk_idx in range(start, num_episodes, rep_ep * num_splits):
    episode_chunks.extend(
      episodes[chunk_idx:chunk_idx+rep_ep]
    )
  new_dataset = copy.copy(dataset)
  new_dataset.episodes = episode_chunks
  del dataset
  return new_dataset

def create_env(habitat_config, config, index=0):
  # create omegaconf
  config = OmegaConf.create(config)
  # get gpu list
  gpus = get_gpus(config.num_processes, config.use_gpus)
  gpu = gpus[index]
  # fapply gpu config
  import torch
  torch.cuda.set_device(gpu)
  habitat_config.defrost()
  habitat_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = gpu
  if 'overwrite_data_path' in config:
    habitat_config.DATASET.DATA_PATH = config.overwrite_data_path
  if 'overwrite_scenes_dir' in config:
    habitat_config.DATASET.SCENES_DIR = config.overwrite_scenes_dir
  habitat_config.freeze()
  # Create dataset
  split_index = config.start_split + index
  dataset = slice_dataset(
    habitat_config,
    config,
    num_splits = config.num_splits,
    split_index = split_index
  )
  # Create base env
  env = Env(habitat_config, dataset)
  env = GTSamplerWrapper(env, config, index)
  return env

def create_env_fn(index, args):
  def _fn():
    return create_env(*args, index)
  return _fn

def main(args):
  config = kemono.utils.load_config(args.config, resolve=True)
  habitat_config = kemono.get_config(config.habitat_config)
  
  env_fns = [
    create_env_fn(i, (habitat_config, config))
    for i in range(config.num_processes)
  ]
  env = SubprocVecEnv(env_fns)
  # start sampling
  env.reset()
  env.step([None] * config.num_processes)
  env.close()



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', type=str, default='/src/configs/atlas/atlas_sampler.yaml')

  args = parser.parse_args()
  main(args)