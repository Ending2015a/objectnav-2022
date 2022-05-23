# --- built in ---
import os
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
  config
):
  print(f'Loading dataset')
  dataset = make_dataset(
    id_dataset = habitat_config.DATASET.TYPE,
    config = habitat_config.DATASET
  )
  num_episodes = config.num_episodes

  iter_option_dict = {
    k.lower(): v
    for k, v in habitat_config.ENVIRONMENT.ITERATOR_OPTIONS.items()
  }
  iter_option_dict["seed"] = habitat_config.SEED
  iter_option_dict["cycle"] = False
  episode_iterator = dataset.get_episode_iterator(
    **iter_option_dict
  )
  print(f'Sampling dataset')
  episodes = []
  try:
    for _ in range(config.slice_dataset.skip_episodes):
      next(episode_iterator)
    # note that here we use dataset iterator to get
    # the sequence of episodes. The order of episodes
    # is expected to be same as we run habitat.Env.
    import tqdm
    for ep in tqdm.tqdm(range(num_episodes)):
      episodes.append(next(episode_iterator))
  except StopIteration:
    pass
  new_dataset = copy.copy(dataset)
  new_dataset.episodes = episodes
  return new_dataset

def main(args):
  config = kemono.utils.load_config(args.config, resolve=True)
  habitat_config = kemono.get_config(config.habitat_config)

  dataset = slice_dataset(habitat_config, config)
  dataset_json = dataset.to_json()
  data_path = config.slice_dataset.data_path
  os.makedirs(os.path.dirname(data_path), exist_ok=True)
  # save dataset to json.gz
  import gzip
  with gzip.open(config.slice_dataset.data_path, 'wt') as f:
    f.write(dataset_json)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', type=str, default='/src/configs/atlas/atlas_sampler.yaml')

  args = parser.parse_args()
  main(args)