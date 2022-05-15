# --- built in ---
# --- 3rd party ---
import habitat
from omegaconf import OmegaConf
import numpy as np
# --- my module ---
import kemono


CONFIG_PATH = '/src/configs/gsm/test_multi_atlas.yaml'

def example():
  np.random.seed(1)
  conf = OmegaConf.load(CONFIG_PATH)
  config = kemono.get_config(conf.habitat_config)
  env = habitat.Env(config=config)
  sampler = kemono.gsm.GsmDataSampler(env, **conf.sampler)

  for i in range(conf.num_episodes):
    env.reset()
    sampler.sample(**conf.sample)

if __name__ == '__main__':
  example()