# --- built in ---
# --- 3rd party ---
import habitat
from omegaconf import OmegaConf
import numpy as np
# --- my module ---
import kemono


CONFIG_PATH = '/src/configs/atlas/test_multi_atlas.yaml'

def example():
  np.random.seed(1)
  conf = OmegaConf.load(CONFIG_PATH)
  config = kemono.get_config(conf.habitat_config)
  env = habitat.Env(config=config)
  sampler = kemono.atlas.AtlasSampler(**conf.sampler)

  for i in range(conf.num_episodes):
    env.reset()
    sampler.create_atlas(env, conf.sample.masking)
    sampler.sample_charts(
      conf.sample.max_charts,
      conf.sample.min_points,
      conf.sample.chart_width,
      conf.sample.chart_height
    )
    sampler.save_atlas(
      conf.sample.dirpath
    )

if __name__ == '__main__':
  example()