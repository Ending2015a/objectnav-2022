# --- built in ---
# --- 3rd party ---
import numpy as np
import torch
from torch import nn
from omegaconf import OmegaConf
import gym
# --- my module ---
from kemono.agents.models.sac import Agent

CONFIG_PATH = '/src/configs/kemono/kemono_train_config.yaml'

def example():
  config = OmegaConf.load(CONFIG_PATH)
  OmegaConf.resolve(config)
  observation_space = gym.spaces.Dict({
    'rgb': gym.spaces.Box(
      low = 0,
      high = 255,
      shape = (3, 128, 128),
      dtype = np.uint8
    ),
    'large_map': gym.spaces.Box(
      low = 0,
      high = 255,
      shape = (3, 128, 128),
      dtype = np.uint8
    ),
    'small_map': gym.spaces.Box(
      low = 0,
      high = 255,
      shape = (3, 128, 128),
      dtype = np.uint8
    )
  })
  action_space = gym.spaces.Discrete(4)
  agent = Agent(
    observation_space,
    action_space,
    policy = config.sac.policy,
    value = config.sac.value
  )
  agent.setup()

  rgb = torch.randn((2, 3, 3, 128, 128), dtype=torch.float32)
  large_map = torch.randn((2, 3, 3, 128, 128), dtype=torch.float32)
  small_map = torch.randn((2, 3, 3, 128, 128), dtype=torch.float32)
  x = dict(rgb=rgb, large_map=large_map, small_map=small_map)
  act, states, val, history = agent(x)
  print(act.shape)
  print(states['policy'][0].shape, states['policy'][1].shape)
  print(states['value'][0].shape, states['value'][1].shape)
  print(val.shape)
  print(history['policy'][0].shape, history['policy'][1].shape)
  print(history['value'][0].shape, history['value'][1].shape)

  rgb = torch.randn((3, 128, 128), dtype=torch.float32)
  large_map = torch.randn((3, 128, 128), dtype=torch.float32)
  small_map = torch.randn((3, 128, 128), dtype=torch.float32)
  x = dict(rgb=rgb, large_map=large_map, small_map=small_map)
  act, states = agent.predict(x)
  print(act.shape)
  print(states['policy'][0].shape, states['policy'][1].shape)
  print(states['value'][0].shape, states['value'][1].shape)

  rgb = torch.randn((3, 128, 128), dtype=torch.float32)
  large_map = torch.randn((3, 128, 128), dtype=torch.float32)
  small_map = torch.randn((3, 128, 128), dtype=torch.float32)
  x = dict(rgb=rgb, large_map=large_map, small_map=small_map)
  act, states = agent.predict(x, states)
  print(act.shape)
  print(states['policy'][0].shape, states['policy'][1].shape)
  print(states['value'][0].shape, states['value'][1].shape)

if __name__ == '__main__':
  example()
