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
    ),
    'objectgoal': gym.spaces.Discrete(6)
  })
  action_space = gym.spaces.Discrete(4)
  agent = Agent(
    observation_space,
    action_space,
    policy = config.sac.policy,
    value = config.sac.value
  )
  agent.setup()
  print(agent.policy)

  rgb = torch.randn((2, 3, 3, 128, 128), dtype=torch.float32)
  large_map = torch.randn((2, 3, 3, 128, 128), dtype=torch.float32)
  small_map = torch.randn((2, 3, 3, 128, 128), dtype=torch.float32)
  objectgoal = torch.randint(0, 6, (2, 3), dtype=torch.int64)
  x = dict(rgb=rgb, large_map=large_map, small_map=small_map, objectgoal=objectgoal)
  act, states, val, history = agent(x)
  assert act.shape == (2, 3)
  assert states['policy'][0].shape == (3, 1024)
  assert states['policy'][1].shape == (3, 1024)
  assert states['value'][0].shape == (3, 1024)
  assert states['value'][1].shape == (3, 1024)
  assert val.shape == (2, 3, 1)
  assert history['policy'][0].shape == (2, 3, 1024)
  assert history['policy'][1].shape == (2, 3, 1024)
  assert history['value'][0].shape == (2, 3, 1024)
  assert history['value'][1].shape == (2, 3, 1024)

  # print(act.shape)
  # print(states['policy'][0].shape, states['policy'][1].shape)
  # print(states['value'][0].shape, states['value'][1].shape)
  # print(val.shape)
  # print(history['policy'][0].shape, history['policy'][1].shape)
  # print(history['value'][0].shape, history['value'][1].shape)

  rgb = torch.randn((3, 128, 128), dtype=torch.float32)
  large_map = torch.randn((3, 128, 128), dtype=torch.float32)
  small_map = torch.randn((3, 128, 128), dtype=torch.float32)
  objectgoal = torch.tensor(1, dtype=torch.int64)
  x = dict(rgb=rgb, large_map=large_map, small_map=small_map, objectgoal=objectgoal)
  act, states = agent.predict(x)
  assert act.shape == tuple()
  assert states['policy'][0].shape == (1, 1024)
  assert states['policy'][1].shape == (1, 1024)
  assert states['value'][0].shape == (1, 1024)
  assert states['value'][1].shape == (1, 1024)
  # print(act.shape)
  # print(states['policy'][0].shape, states['policy'][1].shape)
  # print(states['value'][0].shape, states['value'][1].shape)

  rgb = torch.randn((3, 128, 128), dtype=torch.float32)
  large_map = torch.randn((3, 128, 128), dtype=torch.float32)
  small_map = torch.randn((3, 128, 128), dtype=torch.float32)
  objectgoal = torch.tensor(1, dtype=torch.int64)
  x = dict(rgb=rgb, large_map=large_map, small_map=small_map, objectgoal=objectgoal)
  act, states = agent.predict(x, states)
  assert act.shape == tuple()
  assert states['policy'][0].shape == (1, 1024)
  assert states['policy'][1].shape == (1, 1024)
  assert states['value'][0].shape == (1, 1024)
  assert states['value'][1].shape == (1, 1024)
  # print(act.shape)
  # print(states['policy'][0].shape, states['policy'][1].shape)
  # print(states['value'][0].shape, states['value'][1].shape)

if __name__ == '__main__':
  example()
