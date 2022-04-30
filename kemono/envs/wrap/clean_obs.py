# --- built in ---
import copy
from dataclasses import dataclass
from typing import (
  Optional
)
# --- 3rd party ---
import numpy as np
import torch
from torch import nn
import gym
import cv2
# --- my module ---
from kemono.semantics.utils.image_utils import resize_image

@dataclass
class ResizeConfig:
  width: Optional[int] = None
  height: Optional[int] = None
  mode: str = 'nearest'


@dataclass
class ObsConfig:
  type: str = 'vector'
  resize: Optional[ResizeConfig] = None
  channel_first: Optional[bool] = None
  invert_color: Optional[bool] = None
  def __post_init__(self):
    if self.resize is not None:
      self.resize = ResizeConfig(**self.resize)

class CleanObsWrapper(gym.Wrapper):
  def __init__(self, env, obs_configs, video_config):
    super().__init__(env=env)
    self.obs_config = {
      key: ObsConfig(**config)
      for key, config in obs_configs.items()
    }
    self.video_config = video_config
    self.observation_space = self.make_observation_space()
    self._cached_obs = None
    self._cached_canvas = None

  def reset(self, *args, **kwargs):
    obs = self.env.reset(*args, **kwargs)
    self._cached_obs = obs
    return self.get_observation(obs)
  
  def step(self, action, *args, **kwargs):
    obs, rew, done, info = self.env.step(action, *args, **kwargs)
    self._cached_obs = obs
    return self.get_observation(obs), rew, done, info

  def get_observation(self, obs):
    self._cached_canvas = None
    new_obs = {}
    for key in self.obs_config.keys():
      config = self.obs_config[key]
      o = obs[key]
      if config.type == 'image':
        assert len(o.shape) == 3, f"Observation `{key}` must be an iamge space"
        o = np.transpose(o, (2, 0, 1)) # (c, h, w)
        if config.resize:
          dtype = o.dtype
          height = config.resize.height
          width = config.resize.width
          mode = config.resize.mode
          o = resize_image(o, (height, width), mode=mode).numpy()
        if not config.channel_first:
          o = np.transpose(o, (1, 2, 0)).astype(dtype)
        if config.invert_color:
          o = 255 - o
      elif config.type == 'scalar':
        o = np.asarray(o).item()
      elif config.type == 'vector':
        if config.reshape:
          o = np.asarray(o).reshape(config.reshape)
      new_obs[key] = o
    return new_obs

  def make_observation_space(self):
    if self.env.observation_space is None:
      return None
    obs_space = self.observation_space
    new_obs_spaces = {}
    for key in self.obs_config.keys():
      config = self.obs_config[key]
      space = obs_space[key]
      if isinstance(space, gym.spaces.Box):
        low = np.min(space.low)
        high = np.max(space.high)
        shape = space.shape
        dtype = space.dtype
        if config.resize:
          assert len(space.shape) == 3, f"Space `{key}` must be an image space"
          height = config.resize.height or shape[0]
          width = config.resize.width or shape[1]
          shape = (height, width, shape[-1])
          config.resize.height = height
          config.resize.width = width
        if config.channel_first:
          assert len(space.shape) == 3, f"Space `{key}` must be an image space"
          shape = (shape[2], shape[0], shape[1])
        new_space = gym.spaces.Box(
          low = low,
          high = high,
          shape = shape,
          dtype = dtype
        )
      else:
        new_space = space
      new_obs_spaces[key] = new_space
    new_obs_space = gym.spaces.Dict(new_obs_spaces)
    return new_obs_space

  def render(self, mode='human'):
    if mode == 'interact':
      return self.env.render(mode=mode)
    # rendering mode is human or rgb_array
    goal_id = self._cached_obs[self.video_config.goal]
    goal_name = self.video_config.goal_mapping[np.asarray(goal_id).item()]
    canvas = []
    if self._cached_canvas is None:
      for idx, key in enumerate(self.video_config.image):
        label = self.video_config.label[idx]
        size = tuple(self.video_config.size[idx])
        image = self._cached_obs[key]
        image = cv2.resize(image, size, cv2.INTER_NEAREST)
        image = np.pad(image, ((2, 2), (2, 2), (0, 0)), constant_values=100)
        image = np.pad(image, ((100, 20), (20, 20), (0,0)), constant_values=255)
        if idx == 0:
          label = label + f' (Goal: {goal_name})'
        image = cv2.putText(image, label, (70, 70), cv2.FONT_HERSHEY_TRIPLEX,
          0.65, (0, 0, 0), 1, cv2.LINE_AA)
        canvas.append(image)
      canvas = np.concatenate(canvas, axis=1)
      self._cached_canvas = canvas
    else:
      canvas = self._cached_canvas
    if mode == 'human':
      cv2.imshow('Habitat', canvas[...,::-1])
      cv2.waitKey(1)
    return canvas
