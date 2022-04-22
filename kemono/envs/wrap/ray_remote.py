# --- built in ---
from typing import Any, List, Dict, Tuple
# --- 3rd party ---
import cv2
import gym
import ray
import rlchemy
import numpy as np
# --- my module ---

__all__ = [
  'RayRemoteEnv',
  'VecRayRemoteEnv'
]

class _RayRemoteWorker():
  def __init__(
    self,
    env_fn,
    remote_args: List[Any]=[],
    remote_kwargs: Dict[str, Any]={}
  ):
    remote_args = remote_args or []
    remote_kwargs = remote_kwargs or {}
    self.env = env_fn(*remote_args, **remote_kwargs)
  
  def step(self, action):
    return self.env.step(action)

  def reset(self):
    return self.env.reset()

  def close(self):
    return self.env.close()

  def seed(self, seed):
    return self.env.seed(seed)

  def render(self, mode='human'):
    res = self.env.render(mode=mode)
    if mode in ['human', 'interact']:
      cv2.waitKey(1)
    return res

  def call(self, func, *args, **kwargs):
    return getattr(self.env, func)(*args, **kwargs)

  def _getattr(self, name):
    """Get attribute from env
    Note that here we use an underscore to prevent from
    overriding the ray getattr function
    """
    return getattr(self.env, name)

  def _setattr(self, name, value):
    """Set env's attribute
    Note that here we use an underscore to prevent from
    overriding the ray setattr function
    """
    return setattr(self.env, name, value)

class RayRemoteEnv(gym.Env):
  def __init__(
    self,
    env_fn,
    remote_args: List[Any] = [],
    num_cpus: int = 1,
    num_gpus: float = 1
  ):
    self.env_fn = env_fn
    self.remote_args = remote_args or []
    self.num_cpus = num_cpus
    self.num_gpus = num_gpus
    # wrap the remote controller with ray remote
    self.ray_class = ray.remote(
      num_cpus = self.num_cpus,
      num_gpus = self.num_gpus
    )(_RayRemoteWorker)
    # instantiate remote env
    self.env = self.setup_env()
    # get some attributes from the remote env
    self.observation_space = self.getattr('observation_space')
    self.action_space = self.getattr('action_space')
    self.metadata = self.getattr('metadata')
    self.reward_range = self.getattr('reward_range')
    self.spec = self.getattr('spec')

  def setup_env(self):
    """Instantiate ray remote controller"""
    return self.ray_class.remote(self.env_fn, self.remote_args)

  def step(self, action):
    return ray.get(self.env.step.remote(action))

  def reset(self):
    return ray.get(self.env.reset.remote())

  def close(self):
    return ray.get(self.env.close.remote())

  def seed(self, seed):
    return ray.get(self.env.seed.remote(seed))

  def render(self, mode='human'):
    return ray.get(self.env.render.remote(mode=mode))

  def getattr(self, name):
    return ray.get(self.env._getattr.remote(name))

  def setattr(self, name, value):
    return ray.get(self.env._setattr.remote(name, value))


def stack_op(data_tuple: Tuple[Any, ...]):
  return np.stack(data_tuple, axis=0)

class VecRayRemoteEnv(gym.Env):
  def __init__(
    self,
    env_fn,
    remote_args: List[Any] = [],
    n_envs: int = 1,
    num_cpus: int = 1,
    num_gpus: float = 1,
  ):
    self.env_fn = env_fn
    self.remote_args = remote_args or []
    self.n_envs = n_envs
    self.num_cpus = num_cpus
    self.num_gpus = num_gpus
    # wrap the remote controller with ray remote
    self.ray_class = ray.remote(
      num_cpus = self.num_cpus,
      num_gpus = self.num_gpus
    )(_RayRemoteWorker)
    # instantiate remote envs
    self.envs = [self.setup_env(i) for i in range(n_envs)]
    # get attributes from the remote env
    self.observation_spaces = self.getattrs('observation_space')
    self.action_spaces = self.getattrs('action_space')
    self.metadatas = self.getattrs('metadata')
    self.reward_ranges = self.getattrs('reward_range')
    self.specs = self.getattrs('spec')

    self.observation_space = self.observation_spaces[0]
    self.action_space = self.action_spaces[0]
    self.metadata = self.metadatas[0]
    self.reward_range = self.reward_ranges[0]
    self.spec = self.specs[0]
  
  def setup_env(self, index=None):
    remote_kwargs = {'index': index}
    return self.ray_class.remote(
      self.env_fn,
      self.remote_args,
      remote_kwargs
    )

  def _stack(self, data_tuple: Tuple[Any, ...]):
    return rlchemy.utils.map_nested_tuple(
      tuple(data_tuple), op=stack_op)

  def reset(self):
    res = ray.get([env.reset.remote() for env in self.envs])
    return self._stack(res)
  
  def step(self, actions):
    res = ray.get([
      env.step.remote(act)
      for env, act in zip(self.envs, actions)
    ])
    obs_list, rew_list, done_list, info_list = zip(*res)
    # stack results
    obs = self._stack(obs_list)
    rew = self._stack(rew_list)
    done = self._stack(done_list)
    return obs, rew, done, info_list

  def seed(self, seed):
    res = ray.get([
      env.seed.remote(seed + idx)
      for idx, env in enumerate(self.envs)
    ])
    return res

  def render(self, mode='human'):
    res = ray.get([
      env.render.remote(mode=mode)
      for env in self.envs
    ])
    return res

  def close(self):
    res = ray.get([
      env.close.remote()
      for env in self.envs
    ])
    return res

  def getattrs(self, name):
    res = ray.get([
      env._getattr.remote(name)
      for env in self.envs
    ])
    return res

  def setattrs(self, name, value):
    res = ray.get([
      env._setattr.remote(name, value)
      for env in self.envs
    ])
    return res
