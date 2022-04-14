# --- built in ---
from typing import Any, List
# --- 3rd party ---
import gym
import ray
# --- my module ---

__all__ = [
  'RayRemoteEnv'
]

class _RayRemoteController():
  def __init__(self, env_fn, remote_args: List[Any]=[]):
    remote_args = remote_args or []
    self.env = env_fn(*remote_args)
  
  def step(self, action):
    return self.env.step(action)

  def reset(self):
    return self.env.reset()

  def close(self):
    return self.env.close()

  def render(self, mode='human'):
    return self.env.render(mode=mode)

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
    num_gpus: float = 0.5
  ):
    self.env_fn = env_fn
    self.remote_args = remote_args or []
    self.num_cpus = num_cpus
    self.num_gpus = num_gpus
    # wrap the remote controller with ray remote
    self.ray_class = ray.remote(
      num_cpus = self.num_cpus,
      num_gpus = self.num_gpus
    )(_RayRemoteController)
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

  def seed(self):
    pass

  def render(self, mode='human'):
    return ray.get(self.env.render.remote(mode=mode))

  def getattr(self, name):
    return ray.get(self.env._getattr.remote(name))

  def setattr(self, name, value):
    return ray.get(self.env._setattr.remote(name, value))
