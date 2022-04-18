# --- built in ---
import abc
from typing import (
  Optional,
  Callable
)
# --- 3rd party ---
# --- my module --

__all__ = [
  'BaseController'
]

class BaseController(metaclass=abc.ABCMeta):
  def __init__(self, env, action_map: Optional[Callable]=None):
    """BaseController

    Args:
      env (gym.Env): environment.
      action_map (Callable, optional): mapping function that
        maps habitat sim actions to environment actions.
        Defaults to DEFAULT_MAP.
    """
    self.env = env
    self.action_map = action_map

  @abc.abstractmethod
  def reset(self, *args, **kwargs):
    pass

  @abc.abstractmethod
  def act(self, observations):
    pass