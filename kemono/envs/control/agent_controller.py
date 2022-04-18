# --- built in ---
from typing import (
  Optional,
  Callable
)
# --- 3rd party ---
# --- my module ---
from kemono.envs.control.base_controller import BaseController

class AgentController(BaseController):
  def __init__(self, env, agent, action_map: Optional[Callable]=None):
    super().__init__(env, action_map)
    self.agent = agent
  
  def reset(self, *args, **kwargs):
    pass

  def act(self, observations):
    pass
