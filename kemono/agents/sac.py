# --- built in ---
from typing import (
  Any,
  Optional
)
# --- 3rd party ---
import torch
from torch import nn
import rlchemy
from rlchemy.nets import DelayedModule
# --- my module ---

class CategoricalPolicyNet(DelayedModule):
  def __init__(
    self,
    dim: Optional[int] = None,
    n_actions: Optional[int] = None
  ):
    """Categorical policy for discrete actions

    Args:
      dim (Optional[int], optional): input dimension. Defaults to None.
      n_actions (Optional[int], optional): number of discrete actions.
        Defaults to None.
    """
    super().__init__()
    self.n_actions = n_actions
    # ---
    self.input_dim = None
    self.output_dim = None
    self._model = None
    if dim is not None:
      self.build(torch.Size([dim]))

  def build(self, input_shape: torch.Size):
    in_dim = input_shape[-1]
    out_dim = self.action_space.n
    self._model = self.make_model(in_dim, out_dim)
    self.input_dim = in_dim
    self.output_dim = out_dim
    self.mark_as_built()

  def forward(
    self,
    x: torch.Tensor,
  ):
    pass

