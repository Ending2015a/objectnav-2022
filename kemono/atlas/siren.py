# --- built in ---
import math
import copy
from typing import (
  Any,
  List,
  Tuple,
  Union,
  Callable,
  Optional
)
# --- 3rd party ---
import torch
from torch import nn
from torch.nn.init import _calculate_correct_fan
import einops
import rlchemy
from rlchemy.lib.nets import DelayedModule
from rlchemy import registry
# --- my module ---
from kemono.atlas import model as kemono_model


# borrowed from
# https://github.com/dalmia/siren/tree/master

def siren_uniform_(tensor: torch.Tensor, mode: str='fan_in', c: float=6):
  fan = _calculate_correct_fan(tensor, mode)
  std = 1 / math.sqrt(fan)
  bound = math.sqrt(c) * std # Calculate uniform bounds from standard deviation
  with torch.no_grad():
    tensor.uniform_(-bound, bound)


@registry.register.atlas_activ('sine')
class Sine(DelayedModule):
  def __init__(self, dim: Optional[int]=None, w0: float=1.0, **kwargs):
    super().__init__()
    self.w0 = w0
    self.mark_as_built()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return torch.sin(self.w0 * x)


@registry.register.atlas_net('siren')
class SIREN(DelayedModule):
  def __init__(
    self,
    dim: Optional[int] = None,
    mlp_units: List[int] = [64, 64],
    w0: float = 1.0,
    w0_initial: float = 30.0,
    initializer: str = 'siren',
    c: float = 6,
    final_activ: bool = False
  ):
    super().__init__()
    self.mlp_units = mlp_units
    self.w0 = w0
    self.w0_initial = w0_initial
    self.initializer = initializer
    self.c = c
    self.final_activ = final_activ
    # ---
    self.input_dim = None
    self.output_dim = None
    self._model = None
    if dim is not None:
      self.build(torch.Size([dim]))

  def build(self, input_shape: torch.Size):
    self.input_dim = input_shape[-1]
    in_dim = self.input_dim
    layers = []
    for idx, out_dim in enumerate(self.mlp_units):
      layers.append(nn.Linear(in_dim, out_dim))
      if idx+1 != len(self.mlp_units) or self.final_activ:
        layers.append(
          kemono_model.Activ(out_dim, 'sine', w0=self.w0)
        )
      in_dim = out_dim
    self._model = nn.Sequential(*layers)
    self.output_dim = out_dim
    if self.initializer == 'siren':
      for m in self._model.modules():
        if isinstance(m, nn.Linear):
          siren_uniform_(m.weight, mode='fan_in', c=self.c)
    self.mark_as_built()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    batches = x.shape[:-1]
    x = x.reshape(-1, x.shape[-1])
    x = self._model(x)
    x = x.reshape(*batches, x.shape[-1])
    return x
