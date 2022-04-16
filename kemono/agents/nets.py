# --- built in ---
import copy
from typing import (
  Any,
  List,
  Dict,
  Tuple,
  Union,
  Callable,
  Optional
)
# --- 3rd party ---
import numpy as np
import torch
from torch import nn
import rlchemy
from rlchemy.lib.nets import DelayedModule
from rlchemy import registry
import gym
from omegaconf import OmegaConf
# --- my module ---

__all__ = [
  'AwesomeMlp',
  'AwesomeCnn',
  'AwesomeBackbone',
  'AwesomeRnn'
]

@registry.register.kemono_net('mlp', default=True)
class AwesomeMlp(DelayedModule):
  def __init__(
    self,
    dim: Optional[int] = None,
    mlp_units: List[int] = [64, 64]
  ):
    super().__init__()
    self.mlp_units = mlp_units
    # ---
    self.input_dim = None
    self.output_dim = None
    self._model = None
    if dim is not None:
      if isinstance(dim, int):
        dim = [dim]
      self.build(torch.Size(dim))

  def build(self, input_shape: torch.Size):
    self.input_dim = input_shape[-1]
    in_dim = self.input_dim
    layers = []
    for out_dim in self.mlp_units:
      layers.extend([
        nn.Linear(in_dim, out_dim),
        nn.LeakyReLU(0.01, inplace=True)
      ])
      in_dim = out_dim
    self._model = nn.Sequential(*layers)
    self.output_dim = out_dim
    self.mark_as_built()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # expecting 2D tensor.
    # if nD tensor n > 2, then flatten to 2D
    orig_ndims = len(x.shape)
    batches = x.shape[:-1]
    x = x.view(-1, x.shape[-1])
    x = self._model(x)
    x = x.view(*batches, x.shape[-1])
    return x


@registry.register.kemono_net('cnn')
class AwesomeCnn(DelayedModule):
  def __init__(
    self,
    shape: Tuple[int, ...] = None,
    mlp_units: List[int] = [512]
  ):
    super().__init__()
    self.mlp_units = mlp_units
    # ---
    self.input_shape = None
    self.input_dim = None
    self.output_dim = None
    self._model = None
    if shape is not None:
      self.build(torch.Size(shape))
    
  def build(self, input_shape: torch.Size):
    assert len(input_shape) >= 3
    input_shape = input_shape[-3:]
    dim = input_shape[0]
    cnn = nn.Sequential(
      nn.Conv2d(dim, 32, 8, 4, padding=0),
      nn.LeakyReLU(0.01, inplace=True),
      nn.Conv2d(32, 64, 4, 2, padding=0),
      nn.LeakyReLU(0.01, inplace=True),
      nn.Conv2d(64, 64, 3, 1, padding=0),
      nn.LeakyReLU(0.01, inplace=True),
      nn.Flatten(start_dim=-3, end_dim=-1)
    )
    # forward cnn to get output size
    dummy = torch.zeros((1, *input_shape), dtype=torch.float32)
    outputs = cnn(dummy).detach()
    # create mlp layers
    mlp = AwesomeMlp(outputs.shape[-1], mlp_units=self.mlp_units)
    self._model = nn.Sequential(*cnn, mlp)
    self.output_dim = mlp.output_dim
    self.mark_as_built()
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # expecting 4D tensor.
    # if nD tensor n > 4, then flatten to 4D
    orig_ndims = len(x.shape)
    batches = x.shape[:-3]
    x = x.view(-1, *x.shape[-3:])
    x = self._model(x)
    x = x.view(*batches, x.shape[-1])
    return x


class AwesomeBackbone(DelayedModule):
  def __init__(
    self,
    net_config: Dict[str, Any],
    fuse_config: Dict[str, Any]
  ):
    super().__init__()
    self.net_config = net_config
    self.fuse_config = fuse_config
    # ---
    self._models = None
    self._fuse_model = None

  def get_input_shapes(
    self, x: Any, *args, **kwargs
  ):
    shapes = {}
    for key, tensor in x.items():
      shapes[key] = tensor.shape
    return shapes

  def build(
    self,
    input_shapes: Dict[str, Any]
  ):
    models = {}
    out_dim = 0
    for key, shape in input_shapes.items():
      assert key in self.net_config
      config = self.net_config[key]
      net_class = registry.get.kemono_net(config.type)
      if net_class is not None:
        config = copy.deepcopy(config)
        config.pop('type', None)
        model = net_class(shape, **config)
        out_dim += model.output_dim
      else:
        model = nn.Identity()
        out_dim += shape[-1]
      models[key] = model
    self._models = nn.ModuleDict(models)
    net_class = registry.get.kemono_net(self.fuse_config.type)
    if net_class is not None:
      config = copy.deepcopy(self.fuse_config)
      config.pop('type', None)
      self._fuse_model = net_class(out_dim, **config)
      out_dim = self._fuse_model.output_dim
    else:
      # if type is not specified or does not exist
      self._fuse_model = nn.Identity()
    self.output_dim = out_dim
    self.mark_as_built()

  def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    res = []
    for key, tensor in x.items():
      tensor = self._models[key](tensor)
      res.append(tensor)
    x = torch.cat(res, dim=-1)
    return self._fuse_model(x, *args, **kwargs)


@registry.register.kemono_cell('lstm', default=True)
class AwesomeLstm(nn.Module):
  def __init__(
    self,
    dim: int = 1024,
    units: int = 1024
  ):
    super().__init__()
    self.units = units
    self.input_dim = dim
    self.output_dim = units
    self.cell = nn.LSTMCell(dim, units)
  
  def forward(
    self,
    x: torch.Tensor,
    states: Tuple[torch.Tensor, ...],
    reset: Optional[torch.Tensor]
  ):
    """Forward LSTM Cell

    Args:
      x (torch.Tensor): input tensor, expecting (seq, b, dim)
      states (Tuple[torch.Tensor, ...]): initial states
      reset (Optional[torch.Tensor]): sequence of reset signal to reset
        the input hidden/cell states.

    Returns:
      torch.Tensor: sequence outputs
      Tuple[Tuple[torch.Tensor, ...], ...]: sequence of states
    """
    seq = x.shape[0]
    hx, cx = states
    out = []
    states = []
    mask = 1.0 - reset
    for i in range(seq):
      # reset hidden/cell states to zero if reset = True
      hx, cx = self.cell(x[i], (hx * mask[i], cx * mask[i]))
      out.append(hx)
      states.append((hx, cx))
    return torch.stack(out, dim=0), tuple(states)

  def get_states(self, batch_size=1, device='cuda'):
    return (
      torch.zeros((batch_size, self.units), device=device),
      torch.zeros((batch_size, self.units), device=device)
    )

@registry.register.kemono_cell('gru')
class AwesomeGru(nn.Module):
  def __init__(
    self,
    dim: int = 1024,
    units: int = 1024
  ):
    super().__init__()
    self.units = units
    self.input_dim = dim
    self.output_dim = units
    self.cell = nn.GRUCell(dim, units)
  
  def forward(
    self,
    x: torch.Tensor,
    states: Tuple[torch.Tensor, ...],
    reset: Optional[torch.Tensor]
  ):
    """Forward GRU Cell

    Args:
      x (torch.Tensor): input tensor, expecting (seq, b, dim)
      states (Tuple[torch.Tensor, ...]): initial states
      reset (Optional[torch.Tensor]): sequence of reset signal to reset
        the input hidden/cell states, expecting (seq, b)

    Returns:
      torch.Tensor: sequence outputs
      Tuple[Tuple[torch.Tensor, ...], ...]: sequence of states
    """
    seq = x.shape[0]
    hx = states[0]
    # (seq, b) to (seq, b, 1)
    if len(reset.shape) < 3:
      reset = reset.unsqueeze(reset, -1)
    out = []
    states = []
    mask = 1.0 - reset
    for i in range(seq):
      hx = self.cell(x[i], hx * mask[i])
      out.append(hx)
      states.append((hx,))
    return torch.stack(out, dim=0), tuple(states)

  def get_states(self, batch_size=1, device='cuda'):
    return (
      torch.zeros((batch_size, self.units), device=device),
    )


@registry.register.kemono_net('rnn')
class AwesomeRnn(nn.Module):
  def __init__(
    self,
    dim: int = 1024,
    units: int = 1024,
    skip_conn: bool = True,
    rnn_type: str = 'lstm',
    return_history: bool = True
  ):
    super().__init__()
    self.units = units
    self.skip_conn = skip_conn
    self.rnn_type = rnn_type.lower()
    self.return_history = return_history
    cell_class = registry.get.kemono_cell(self.rnn_type)
    assert cell_class is not None, \
      f"rnn type does not exist: {self.rnn_type}"
    self.cell = cell_class(dim, units)

  def forward(
    self,
    x: torch.Tensor,
    states: Tuple[torch.Tensor, ...] = None,
    reset: torch.Tensor = None
  ):
    """Forward rnn module

    Args:
      x (torch.Tensor): input tensor, expecting (seq, b, dim), (b, dim)
      states (Tuple[torch.Tensor, ...]): [(batch, out_dim), ...]
      reset (torch.Tensor, optional): reset signal, expecting (seq, b) or
        (b,). Defaults to None.

    Returns:
      Tuple[torch.Tensor, Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        torch.Tensor: output tensor
        Tuple[torch.Tensor]: next states
        Tuple[torch.Tensor]: final states
    """
    # padding x from (b, dim) to (seq, b, dim)
    orig_shape = x.shape
    if len(x.shape) == 2:
      x = torch.unsqueeze(x, 0)
    seq, batch = x.shape[:2]
    # create states
    if states is None:
      states = self.get_states(batch, device=x.device)
    # create reset signal tensor
    if reset is None:
      reset = torch.tensor(0)
    reset = reset.type_as(x)
    seq, batch = x.shape[0:2]
    reset = torch.broadcast_to(reset, (seq, batch, 1))
    # out: sequence of rnn outputs (seq, b, dim)
    # new_states: sequence of hidden/cell states
    #   Tuple[Tuple[(b, dim), ...], ...]
    out, states_seq = self.cell(
      x,
      states = states,
      reset = reset
    )
    if self.skip_conn:
      out = out + x
    # unpad seq dim
    if len(orig_shape) < 3:
      out = torch.squeeze(out, 0)
    final_states = states_seq[-1]
    if self.return_history:
      _stack_op = lambda xs: torch.stack(xs, dim=0)
      history = rlchemy.utils.map_nested_tuple(states_seq, op=_stack_op)
      return out, final_states, history
    return out, final_states

  def get_states(self, batch_size=1, device='cuda'):
    return self.cell.get_states(batch_size=batch_size, device=device)


@registry.register.kemono_net('embed')
class AwesomeEmbed(nn.Module):
  def __init__(
    self,
    *args,
    num_embed: int,
    embed_dim: int = 128,
    **kwargs
  ):
    super().__init__()
    self._model = nn.Embedding(num_embed, embed_dim, **kwargs)
    self.output_dim = embed_dim
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x.to(dtype=torch.int64)
    return self._model(x)
