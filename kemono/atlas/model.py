# --- built in ---
import copy
from typing import (
  Any,
  List,
  Tuple,
  Union,
  Optional
)
# --- 3rd party ---
import torch
from torch import nn
import einops
import rlchemy
from rlchemy.lib.nets import DelayedModule
from rlchemy import registry
# --- my module ---

class Swish(nn.Module):
  def __init__(self, dim=-1):
    super().__init__()
    if dim > 0:
      self.beta = nn.Parameter(torch.ones((dim,)))
    else:
      self.beta = torch.ones((1,))

  def forward(self, x):
    if len(x.size()) == 2:
      return x * torch.sigmoid(self.beta[None, :] * x)
    else:
      return x * torch.sigmoid(self.beta[None, :, None, None] * x)


@registry.register.atlas_net('identity')
class Identity(DelayedModule):
  def __init__(
    self,
    dim: Optional[int] = None
  ):
    super().__init__()
    # ---
    self.input_dim = None
    self.output_dim = None
    if dim is not None:
      if isinstance(dim, int):
        dim = [dim]
      self.build(torch.Size(dim))
  
  def build(self, input_shape: torch.Size):
    self.input_dim = input_shape[-1]
    self.output_dim = input_shape[-1]
    self.mark_as_built()

  def forward(self, x):
    return x


@registry.register.atlas_net('mlp', default=True)
class MLP(DelayedModule):
  def __init__(
    self,
    dim: Optional[int] = None,
    mlp_units: List[int] = [64, 64],
    use_swish: bool = True,
    final_activ: bool = False
  ):
    super().__init__()
    self.mlp_units = mlp_units
    self.use_swish = use_swish
    self.final_activ = final_activ
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
    for idx, out_dim in enumerate(self.mlp_units):
      layers.append(nn.Linear(in_dim, out_dim))
      if idx+1 != len(self.mlp_units) or self.final_activ:
        layers.append(
          Swish(out_dim) if self.use_swish
          else nn.LeakyReLU(0.01, inplace=True)
        )
      in_dim = out_dim
    self._model = nn.Sequential(*layers)
    self.output_dim = out_dim
    self.mark_as_built()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    batches = x.shape[:-1]
    x = x.reshape(-1, x.shape[-1])
    x = self._model(x)
    x = x.reshape(*batches, x.shape[-1])
    return x


@registry.register.atlas_net('nature_cnn')
class NatureCnn(DelayedModule):
  def __init__(
    self,
    shape: Tuple[int, ...] = None,
    mlp_units: List[int] = [512],
    use_swish: bool = True,
    final_activ: bool = False
  ):
    super().__init__()
    self.mlp_units = mlp_units
    self.use_swish = use_swish
    self.final_activ = final_activ
    # ---
    self.input_shape = None
    self.input_dim = None
    self.output_dim = None
    self._model = None
    if shape is not None:
      self.build(torch.Size(shape))

  def _activ(self, dim):
    return (Swish(dim) if self.use_swish
      else nn.LeakyReLU(0.01, inplace=True))

  def build(self, input_shape: torch.Size):
    assert len(input_shape) >= 3
    self.input_shape = input_shape
    input_shape = input_shape[-3:]
    dim = input_shape[0]
    self.input_dim = dim
    cnn = nn.Sequential(
      nn.Conv2d(dim, 32, 8, 4, padding=0),
      self._activ(32),
      nn.Conv2d(32, 64, 4, 2, padding=0),
      self._activ(64),
      nn.Conv2d(64, 64, 3, 1, padding=0),
      self._activ(64),
      nn.Flatten(start_dim=-3, end_dim=-1)
    )
    # forward cnn to get output size
    dummy = torch.zeros((1, *input_shape), dtype=torch.float32)
    outputs = cnn(dummy).detach()
    # create mlp layers
    mlp = MLP(
      outputs.shape[-1],
      mlp_units = self.mlp_units,
      use_swish = self.use_swish,
      final_activ = self.final_activ
    )
    self._model = nn.Sequential(*cnn, mlp)
    self.output_dim = mlp.output_dim
    self.mark_as_built()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    batches = x.shape[:-3]
    x = x.reshape(-1, *x.shape[-3:])
    x = self._model(x)
    x = x.reshape(*batches, x.shape[-1])
    return x


@registry.register.atlas_net('one_hot')
class OneHot(nn.Module):
  def __init__(
    self,
    num_classes: int
  ):
    super().__init__()
    self.num_classes = num_classes
    self.output_dim = num_classes

  def forward(self, x):
    x = x.to(dtype=torch.int64)
    x = nn.functional.one_hot(x, self.num_classes)
    x = x.to(dtype=torch.float32)
    return x

class ToyNet(nn.Module):
  def __init__(
    self,
    x_config,
    cond_config,
    vec_config,
    chart_config,
    fuse_config
  ):
    super().__init__()
    self.x_config = copy.deepcopy(x_config)
    self.cond_config = copy.deepcopy(cond_config)
    self.vec_config = copy.deepcopy(vec_config)
    self.chart_config = copy.deepcopy(chart_config)
    self.fuse_config = copy.deepcopy(fuse_config)

    self._x_net = self._instantiate(x_config)
    self._cond_net = self._instantiate(cond_config)
    self._vec_net = self._instantiate(vec_config)
    self._chart_net = self._instantiate(chart_config)
    self._fuse_net = self._instantiate(fuse_config)

  def _instantiate(self, config):
    net_class = registry.get.atlas_net(config._type_)
    config.pop('_type_')
    net = net_class(**config)
    return net

  def forward(
    self,
    x: torch.Tensor,
    cond: torch.Tensor,
    chart: torch.Tensor,
    chart_cache: Optional[torch.Tensor] = None
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    x = self._x_net(x)
    cond = self._cond_net(cond)
    if chart_cache is not None:
      chart = chart_cache
    else:
      chart = self._chart_net(chart)
    x = torch.cat((x, cond), dim=-1)
    x = self._vec_net(x)
    x = torch.cat((x, chart), dim=-1)
    return self._fuse_net(x), chart

class Energy(nn.Module):
  def __init__(self, net: ToyNet):
    super().__init__()
    self.net = net
  
  def forward(
    self,
    x: torch.Tensor,
    cond: torch.Tensor,
    chart: torch.Tensor,
    chart_cache: Optional[torch.Tensor] = None,
    get_score: bool = False
  ):
    x = x.to(dtype=torch.float32)
    if get_score:
      x = x.requires_grad_()
    e, cache = self.net(x, cond, chart, chart_cache=chart_cache)
    if get_score:
      logp = -e.sum()
      s = torch.autograd.grad(logp, x, create_graph=True)[0]
      return e, s, cache.detach()
    return e, cache.detach()

