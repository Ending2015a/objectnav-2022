# --- built in ---
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
import einops
import rlchemy
from rlchemy.lib.nets import DelayedModule
from rlchemy import registry
# --- my module ---

def swish(x: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
  """Swish activation function
  from arXiv:1710.05941
  
  Args:
    x (torch.Tensor): input tensors can be either 2D or 4D.
    beta (torch.Tensor): swish-beta can be a constant or a trainable params.
  
  Returns:
    torch.Tensor: output tensors
  """
  if len(x.size()) == 2:
    return x * torch.sigmoid(beta[None, :] * x)
  else:
    return x * torch.sigmoid(beta[None, :, None, None] * x)

def get_activ_fn(
  activ_fn: Union[str, Callable],
  dim: int,
  **kwargs
) -> Callable:
  """Get activation function by name or function
  
  Args:
    activ_fn (Union[str, Callable]): activation function can be the name of the
      function or the function itself.
    dim (int): input dimensions.
  
  Returns:
    Callable: activation function
  """
  if isinstance(activ_fn, str):
    if registry.get.atlas_activ(activ_fn) is not None:
      # get activation function from registry
      activ_class = registry.get.atlas_activ(activ_fn)
      assert activ_class is not None
      activ_fn = activ_class(dim, **kwargs)
    else:
      # get activation function from pure pytorch module
      activ_class = getattr(nn, activ_fn, None)
      assert activ_class is not None
      activ_fn = activ_class(**kwargs)
  elif callable(activ_fn):
    return activ_fn
  else:
    raise ValueError(f"`activ_fn` must be a str or Callable, got {type(activ_fn)}")
  return activ_fn

@registry.register.atlas_activ('swish')
class Swish(DelayedModule):
  def __init__(self, dim: Optional[int] = None):
    super().__init__()
    if dim is not None:
      self.build(torch.Size([dim]))
  
  def build(self, input_shape: torch.Size):
    dim = input_shape[-1]
    self.beta = nn.Parameter(torch.ones(dim,))
    # ---
    self.mark_as_built()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return swish(x, self.beta)


class Activ(DelayedModule):
  def __init__(
    self,
    dim: Optional[int] = None,
    activ: Union[str, Callable] = 'SiLU',
    **kwargs
  ):
    super().__init__()
    self.activ = activ
    self.kwargs = kwargs
    # ---
    if dim is not None:
      self.build(torch.Size([dim]))

  def build(self, input_shape: torch.Size):
    dim = input_shape[-1]
    self.activ_fn = get_activ_fn(self.activ, dim, **self.kwargs)
    self.mark_as_built()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.activ_fn(x)


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
    activ: str = 'SiLU',
    final_activ: bool = False
  ):
    super().__init__()
    self.mlp_units = mlp_units
    self.activ = activ
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
          Activ(out_dim, self.activ, inplace=True)
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
    activ: str = 'SiLU',
    final_activ: bool = False
  ):
    super().__init__()
    self.mlp_units = mlp_units
    self.activ = activ
    self.final_activ = final_activ
    # ---
    self.input_shape = None
    self.input_dim = None
    self.output_dim = None
    self._model = None
    if shape is not None:
      self.build(torch.Size(shape))

  def build(self, input_shape: torch.Size):
    assert len(input_shape) >= 3
    self.input_shape = input_shape
    input_shape = input_shape[-3:]
    dim = input_shape[0]
    self.input_dim = dim
    cnn = nn.Sequential(
      nn.Conv2d(dim, 32, 8, 4, padding=0),
      Activ(32, self.activ, inplace=True),
      nn.Conv2d(32, 64, 4, 2, padding=0),
      Activ(64, self.activ, inplace=True),
      nn.Conv2d(64, 64, 3, 1, padding=0),
      Activ(64, self.activ, inplace=True),
      nn.Flatten(start_dim=-3, end_dim=-1)
    )
    # forward cnn to get output size
    dummy = torch.zeros((1, *input_shape), dtype=torch.float32)
    outputs = cnn(dummy).detach()
    # create mlp layers
    mlp = MLP(
      outputs.shape[-1],
      mlp_units = self.mlp_units,
      activ = self.activ,
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

@registry.register.atlas_net('embed')
class Embedding(nn.Module):
  def __init__(
    self,
    num_classes: int,
    embed_dim: int = 128,
    **kwargs
  ):
    super().__init__()
    self.num_classes = num_classes
    self.embed_dim = embed_dim
    self._model = nn.Embedding(num_classes, embed_dim, **kwargs)
    self.output_dim = embed_dim
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x.to(dtype=torch.int64)
    return self._model(x)


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
    # broadcast batch dims
    shape = torch.broadcast_shapes(x.shape[:-1], cond.shape[:-1])
    x = torch.broadcast_to(x, (*shape, x.shape[-1]))
    cond = torch.broadcast_to(cond, (*shape, cond.shape[-1]))
    # concat, forward
    x = self._vec_net(torch.cat((x, cond), dim=-1))
    # broadcast batch dims
    shape = torch.broadcast_shapes(x.shape[:-1], chart.shape[:-1])
    x = torch.broadcast_to(x, (*shape, x.shape[-1]))
    chart = torch.broadcast_to(chart, (*shape, chart.shape[-1]))
    # concat, forward
    return self._fuse_net(torch.cat((x, chart), dim=-1)), chart

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

