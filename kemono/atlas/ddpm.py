# --- built in ---
import functools
from typing import (
  Any,
  List,
  Tuple,
  Union,
  Callable,
  Optional
)
# --- 3rd party ---
import numpy as np
import torch
from torch import nn
from rlchemy import registry
from rlchemy.lib.nets import DelayedModule
import einops
# --- my module ---
from kemono.atlas import model as kemono_model


class Residual(nn.Module):
  def __init__(self, fn):
    super().__init__()
    self.fn = fn

  def forward(self, x, *args, **kwargs):
    return self.fn(x, *args, **kwargs) + x

def Downsample(dim):
  return nn.Conv2d(dim, dim, 4, 2, 1)


class LayerNorm(nn.Module):
  def __init__(self, dim, eps = 1e-5):
    super().__init__()
    self.eps = eps
    self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
    self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

  def forward(self, x):
    var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
    mean = torch.mean(x, dim = 1, keepdim = True)
    return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class PreNorm(nn.Module):
  def __init__(self, dim, fn):
    super().__init__()
    self.fn = fn
    self.norm = LayerNorm(dim)

  def forward(self, x):
    x = self.norm(x)
    return self.fn(x)


class Block(nn.Module):
  def __init__(self, dim, dim_out, groups = 8):
    super().__init__()
    self.block = nn.Sequential(
        nn.Conv2d(dim, dim_out, 3, padding = 1),
        nn.GroupNorm(groups, dim_out),
        nn.SiLU()
    )

  def forward(self, x):
    return self.block(x)

class ResnetBlock(nn.Module):
  def __init__(self, dim, dim_out, *, groups = 8):
    super().__init__()

    self.block1 = Block(dim, dim_out, groups = groups)
    self.block2 = Block(dim_out, dim_out, groups = groups)
    self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

  def forward(self, x):
    h = self.block1(x)
    h = self.block2(h)
    return h + self.res_conv(x)

class LinearAttention(nn.Module):
  def __init__(self, dim, heads = 4, dim_head = 32):
    super().__init__()
    self.scale = dim_head ** -0.5
    self.heads = heads
    hidden_dim = dim_head * heads
    self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

    self.to_out = nn.Sequential(
      nn.Conv2d(hidden_dim, dim, 1),
      LayerNorm(dim)
    )

  def forward(self, x):
    b, c, h, w = x.shape
    qkv = self.to_qkv(x).chunk(3, dim = 1)
    q, k, v = map(lambda t: einops.rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

    q = q.softmax(dim = -2)
    k = k.softmax(dim = -1)

    q = q * self.scale
    context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

    out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
    out = einops.rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
    return self.to_out(out)

class Attention(nn.Module):
  def __init__(self, dim, heads = 4, dim_head = 32):
    super().__init__()
    self.scale = dim_head ** -0.5
    self.heads = heads
    hidden_dim = dim_head * heads
    self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
    self.to_out = nn.Conv2d(hidden_dim, dim, 1)

  def forward(self, x):
    b, c, h, w = x.shape
    qkv = self.to_qkv(x).chunk(3, dim = 1)
    q, k, v = map(lambda t: einops.rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)
    q = q * self.scale

    sim = torch.einsum('b h d i, b h d j -> b h i j', q, k)
    sim = sim - sim.amax(dim = -1, keepdim = True).detach()
    attn = sim.softmax(dim = -1)

    out = torch.einsum('b h i j, b h d j -> b h i d', attn, v)
    out = einops.rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
    return self.to_out(out)

class DDPM_Encoder(nn.Module):
  def __init__(
    self,
    channels,
    dim_mults = (1, 2, 4, 8),
    resnet_block_groups = 8,
  ):
    """
    https://github.com/lucidrains/denoising-diffusion-pytorch
    """
    super().__init__()

    # determine dimensions

    self.channels = channels

    dim = 64
    init_dim = dim//3*2
    self.init_conv = nn.Conv2d(channels, init_dim, 7, padding = 3)

    dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
    in_out = list(zip(dims[:-1], dims[1:]))

    block_klass = functools.partial(ResnetBlock, groups = resnet_block_groups)

    # layers
    self.downs = nn.ModuleList([])
    num_resolutions = len(in_out)

    for ind, (dim_in, dim_out) in enumerate(in_out):
      is_last = ind >= (num_resolutions - 1)

      self.downs.append(nn.ModuleList([
        block_klass(dim_in, dim_out),
        block_klass(dim_out, dim_out),
        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
        Downsample(dim_out) if not is_last else nn.Identity()
      ]))

    mid_dim = dims[-1]
    self.mid_block1 = block_klass(mid_dim, mid_dim)
    self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
    self.mid_down = Downsample(mid_dim)
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


  def forward(self, x):
    x = self.init_conv(x)

    for block1, block2, attn, downsample in self.downs:
      x = block1(x)
      x = block2(x)
      x = attn(x)
      x = downsample(x)

    x = self.mid_block1(x)
    x = self.mid_attn(x)
    x = self.mid_down(x)
    x = self.avgpool(x)

    return x

@registry.register.atlas_net('ddpm')
class DDPM(DelayedModule):
  def __init__(
    self,
    shape: Tuple[int, ...] = None,
    mlp_units: List[int] = [1024],
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
    self.input_shpae = input_shape
    input_shape = input_shape[-3:]
    dim = input_shape[0]
    encoder = DDPM_Encoder(dim)
    # forard cnn to get output size
    dummy = torch.zeros((1, *input_shape), dtype=torch.float32)
    outputs = encoder(dummy).detach()
    outputs = torch.flatten(outputs, -3, -1)
    # create mlp layers
    mlp = kemono_model.MLP(
      outputs.shape[-1],
      mlp_units = self.mlp_units,
      activ = self.activ,
      final_activ = self.final_activ
    )
    self._model = nn.Sequential(
      encoder,
      nn.Flatten(start_dim=-3, end_dim=-1),
      mlp
    )
    self.output_dim = mlp.output_dim
    self.mark_as_built()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    batches = x.shape[:-3]
    x = x.reshape(-1, *x.shape[-3:])
    x = self._model(x)
    x = x.reshape(*batches, x.shape[-1])
    return x
