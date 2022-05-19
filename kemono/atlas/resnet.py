# --- built in ---
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
import torch.utils.model_zoo as model_zoo
from rlchemy import registry
from rlchemy.lib.nets import DelayedModule
# --- my module ---
from kemono.atlas import kemono_model

model_urls = {
  'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
  'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
  'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
  'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
  'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(
  in_planes: int,
  out_planes: int,
  stride: int = 1,
  groups: int = 1,
  dilation: int = 1
) -> nn.Conv2d:
  """3x3 convolution with padding"""
  return nn.Conv2d(
    in_planes,
    out_planes,
    kernel_size = 3,
    stride = stride,
    padding = dilation,
    groups = groups,
    bias = False,
    dilation = dilation,
  )

def conv1x1(
  in_planes: int,
  out_planes: int,
  stride: int = 1
) -> nn.Conv2d:
  """1x1 convolution"""
  return nn.Conv2d(
    in_planes,
    out_planes,
    kernel_size = 1,
    stride = stride,
    bias = False
  )


class BasicBlock(nn.Module):
  expansion: int = 1

  def __init__(
    self,
    inplanes: int,
    planes: int,
    stride: int = 1,
    downsample: Optional[nn.Module] = None,
    groups: int = 1,
    base_width: int = 64,
    dilation: int = 1,
    norm_layer: Optional[Callable[..., nn.Module]] = None,
    activ: str = 'SiLU',
  ) -> None:
    super().__init__()
    if norm_layer is None:
        norm_layer = nn.BatchNorm2d
    if groups != 1 or base_width != 64:
        raise ValueError("BasicBlock only supports groups=1 and base_width=64")
    if dilation > 1:
        raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
    # Both self.conv1 and self.downsample layers downsample the input when stride != 1
    self.conv1 = conv3x3(inplanes, planes, stride)
    self.bn1 = norm_layer(planes)
    self.activ_fn = kemono_model(planes, activ, inplace=True)
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = norm_layer(planes)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.activ_fn(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
        identity = self.downsample(x)

    out += identity
    out = self.activ_fn(out)

    return out


class ResNet(DelayedModule):
  def __init__(
    self,
    layers: List[int],
    name: str,
    shape: Tuple[int, ...] = None,
    mlp_units: List[int] = [1024],
    activ: str = 'SiLU',
    final_activ: bool = False,
    pretrained: bool = False
  ):
    super().__init__()
    self.layers = layers
    self.name = name
    self.mlp_units = mlp_units
    self.activ = activ
    self.final_activ = final_activ
    self.pretrained = pretrained
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
    self.output_dim = self.mlp_units[-1]
    # create resnet
    block = BasicBlock
    layers = self.layers
    norm_layer = nn.BatchNorm2d
    self._norm_layer = norm_layer
    self.inplanes = 64
    self.dilation = 1
    if replace_stride_with_dilation is None:
      # each element in the tuple indicates if we should replace
      # the 2x2 stride with a dilated convolution instead
      replace_stride_with_dilation = [False, False, False]
    if len(replace_stride_with_dilation) != 3:
      raise ValueError(
        "replace_stride_with_dilation should be None "
        f"or a 3-element tuple, got {replace_stride_with_dilation}"
      )
    self.groups = 1
    self.base_width = 64
    self.conv1 = nn.Conv2d(dim, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
    self.bn1 = norm_layer(self.inplanes)
    self.activ_fn = kemono_model.Activ(self.inplanes, self.activ, inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.layer1 = self._make_layer(block, 64, layers[0])
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
    self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.mlp = kemono_model.MLP(
      512 * block.expansion,
      mlp_units = self.mlp_units,
      activ = self.activ,
      final_activ = self.final_activ
    )

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(
          m.weight,
          mode = "fan_out",
          nonlinearity = "relu"
        )
      elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

    if self.pretrained:
      self._load_pretrained()
    self.mark_as_built()

  def _load_pretrained(self):
    pretrain_dict = model_zoo.load_url(model_urls[self.name])
    model_dict = {}
    state_dict = self.state_dict()
    for k, v in pretrain_dict.items():
      if k in state_dict:
        if k.startswith('conv1'):
          if self.input_dim == 3:
            model_dict[k] = v
          else:
            model_dict[k] = torch.mean(v, 1, keepdim=True).detach(). \
              repeat(1, self.input_dim, 1, 1)
    state_dict.update(model_dict)
    self.load_state_dict(state_dict)

  def _make_layer(
      self,
      block: BasicBlock,
      planes: int,
      blocks: int,
      stride: int = 1,
      dilate: bool = False,
  ) -> nn.Sequential:
    norm_layer = self._norm_layer
    downsample = None
    previous_dilation = self.dilation
    if dilate:
      self.dilation *= stride
      stride = 1
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        conv1x1(self.inplanes, planes * block.expansion, stride),
        norm_layer(planes * block.expansion),
      )

    layers = []
    layers.append(
      block(
        self.inplanes,
        planes,
        stride,
        downsample,
        self.groups,
        self.base_width,
        previous_dilation,
        norm_layer,
        self.activ
      )
    )
    self.inplanes = planes * block.expansion
    for _ in range(1, blocks):
      layers.append(
        block(
          self.inplanes,
          planes,
          groups=self.groups,
          base_width=self.base_width,
          dilation=self.dilation,
          norm_layer=norm_layer,
        )
      )

    return nn.Sequential(*layers)

  def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
    # See note [TorchScript super()]
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.activ_fn(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.mlp(x)

    return x

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    batches = x.shape[:-3]
    x = x.reshape(-1, *x.shape[-3:])
    x = self._forward_impl(x)
    x = x.reshape(*batches, x.shape[-1])
    return x


@registry.register.atlas_net('resnet18')
class ResNet18(ResNet):
  def __init__(
    self,
    shape: Tuple[int, ...] = None,
    **kwargs
  ):
    kwargs.pop('layers', None)
    super().__init__([2, 2, 2, 2], 'resnet18', shape, **kwargs)


