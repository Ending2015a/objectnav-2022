# --- bulit in ---
import os
from typing import Dict, List, Union, Tuple
import dataclasses
# --- 3rd party ---
import einops
import numpy as np
import pandas as pd
import torch
from torch import nn
import dungeon_maps as dmap
# --- my module ---

__all__ = [
  'MPCat40Category',
  'HM3DCategory',
  'mpcat40categories',
  'hm3dcategories',
  'mp3d_category_map',
  'hm3d_manual_map',
  'mpcat40_color_map_rgb',
  'mpcat40_meaningful_ids',
  'mpcat40_trivial_ids',
  'Normalize',
  'resize_tensor',
  'classIoU',
  'mIoU'
]

# === Loading mp3d hm3d semantic labels ===
LIB_PATH = os.path.dirname(os.path.abspath(__file__))
# See https://github.com/niessner/Matterport/blob/master/metadata/mpcat40.tsv
MPCAT40_PATH = os.path.join(LIB_PATH, 'mpcat40.tsv')
CATEGORY_MAPPING_PATH = os.path.join(LIB_PATH, 'category_mapping.tsv')
HM3D_MANUAL_MAPPING_PATH = os.path.join(LIB_PATH, 'manual_mapping.csv')
MPCAT40_DF = pd.read_csv(MPCAT40_PATH, sep='\t')
CATEGORY_MAPPING_DF = pd.read_csv(CATEGORY_MAPPING_PATH, sep='\t')
HM3D_MANUAL_MAPPING_DF = pd.read_csv(HM3D_MANUAL_MAPPING_PATH)

MPCAT40_TRIVIAL_LABELS = [
  'void',
  'misc',
  'unlabeled'
]

def dataclass_factory(df, name):
  _class = dataclasses.make_dataclass(
    name,
    [
      (column_name, column_dtype)
      for column_name, column_dtype in
      zip(df.dtypes.index, df.dtypes.values)
    ]
  )
  cat_insts = []
  for index, row in df.iterrows():
    cat_inst = _class(**{
      column: row[column]
      for column in df.columns
    })
    cat_insts.append(cat_inst)
  return _class, cat_insts

MPCat40Category, mpcat40categories = \
  dataclass_factory(MPCAT40_DF, 'MPCat40Category')
HM3DCategory, hm3dcategories = \
  dataclass_factory(CATEGORY_MAPPING_DF, 'HM3DCategory')

hex2rgb = lambda hex: [int(hex[i:i+2], 16) for i in (1, 3, 5)]
hex2bgr = lambda hex: hex2rgb(hex)[::-1]


def get_mp3d_category_map() -> Dict[str, MPCat40Category]:
  # mapping by either category name
  # default: category name
  mpcat40_name_to_category = {
    mpcat40cat.mpcat40: mpcat40cat
    for mpcat40cat in mpcat40categories
  }
  hm3d_name_to_mpcat40 = {}
  for hm3dcat in hm3dcategories:
    hm3d_name_to_mpcat40[hm3dcat.category] = \
      mpcat40_name_to_category[hm3dcat.mpcat40]
    hm3d_name_to_mpcat40[hm3dcat.raw_category] = \
      mpcat40_name_to_category[hm3dcat.mpcat40]
  return hm3d_name_to_mpcat40

mp3d_category_map = get_mp3d_category_map()

def get_mpcat40_color_map(bgr: bool=False) -> np.ndarray:
  colors = [
    hex2rgb(mpcat40cat.hex) if not bgr else hex2bgr(mpcat40cat.hex)
    for mpcat40cat in mpcat40categories
  ]
  colors = np.asarray(colors, dtype=np.uint8)
  return colors

mpcat40_color_map_rgb = get_mpcat40_color_map(bgr=False)

def get_hm3d_manual_mapping() -> Dict[str, str]:
  hm3d_map = HM3D_MANUAL_MAPPING_DF.to_dict()
  sources = hm3d_map['source'].values()
  targets = hm3d_map['target'].values()
  hm3d_map = {}
  for src, tar in zip(sources, targets):
    hm3d_map[src.strip()] = tar.strip()
  return hm3d_map

hm3d_manual_map = get_hm3d_manual_mapping()

def get_mpcat40_label_lists(
  trivial_lists: List[str] = MPCAT40_TRIVIAL_LABELS
):
  meaningful_ids = []
  trivial_ids = []
  for idx, cat in enumerate(mpcat40categories):
    if cat.mpcat40 in trivial_lists:
      trivial_ids.append(idx)
    else:
      meaningful_ids.append(idx)
  return meaningful_ids, trivial_ids

mpcat40_meaningful_ids, mpcat40_trivial_ids = get_mpcat40_label_lists()
# totally we have 40 classes: 39 meaningful classes + 1 trivial class


def to_4D_tensor(t: torch.Tensor) -> torch.Tensor:
  t = dmap.utils.to_tensor(t)
  if len(t.shape) == 0:
    # () -> (b, c, h, w)
    t = torch.broadcast_to(t, (1,1,1,1))
  elif len(t.shape) == 1:
    # (c,) -> (b, c, h, w)
    t = t[None, :, None, None]
  elif len(t.shape) == 2:
    # (h, w) -> (b, c, h, w)
    t = t[None, None, :, :]
  elif len(t.shape) == 3:
    # (c, h, w) -> (b, c, h, w)
    t = t[None, :, :, :]
  return t

# === Tools ===
class Normalize(nn.Module):
  def __init__(
    self,
    mean: Union[np.ndarray, torch.Tensor],
    std: Union[np.ndarray, torch.Tensor],
  ):
    super().__init__()
    mean = to_4D_tensor(mean)
    std = to_4D_tensor(std)
    self.register_buffer('mean', mean, persistent=False)
    self.register_buffer('std', std, persistent=False)

  def forward(self, x):
    return (x - self.mean) / self.std

def resize_tensor(
  tensor_image: Union[np.ndarray, torch.Tensor],
  size: Tuple[int, int],
  mode: str = 'nearest',
  **kwargs
) -> torch.Tensor:
  """Resize image tensor

  Args:
      tensor_image (Union[np.ndarray, torch.Tensor]): expecting
        (h, w), (c, h, w), (b, c, h, w)
      size (Tuple[int, int]): target size in (h, w)
      mode (str, optional): resize mode. Defaults to 'nearest'.
  """
  t = dmap.utils.to_tensor(tensor_image)
  orig_shape = t.shape
  orig_dtype = t.dtype
  orig_ndims = len(orig_shape)
  t = dmap.utils.to_4D_image(t) # (b, c, h, w)
  t = t.to(dtype=torch.float32)
  t = nn.functional.interpolate(t, size=tuple(size), mode=mode, **kwargs)
  t = dmap.utils.from_4D_image(t, orig_ndims)
  return t.to(dtype=orig_dtype)


@torch.no_grad()
def classIoU(
  seg_pred: torch.Tensor,
  seg_gt: torch.Tensor,
  class_idx: int,
  eps: float = 1e-8
):
  true_class = seg_pred == class_idx
  true_label = seg_gt == class_idx
  if true_label.long().sum().item() == 0:
    return np.nan
  else:
    intersect = torch.logical_and(true_class, true_label)
    union = torch.logical_or(true_class, true_label)
    intersect = intersect.sum().float().item()
    union = union.sum().float().item()
    iou = (intersect + eps) / (union + eps)
    return iou

@torch.no_grad()
def mIoU(
  seg_pred: torch.Tensor,
  seg_gt: torch.Tensor,
  num_classes: int = 40,
  eps: float = 1e-8
) -> float:
  """calculate mIoU

  Args:
    seg_pred (torch.Tensor): predicted category indices
    seg_gt (torch.Tensor): ground truth category indices
    eps (float, optional): epsilon. Defaults to 1e-8.
    num_classes (int, optional): number of classes. Defaults to 40.

  Returns:
    float: mIoU
  """  
  # flatten tensors
  iou_per_class = []
  for idx in range(num_classes):
    class_iou = classIoU(seg_pred, seg_gt, idx, eps=eps)
    iou_per_class.append(class_iou)
  return np.nanmean(iou_per_class)


class FocalLoss(nn.Module):
  def __init__(
    self,
    weight = None,
    gamma = 2,
    reduction = 'mean'
  ):
    super().__init__()
    self.weight = weight
    self.gamma = gamma
    self.reduction = reduction
  
  def forward(self, input, target):
    log_prob = nn.functional.log_softmax(input, dim=1)
    prob = torch.exp(log_prob)
    return nn.functional.nll_loss(
      ((1-prob) ** self.gamma) * log_prob,
      target,
      weight = self.weight,
      reduction = self.reduction
    )

def none_for_nan(value):
  if np.any(value == np.nan):
    return None
  return value
