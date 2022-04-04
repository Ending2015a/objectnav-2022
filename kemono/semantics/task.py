# --- built in ---
import os
import glob
from typing import (
  List,
  Tuple,
  Union,
  Optional
)
# --- 3rd party ---
import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import omegaconf
from omegaconf import OmegaConf
import rlchemy
from rlchemy import registry
# --- my module ---
from kemono.semantics import utils
from kemono.semantics.semantic_mapping import CategoryMapping

RGB_GLOB_PATTERN = 'rgb/**/*.png'
DEPTH_GLOB_PATTERN = 'depth/**/*.png'
SEG_GLOB_PATTERN = 'seg/**/*.npy'

registry.register.optim('sgd')(torch.optim.SGD)
registry.register.optim('adam', default=True)(torch.optim.Adam)
registry.register.loss('ce', default=True)(nn.CrossEntropyLoss)
registry.register.loss('focal')(utils.FocalLoss)

def glob_filenames(
  root_path: str,
  pattern: str
) -> List[str]:
  """Find all filenames
  example:
    root_path: /src/log/rednet_dataset/val/dataset/
    pattern: rgb/**/*.png
    globbed path:
      /src/log/rednet_dataset/val/dataset/rgb/4ok3usBNeis/episode_id-1/000.png
    returned path:
      ./rgb/4ok3usBNeis/episode_id-1/000.png

  Args:
    root_path (str): root path to glob
    pattern (str): globbing pattern

  Returns:
    List[str]: list of globed file paths, relative to `root_path`
  """
  glob_path = os.path.join(root_path, pattern)
  # glob abspaths
  abspaths = glob.glob(glob_path, recursive=True)
  # convert to relpaths
  relpaths = [os.path.relpath(path, start=root_path) for path in abspaths]
  relpaths = sorted(relpaths)
  return relpaths

class HabitatDataset(Dataset):
  def __init__(
    self,
    root_path: str,
    img_size: tuple = (480, 640),
    void_ids: List[int] = utils.mpcat40_trivial_ids,
    valid_ids: List[int] = utils.mpcat40_meaningful_ids,
    assert_num_classes: Optional[int] = 40,
    multi_scale_seg: List[int] = None,
    max_length: Optional[int] = None,
    shuffle: bool = False,
  ):
    self.root_path = root_path
    self.img_size = img_size
    self.void_ids = void_ids
    self.valid_ids = valid_ids
    self.multi_scale_seg = multi_scale_seg
    self.max_length = max_length
    self.shuffle = shuffle
    self.rgb_list = glob_filenames(root_path, RGB_GLOB_PATTERN)
    self.depth_list = glob_filenames(root_path, DEPTH_GLOB_PATTERN)
    self.seg_list = glob_filenames(root_path, SEG_GLOB_PATTERN)
    assert len(self.rgb_list) == len(self.depth_list)
    assert len(self.rgb_list) == len(self.seg_list)
    self.id_list = np.arange(len(self.rgb_list))
    if shuffle:
      np.random.shuffle(self.id_list)
    if max_length is not None:
      self.id_list = self.id_list[:max_length]
    # meaningful class index start from 1
    self.class_map = dict(zip(valid_ids, range(1, 1+len(valid_ids))))
    # void class set to 0
    self.void_id = 0
    self.num_classes = len(valid_ids) + 1
    if assert_num_classes is not None:
      assert self.num_classes == assert_num_classes, \
        "The actual number of classes does not match, " \
        f"got {self.num_classes}, expect {assert_num_classes}"

  def __len__(self):
    return len(self.id_list)

  @torch.no_grad()
  def __getitem__(self, idx):
    sample_id = self.id_list[idx]
    rgb = self._get_rgb(sample_id)
    depth = self._get_depth(sample_id)
    seg = self._get_seg(sample_id)
    return rgb, depth, seg

  def get_rgb_path(self, idx):
    sample_id = self.id_list[idx]
    return self.rgb_list[sample_id]
  
  def get_root_path(self):
    return self.root_path

  def _get_rgb(self, sample_id):
    # return (c, h, w), torch.float32
    rgb_path = os.path.join(self.root_path, self.rgb_list[sample_id])
    # load data
    rgb = cv2.imread(rgb_path)[...,::-1] # (h, w, c), bgr->rgb
    rgb = rgb.astype(np.float32) / 255.
    rgb = torch.from_numpy(rgb)
    rgb = torch.permute(rgb, (2, 0, 1)) # (c, h, w)
    rgb = utils.resize_tensor(
      rgb,
      size = self.img_size,
      mode = 'bilinear',
      align_corners = True
    )
    return rlchemy.utils.to_numpy(rgb)

  def _get_depth(self, sample_id):
    # return (1, h, w), torch.float32
    depth_path = os.path.join(self.root_path, self.depth_list[sample_id])
    # (h, w), np.uint16 -> (1, h, w), torch.float32
    depth = cv2.imread(depth_path, -1) # (h, w), torch.uint16
    depth = depth.astype(np.float32) / 65535.
    depth = torch.from_numpy(depth)
    if len(depth.shape) == 2:
      depth = depth[None, :, :] # (1, h, w)
    else:
      depth = torch.permute(depth, (2, 0, 1)) # (c, h, w)
      depth = depth[:1, :, :] # (1, h, w)
    depth = utils.resize_tensor(depth, self.img_size, 'nearest')
    return rlchemy.utils.to_numpy(depth)

  def _get_seg(self, sample_id):
    # return (h, w), torch.int64
    seg_path = os.path.join(self.root_path, self.seg_list[sample_id])
    # (h, w), np.int32 -> torch.int64
    seg = np.load(seg_path)
    seg = torch.from_numpy(seg)
    # set void id
    for void_id in self.void_ids:
      seg[seg == void_id] = self.void_id
    # set meaningful id
    for valid_id in self.valid_ids:
      seg[seg == valid_id] = self.class_map[valid_id]
    seg[seg >= self.num_classes] = self.void_id
    # scaling
    seg = utils.resize_tensor(seg, self.img_size, 'nearest')
    seg = seg.to(dtype=torch.int64)
    if self.multi_scale_seg is not None:
      # 0: original size
      # n: smaller size
      segs = [seg]
      for scale in self.multi_scale_seg:
        size = (self.img_size[0]//scale, self.img_size[1]//scale)
        seg = utils.resize_tensor(segs[0], size, 'nearest')
        seg = seg.to(dtype=torch.int64)
        segs.append(seg)
      seg = tuple(segs)
      return rlchemy.utils.nested_to_numpy(seg)
    return rlchemy.utils.to_numpy(seg)

class Preprocess(nn.Module):
  def __init__(
    self,
    rgb_norm: Optional[dict] = None,
    depth_norm: Optional[dict] = None,
    img_size: Optional[Tuple[int, int]] = None,
  ):
    super().__init__()
    if rgb_norm is not None:
      rgb_norm = utils.Normalize(**rgb_norm)
    if depth_norm is not None:
      depth_norm = utils.Normalize(**depth_norm)
    self.rgb_norm = rgb_norm
    self.depth_norm = depth_norm
    self.img_size = img_size
  
  def forward(
    self,
    rgb: torch.Tensor,
    depth: torch.Tensor
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    # expecting
    # rgb (c, h, w) or (b, c, h, w), torch.uint8 or torch.float32
    # depth (1, h, w) or (b, 1, h, w), torch.float32
    rgb = utils.to_4D_tensor(rgb)
    depth = utils.to_4D_tensor(depth)
    if rgb.dtype == torch.uint8:
      rgb = rgb.to(dtype=torch.float32) / 255.
    rgb = rgb.to(dtype=torch.float32)
    depth = depth.to(dtype=torch.float32)
    if self.img_size is not None:
      # ensure size == img_size
      rgb = utils.resize_tensor(rgb, self.img_size, 'bilinear', align_corners=True)
      depth = utils.resize_tensor(depth, self.img_size, 'nearest')
    rgb = self.rgb_norm(rgb)
    depth = self.depth_norm(depth)
    # ready for forwarding into semantic models
    return rgb, depth

class SemanticTask(pl.LightningModule):
  def __init__(
    self,
    config: Union[dict, omegaconf.Container],
    inference_only: bool = False,
    track_iou_index: List[int] = [3, 10, 11, 14, 18, 22]
  ):
    super().__init__()
    config = OmegaConf.create(config)
    OmegaConf.resolve(config)
    self.save_hyperparameters("config")
    # ---
    self.config = config
    self.inference_only = inference_only
    self.track_iou_index = track_iou_index
    self.preprocess = None
    self.model = None
    self.weighted_cross_entropy = None
    self.setup_model()
    if not self.inference_only:
      self.setup_loss()
      self.setup_dataset()

  def setup_model(self):
    """Setup segmentation model"""
    self.preprocess = Preprocess(**self.config.preprocess)
    model_class = registry.get.semantic(self.config.model_name)
    self.model = model_class(**self.config.model)

  def setup_loss(self):
    loss_class = registry.get.loss(self.config.loss_name)
    self.segmentation_loss = loss_class(**self.config.loss)

  def setup_dataset(self):
    """Setup datasets (non-inference mode)"""
    self.trainset = HabitatDataset(**self.config.trainset)
    self.valset = HabitatDataset(**self.config.valset)
    self.predset = HabitatDataset(**self.config.predset)

  def configure_optimizers(self):
    optim_class = registry.get.optim(self.config.optimizer_name)
    optim = optim_class(
      self.model.parameters(),
      **self.config.optimizer
    )
    #sche = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=10)
    sche = torch.optim.lr_scheduler.LambdaLR(
      optim,
      lr_lambda = lambda ep: 0.8 ** (ep // 100)
    )
    return [optim], [sche]

  def train_dataloader(self) -> DataLoader:
    return DataLoader(
      self.trainset,
      batch_size = self.config.batch_size,
      num_workers = self.config.num_workers,
      shuffle = True
    )
  
  def val_dataloader(self) -> DataLoader:
    return DataLoader(
      self.valset,
      batch_size = self.config.batch_size,
      num_workers = self.config.num_workers,
      shuffle = False
    )
  
  def forward(self, rgb, depth, **model_kwargs):
    rgb, depth = self.preprocess(rgb, depth)
    return self.model(rgb, depth, **model_kwargs)

  @torch.no_grad()
  def predict(self, rgb, depth):
    orig_shape = rgb.shape
    one_sample = (len(orig_shape) == 3)
    rgb = rlchemy.utils.to_tensor(rgb, device=self.device)
    depth = rlchemy.utils.to_tensor(depth, device=self.device)
    out = self(rgb, depth, training=False)
    out = torch.argmax(out, dim=1, keepdim=True)
    orig_size = orig_shape[-2:]
    out = utils.resize_tensor(out, orig_size, 'nearest')
    out = out.squeeze(dim=1) # (b, h, w)
    out = rlchemy.utils.to_numpy(out).astype(np.int64)
    if one_sample:
      out = out[0]
    return out

  def training_step(self, batch, batch_idx):
    rgb, depth, segs = batch
    rgb = rgb.to(dtype=torch.float32)
    depth = depth.to(dtype=torch.float32)
    outs = self(rgb, depth)
    log_dict = {}
    if self.trainset.multi_scale_seg is not None:
      assert len(outs) == len(segs)
      losses = []
      # compute losses for multi-scale segmentations
      for index, (out, seg) in enumerate(zip(outs, segs)):
        loss = self.segmentation_loss(out, seg)
        losses.append(loss)
        log_dict[f'train/loss_{index}'] = loss
      total_loss = sum(losses)
    else:
      loss = self.segmentation_loss(outs, segs)
      total_loss = loss
    # log on every step and epoch
    self.log(
      "train/loss",
      total_loss,
      on_step = True,
      on_epoch = True,
      sync_dist = True,
      prog_bar = True
    )
    # log on every step and epoch
    self.log_dict(
      log_dict,
      on_step = True,
      on_epoch = True,
      sync_dist = True
    )
    return total_loss

  def validation_step(self, batch, batch_idx):
    # in validation step, the self.training is set to False
    # (eval mode)
    rgb, depth, seg = batch
    with torch.no_grad():
      rgb = rgb.to(dtype=torch.float32)
      depth = depth.to(dtype=torch.float32)
      out = self(rgb, depth) # only one outputs
      loss = self.segmentation_loss(out, seg)
    out = torch.argmax(out, dim=1)
    log_dict = {
      "validation/val_loss": loss
    }
    track_iou = []
    for index in self.track_iou_index:
      iou = utils.classIoU(out, seg, index)
      track_iou.append(iou)
      log_dict[f"validation/IoU_{index}"] = utils.none_for_nan(iou)
    self.log_dict(
      log_dict,
      on_epoch = True,
      sync_dist = True
    )
    self.log(
      "validation/track_mIoU",
      utils.none_for_nan(np.nanmean(track_iou)),
      on_epoch = True,
      sync_dist = True
    )
    # log main score
    miou = utils.mIoU(out, seg, self.config.num_classes)
    self.log(
      "validation/mIoU",
      miou,
      on_epoch = True,
      sync_dist = True,
      prog_bar = True
    )
    return loss

  def _preview_predictions(self, pred_idx):
    rgb, depth, seg_gt = self.predset[pred_idx]
    seg_pred = self.predict(rgb, depth)
    mapping = CategoryMapping()
    # expecting rgb: (c, h, w), np.float32
    rgb = np.transpose(rgb * 255., (1, 2, 0))
    rgb = rgb.astype(np.uint8)
    # expecting depth: (1, h, w), np.float32
    depth = np.transpose(depth * 255., (1, 2, 0))
    depth = depth.astype(np.uint8)
    depth = np.concatenate((depth,) * 3, axis=-1) # (h, w, 3)
    # expecting seg_gt: (h, w), np.int64
    seg_gt = mapping.get_colorized_mpcat40_category_map(seg_gt)
    # expecting seg_pred: (h, w), np.int64
    seg_pred = mapping.get_colorized_mpcat40_category_map(seg_pred)
    # concat scene
    scene = np.concatenate((rgb, depth, seg_pred, seg_gt), axis=1)
    scene = scene[...,::-1] # rgb -> bgr
    # make save path
    rgb_path = self.predset.get_rgb_path(pred_idx)
    path = os.path.relpath(rgb_path, start='rgb/')
    path = os.path.join(
      self.logger.log_dir,
      f'predictions/epoch_{self.current_epoch}',
      path
    )
    # make directories
    rlchemy.utils.safe_makedirs(filepath=path)
    cv2.imwrite(path, scene)

  def on_save_checkpoint(self, checkpoint):
    if self.trainer.is_global_zero:
      num_samples = len(self.predset)
      training = self.training
      if training:
        self.eval()
      for n in range(num_samples):
        self._preview_predictions(n)
      if training:
        self.train()