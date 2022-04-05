# --- built in ---
import os
import glob
from typing import (
  Any,
  Dict,
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
    """Habitat dataset loader
    This loader is expecting the following dataset sturcture
    root_path: /src/log/rednet_dataset/val/dataset
    rgb path: {root_path}/rgb/**/*.png
    depth path: {root_path}/depth/**/*.png
    seg path: {root_path}/seg/**/*.png

    Note that this dataset sets all void labels to index = 0 and other meaningful
    labels are labeld sequentially as listed in `valid_ids` from index = 1...n.

    Args:
      root_path (str): path to dataset root.
      img_size (tuple, optional): output image size. All samples will be resized to
        this image size. Defaults to (480, 640).
      void_ids (List[int], optional): list of void/unlabeled/unknown labels.
        Defaults to utils.mpcat40_trivial_ids.
      valid_ids (List[int], optional): list of meaningful labels.
        Defaults to utils.mpcat40_meaningful_ids.
      assert_num_classes (Optional[int], optional): ensure the number of categories
        match to our expectation. Defaults to 40.
      multi_scale_seg (List[int], optional): the size of samples are devided by
        this number. This is used to train a multi scale model. For example:
        image size (480, 640), scale = [2, 4, 8], then the dataset will generates
        ground truth segmentations in size: [(480, 640), (240, 320), (120, 160),
        (60, 80)]. Defaults to None.
      max_length (int, optional): max length of the dataset. Defaults to None.
      shuffle (bool, optional): shuffle dataset index. Defaults to False.
    """
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

  def __len__(self) -> int:
    return len(self.id_list)

  @torch.no_grad()
  def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sample_id = self.id_list[idx]
    rgb = self._get_rgb(sample_id)
    depth = self._get_depth(sample_id)
    seg = self._get_seg(sample_id)
    return rgb, depth, seg

  def get_rgb_path(self, idx: int) -> str:
    sample_id = self.id_list[idx]
    return self.rgb_list[sample_id]
  
  def get_root_path(self) -> str:
    return self.root_path

  def _get_rgb(self, sample_id: int) -> np.ndarray:
    """Read rgb data from file

    Args:
      sample_id (int): sample ID

    Returns:
      np.ndarray: RGB data in (c, h, w), np.float32
    """    
    # return (c, h, w), torch.float32
    rgb_path = os.path.join(self.root_path, self.rgb_list[sample_id])
    # load data
    rgb = cv2.imread(rgb_path)[...,::-1] # (h, w, c), bgr->rgb
    rgb = rgb.astype(np.float32) / 255.
    rgb = torch.from_numpy(rgb)
    rgb = torch.permute(rgb, (2, 0, 1)) # (c, h, w)
    rgb = utils.resize_image(
      rgb,
      size = self.img_size,
      mode = 'bilinear',
      align_corners = True
    )
    return rlchemy.utils.to_numpy(rgb)

  def _get_depth(self, sample_id: int) -> np.ndarray:
    """Read depth data from file

    Args:
      sample_id (int): sample ID

    Returns:
      np.ndarray: depth data in (1, h, w), np.float32
    """
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
    depth = utils.resize_image(depth, self.img_size, 'nearest')
    return rlchemy.utils.to_numpy(depth)

  def _get_seg(self, sample_id: int) -> Union[np.ndarray, Tuple[np.ndarray]]:
    """Read ground truth segmentation data

    Args:
      sample_id (int): sample ID

    Returns:
      Union[np.ndarray, Tuple[np.ndarray]]: ground truth segmentation
        * ground truth segmentation in (h, w), torch.int64
          if multi_scale_seg is None.
        * a tuple of ground truth segmentation with different scales,
          if multi_scale_seg is set.
    """    
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
    seg = utils.resize_image(seg, self.img_size, 'nearest')
    seg = seg.to(dtype=torch.int64)
    if self.multi_scale_seg is not None:
      # 0: original size
      # n: smaller size
      segs = [seg]
      for scale in self.multi_scale_seg:
        size = (self.img_size[0]//scale, self.img_size[1]//scale)
        seg = utils.resize_image(segs[0], size, 'nearest')
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
    """Preprocess layer

    Args:
      rgb_norm (dict, optional): arguments for RGB normalizer.
        See kemono.semantic.utils.ImageNormalize. Defaults to None.
      depth_norm (dict, optional): arguments for depth normalizer.
        See kemono.semantic.utils.ImageNormalize. Defaults to None.
      img_size (Tuple[int, int], optional): target image size.
        Defaults to None.
    """
    super().__init__()
    if rgb_norm is not None:
      rgb_norm = utils.ImageNormalize(**rgb_norm)
    if depth_norm is not None:
      depth_norm = utils.ImageNormalize(**depth_norm)
    self.rgb_norm = rgb_norm
    self.depth_norm = depth_norm
    self.img_size = img_size

  @torch.no_grad()
  def forward(
    self,
    rgb: torch.Tensor,
    depth: torch.Tensor
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward preprocessing
    Note that the input must be torch.Tensor

    Args:
      rgb (torch.Tensor): RGB image, expecting 3/4D image.
        torch.uint8 or torch.float32
      depth (torch.Tensor): Depth image, expecting 3/4D image
        torch.float32

    Returns:
      Tuple[torch.Tensor, torch.Tensor]: preorpcessed rgb and depth
    """
    # normalize uint8 -> float32
    if rgb.dtype == torch.uint8:
      rgb = rgb.to(dtype=torch.float32) / 255.
    rgb = rgb.to(dtype=torch.float32)
    depth = depth.to(dtype=torch.float32)
    # convert to 4D image
    rgb = utils.to_4D_tensor(rgb)
    depth = utils.to_4D_tensor(depth)
    # resize if needed
    if self.img_size is not None:
      rgb = utils.resize_image(rgb, self.img_size, 'bilinear', align_corners=True)
      depth = utils.resize_image(depth, self.img_size, 'nearest')
    rgb = self.rgb_norm(rgb)
    depth = self.depth_norm(depth)
    # ready for forwarding into semantic models
    return rgb, depth

class SemanticTask(pl.LightningModule):
  def __init__(
    self,
    config: Union[dict, omegaconf.Container],
    inference_only: bool = False
  ):
    """Semantic segmentation task wrapper

    Args:
        config (Union[dict, omegaconf.Container]): configurations.
        inference_only (bool, optional): inference mode only.
          Defaults to False.
    """
    super().__init__()
    config = OmegaConf.create(config)
    OmegaConf.resolve(config)
    self.save_hyperparameters("config")
    # ---
    self.config = config
    self.inference_only = inference_only
    self.track_iou_index = config.get('track_iou_index', None)
    self.preprocess: Preprocess = None
    self.model = None
    self.semantic_loss = None
    self.setup_model()
    if not self.inference_only:
      self.setup_loss()
      self.setup_dataset()

  def setup_model(self):
    """Setup segmentation model"""
    self.preprocess = Preprocess(**self.config.preprocess)
    model_class = registry.get.semantic_model(self.config.model_name)
    self.model = model_class(**self.config.model)

  def setup_loss(self):
    """Setup loss module"""
    loss_class = registry.get.semantic_loss(self.config.loss_name)
    self.semantic_loss = loss_class(**self.config.loss)

  def setup_dataset(self):
    """Setup datasets"""
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
    decay_rate = self.config.get('decay_rate', 0.8)
    decay_epochs = self.config.get('decay_epochs', 100)
    sche = torch.optim.lr_scheduler.LambdaLR(
      optim,
      lr_lambda = lambda ep: decay_rate ** (ep // decay_epochs)
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
  
  def forward(
    self,
    rgb: torch.Tensor,
    depth: torch.Tensor,
    **kwargs
  ) -> torch.Tensor:
    """Forward model

    Args:
      rgb (torch.Tensor): expecting 4D image tensor, torch.float32
      depth (torch.Tensor): expecting 4D image tensor, torch.float32

    Returns:
      torch.Tensor: predicted class index. (b, h, w), torch.int64
    """
    rgb = torch.as_tensor(rgb, device=self.device)
    depth = torch.as_tensor(depth, device=self.device)
    rgb, depth = self.preprocess(rgb, depth)
    return self.model(rgb, depth, **kwargs)

  @torch.no_grad()
  def predict(
    self,
    rgb: Union[np.ndarray, torch.Tensor],
    depth: Union[np.ndarray, torch.Tensor],
    **kwargs
  ) -> np.ndarray:
    """Predict one sample or batch of samples

    Args:
      rgb (Union[np.ndarray, torch.Tensor]): expecting 3/4D image
        tensor or np.ndarray, type can be uint8 or float32. The uint8
        image is normalized to float32
      depth (Union[np.ndarray, torch.Tensor]): expecting 3/4D image
        tensor or np.ndarray, type must be float32.

    Returns:
      np.ndarray: predicted class index.
    """
    orig_shape = rgb.shape
    one_sample = (len(orig_shape) == 3)
    with utils.evaluate(self):
      out = self(rgb, depth, **kwargs)
    out = torch.argmax(out, dim=1, keepdim=True)
    orig_size = orig_shape[-2:]
    out = utils.resize_image(out, orig_size, 'nearest')
    out = out.squeeze(dim=1) # (b, h, w)
    out = rlchemy.utils.to_numpy(out).astype(np.int64)
    if one_sample:
      out = out[0]
    return out

  def training_step(
    self,
    batch: Any,
    batch_idx: int
  ) -> torch.Tensor:
    """Lightning hook"""
    rgb, depth, segs = batch
    rgb = rgb.to(dtype=torch.float32)
    depth = depth.to(dtype=torch.float32)
    outs = self(rgb, depth, use_checkpoint=self.config.use_checkpoint)
    log_dict = {}
    if self.trainset.multi_scale_seg is not None:
      assert len(outs) == len(segs)
      losses = []
      # compute losses for multi-scale segmentations
      for index, (out, seg) in enumerate(zip(outs, segs)):
        loss = self.semantic_loss(out, seg)
        losses.append(loss)
        log_dict[f'train/loss_{index}'] = loss.item()
      total_loss = sum(losses)
    else:
      loss = self.semantic_loss(outs, segs)
      total_loss = loss
    # log on every step and epoch
    self.log(
      "train/loss",
      total_loss.item(),
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

  def validation_step(
    self,
    batch: Any,
    batch_idx: int
  ) -> torch.Tensor:
    """Lightning hook"""
    # in validation step, the self.training is set to False
    # (eval mode)
    rgb, depth, seg = batch
    # compute validation loss
    with utils.evaluate(self), torch.no_grad():
      rgb = rgb.to(dtype=torch.float32)
      depth = depth.to(dtype=torch.float32)
      out = self(rgb, depth, use_checkpoint=self.config.use_checkpoint)
      loss = self.semantic_loss(out, seg)
    out = torch.argmax(out, dim=1)
    log_dict = {
      "validation/val_loss": loss.item()
    }
    # compute class IoU
    if self.track_iou_index is not None:
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

  def _preview_predictions(self, pred_idx: int):
    """Plot one sample and save to disk"""
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
      with utils.evaluate(self):
        for n in range(num_samples):
          self._preview_predictions(n)