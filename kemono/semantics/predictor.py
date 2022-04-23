# --- built in ---
from typing import (
  Union
)
# --- 3rd party ---
import numpy as np
import torch
from torch import nn
import rlchemy
from rlchemy import registry
# --- my module ---
from kemono.semantics.task import SemanticTask

__all__ = [
  'SemanticPredictor',
  'RedNetPredictor',
  'ACNetPredictor'
]


class BaseSemanticPredictor():
  def __init__(
    self,
  ):
    pass

  def assert_input(self, obs):
    assert 'rgb' in obs
    assert 'depth' in obs
    assert len(obs['rgb'].shape) == 3, \
      f"Expecting (h, w, c), got {obs['rgb'].shape}"
    assert len(obs['depth'].shape) == 3, \
      f"Expecting (h, w, c), got {obs['depth'].shape}"

  def reset(self, goal_id: int=None):
    pass

  def predict(self, obs: torch.Tensor):
    self.assert_input(obs)


@registry.register.semantic_predictor('rednet')
class RedNetPredictor(BaseSemanticPredictor):
  def __init__(
    self,
    restore_path: str,
    goal_logits_scale: float = 1.0,
    device: torch.device = 'cuda'
  ):
    """SemanticPredictor

    Args:
      restore_path (str): path to pretrained weights
      goal_logits_scale (float, optional): used to prevent from
        over-confidence to the goal category. Defaults to 1.0.
      device (torch.device, optional): torch device. Defaults to 'cuda'.
    """
    self.goal_logits_scale = goal_logits_scale
    self.model = SemanticTask.load_from_checkpoint(
      restore_path,
      map_location=torch.device(device),
      override_params = ['model.pretrained=false']
    )
    self.model = self.model.eval()
    self.num_classes = self.model.num_classes
    self._logits_scale = None

  def reset(self, goal_id=None):
    self._logits_scale = np.ones((self.num_classes,), dtype=np.float32)
    if goal_id is not None:
      self._logits_scale[goal_id] = self.goal_logits_scale

  def predict(self, obs):
    super().assert_input(obs)
    rgb = np.transpose(obs['rgb'], (2, 0, 1))
    depth = np.transpose(obs['depth'], (2, 0, 1))
    seg = self.model.predict(
      rgb,
      depth,
      logits_scale = self._logits_scale
    )
    return seg

@registry.register.semantic_predictor('acnet')
class ACNetPredictor(RedNetPredictor):
  pass

# @registry.register.semantic_predictor('crf')
# class CRFModelPredictor(RedNetPredictor):
#   def __init__(
#     self,
#     restore_path: str,
#     goal_logits_scale: float = 1.0,
#     device: torch.device = 'cuda'
#   ):
#     """SemanticPredictor

#     Args:
#       restore_path (str): path to pretrained weights
#       goal_logits_scale (float, optional): used to prevent from
#         over-confidence to the goal category. Defaults to 1.0.
#       device (torch.device, optional): torch device. Defaults to 'cuda'.
#     """
#     self.goal_logits_scale = goal_logits_scale
#     self.model = SemanticTask.load_from_checkpoint(
#       restore_path,
#       map_location=torch.device(device),
#       override_params = [
#         'model.base_model_kwargs.pretrained=false',
#         'model.inference_only=true'
#       ]
#     )
#     self.model = self.model.eval()
#     self.num_classes = self.model.num_classes
#     self._logits_scale = None
