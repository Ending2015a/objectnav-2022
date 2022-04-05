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
  'SemanticPredictor'
]

@registry.register.semantic_predictor('model')
class SemanticPredictor():
  def __init__(
    self,
    restore_path: str,
    device: torch.device = 'cuda'
  ):
    self.model = SemanticTask.load_from_checkpoint(restore_path)
    self.model = self.model.to(device=device).eval()

  def reset(self):
    pass

  def predict(self, obs):
    assert 'rgb' in obs
    assert 'depth' in obs
    assert len(obs['rgb']) == 3, f"Expecting (h, w, c), got {obs['rgb'].shape}"
    assert len(obs['depth']) == 3, f"Expecting (h, w, c), got {obs['depth'].shape}"
    rgb = np.transpose(obs['rgb'], (2, 0, 1))
    depth = np.transpose(obs['depth'], (2, 0, 1))
    return self.model.predict(rgb, depth)
