# --- built in ---
from typing import (
  Optional
)
# --- 3rd party ---
import torch
from torch import nn
from rlchemy import registry

# --- my module ---

__all__ = [
  'FocalLoss'
]

registry.register.semantic_loss('ce', default=True)(nn.CrossEntropyLoss)

@registry.register.semantic_loss('focal')
class FocalLoss(nn.Module):
  def __init__(
    self,
    weight: Optional[torch.Tensor] = None,
    gamma: float = 2,
    reduction: str = 'mean',
    class_dim: int = 1
  ):
    """Focal loss
    See "Focal Loss for Dense Object Detection" arxiv:1708.02002

    Args:
      weight (torch.Tensor, optional): class weights. Defaults to None.
      gamma (float, optional): gamma. Defaults to 2.
      reduction (str, optional): reduction mean. Defaults to 'mean'.
      class_dim (int, optional): class dimension. Defaults to 1.
    """
    super().__init__()
    if weight is not None:
      weight = torch.as_tensor(weight)
      self.register_buffer('weight', weight)
    else:
      self.weight = None
    self.gamma = gamma
    self.reduction = reduction
    self.class_dim = class_dim
  
  def forward(
    self,
    input: torch.Tensor,
    target: torch.Tensor
  ) -> torch.Tensor:
    """Compute focal loss

    Args:
      input (torch.Tensor): 4D tensor (b, c, h, w).
      target (torch.Tensor): ground truth label. Usually
        the shape is (b, h, w).

    Returns:
      torch.Tensor: focal loss
    """  
    log_prob = nn.functional.log_softmax(input, dim=self.class_dim)
    prob = torch.exp(log_prob)
    return nn.functional.nll_loss(
      ((1-prob) ** self.gamma) * log_prob,
      target,
      weight = self.weight,
      reduction = self.reduction
    )