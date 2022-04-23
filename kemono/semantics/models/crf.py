# --- built in ---
from typing import Any, Dict, Tuple, Union
# --- 3rd party ---
import torch
from torch import nn
import CRF
import rlchemy
from rlchemy import registry
# --- my module ---
from kemono.semantics.models.base import BaseSemanticModel

@registry.register.semantic_model('crf')
class CRFModel(BaseSemanticModel):
  def __init__(
    self,
    base_model: str,
    num_classes: int = 40,
    base_model_kwargs: Dict[str, Any] = {},
    inference_only: bool = False
  ):
    super().__init__()
    model_class = registry.get.semantic_model(base_model)
    assert model_class is not None
    self.model = model_class(
      **base_model_kwargs
    )
    x0_weight = 0.0 if inference_only else 0.5
    self.params = CRF.FrankWolfeParams(
      scheme = 'fixed', # constant stepsize
      stepsize = 1.0,
      regularizer = 'l2',
      lambda_ = 1.0, # regularization weight
      lambda_learnable = False,
      x0_weight = x0_weight, # useful for training, set to 0 if inference only
      x0_weight_learnable = False
    )
    self.crf = CRF.DenseGaussianCRF(
      classes = num_classes,
      alpha = 160,
      beta = 0.05,
      gamma = 3.0,
      spatial_weight = 1.0,
      bilateral_weight = 1.0,
      compatibility = 1.0,
      init = 'potts',
      solver = 'fw',
      iterations = 5,
      params = self.params
    )

  def forward(
    self,
    rgb: torch.Tensor,
    depth: torch.Tensor,
    **kwargs
  ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
    out = self.model(rgb, depth, **kwargs)
    if isinstance(out, tuple):
      logits = self.crf(rgb, out[0])
      return (logits, *out[1:])
    else:
      out = self.crf(rgb, out)
      return out
