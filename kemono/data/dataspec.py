# --- built in ---
import os
from dataclasses import dataclass
from typing import (
  Any,
  Optional,
  Callable,
  Tuple
)
from distutils.version import StrictVersion as Version
# --- 3rd party ---
import numpy as np
# --- my module ---

__all__ = [
  'DataSpec',
  'generate_dataspec',
]

@dataclass
class DataSpec:
  shape: Tuple[int]
  dtype: np.dtype

def generate_dataspec(
  data: Any,
  slicing: bool = True,
  stack_size: Optional[int] = None,
  spec_type: type = DataSpec
) -> Any:
  """
  Generate nested data spec from the given nested data. The returned
  object has the same structure as the given data.

  Args:
    data (Any): nested data
    slicing (bool, optional): slice data
    stack_size (int, optional): number of data to stack per sample. Set None to
     disable stacking. Defaults to None.
    spec_type (type, optional): data spec type.
  
  Returns:
    Any: nested data spec
  """

  def _nested_generate_dataspec(data):
    if isinstance(data, dict):
      return {
        k: _nested_generate_dataspec(v)
        for k, v in data.items()
      }
    elif isinstance(data, tuple):
      return tuple(
        _nested_generate_dataspec(v)
        for v in data
      )
    else:
      # slice data
      if slicing:
        data = data[0]
      # get dtype
      if hasattr(data, 'dtype'):
        dtype = data.dtype
      else:
        dtype = type(data)
      # get shape
      if hasattr(data, 'shape'):
        shape = data.shape
      elif isinstance(data, (list)):
        shape = np.asarray(data).shape
      else:
        raise RuntimeError(f"Unknown data type: {type(data)}")
      # stack shape
      if stack_size is not None:
        shape = [stack_size] + list(shape)
      return spec_type(shape=shape, dtype=dtype)

  return _nested_generate_dataspec(data)
