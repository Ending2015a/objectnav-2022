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
import tensorflow as tf
import numpy as np
# --- my module ---

__all__ = [
  'DataSpec',
  'generate_dataspec',
  'generate_tf_spec',
  'generate_tf_dataset'
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
):
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

def generate_tf_spec(
  dataspec: DataSpec,
  stack_size: Optional[int] = None
) -> Tuple[Any, Any]:
  """Generate data spec for tensorflow Dataset API (tensorflow < 2.4.0)
  from the given dataspec. The returned object has the same structure as
  the given data.`tf.data.Dataset.from_generator` needs `output_types`
  and `output_shapes`. Use this function to generate them.

  Args:
    dataspec (DataSpec): data spec.
    stack_size (int, optional): if the data stacking is enabled, specify this
      value. Defaults to None.

  return:
    Tuple[Any, Any]: nested shapes, nested dtypes.
  """
  def _nested_generate_tf_spec(dataspec):

    if isinstance(dataspec, dict):
      shape_dict = type(dataspec)()
      dtype_dict = type(dataspec)()
      for k, v in dataspec.items():
        shape, dtype = _nested_generate_tf_spec(v)
        shape_dict[k] = shape
        dtype_dict[k] = dtype
      return shape_dict, dtype_dict
    elif isinstance(dataspec, tuple):
      shape_tuple = []
      dtype_tuple = []
      for v in dataspec:
        shape, dtype = _nested_generate_tf_spec(v)
        shape_tuple.append(shape)
        dtype_tuple.append(dtype)
      return tuple(shape_tuple), tuple(dtype_tuple)
    else:
      shape = dataspec.shape
      dtype = tf.dtypes.as_dtype(dataspec.dtype)
      # stack shape
      if stack_size is not None:
        shape = [stack_size] + list(shape)
      # return dtype/shape
      return shape, dtype

  return _nested_generate_tf_spec(dataspec)

def generate_tf_dataset(
  generator: Callable,
  dataspec: DataSpec,
  chunk_size: Optional[int] = None,
  convert_dataspec: bool = True
):
  """
  Generate `tf.data.Dataset` by giving dataspec
  """
  if Version(tf.__version__) >= '2.4.0':
    # convert nested DataSpec to nested tf.TensorSpec
    if convert_dataspec:
        dataspec = generate_dataspec(
          dataspec,
          slicing = False,
          stack_size = chunk_size,
          spec_type = tf.TensorSpec
        )
    # generate dataset
    return tf.data.Dataset.from_generator(
      generator,
      output_signature = dataspec
    )
  else:
    # for tensorflow < 2.4.0
    shapes, dtypes = generate_tf_spec(dataspec, stack_size=chunk_size)
    # generate dataset
    return tf.data.Dataset.from_generator(
      generator,
      output_shapes = shapes,
      output_types = dtypes
    )