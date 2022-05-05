# --- built in ---
import sys
import collections
import copy
from typing import (
  Any,
  List,
  Tuple,
  Union,
  Callable,
  Iterator,
  Iterable,
  Optional
)
# --- 3rd party ---
import numpy as np
import torch
from torch.utils.data import IterableDataset
import rlchemy
# --- my module --
from kemono.data.stream_producer import BaseStreamProducer
from kemono.data.wrap import StreamProducerWrapper


__all__ = [
  'Dataset',
  'StreamDataset'
]

class Dataset(IterableDataset):
  def repeat(self, count: Optional[int]=None):
    return RepeatDataset(self, count=count)

  def shuffle(
    self,
    buffer_size: int,
    reshuffle_each_iteration: bool = True
  ) -> "Dataset":
    return ShuffleDataset(
      self,
      buffer_size = buffer_size,
      reshuffle_each_iteration = reshuffle_each_iteration
    )

  def batch(
    self,
    batch_size: int,
    drop_remainder: bool = False
  ) -> "Dataset":
    return BatchDataset(
      self,
      batch_size,
      drop_remainder = drop_remainder
    )

  @staticmethod
  def range(*args, **kwargs) -> "Dataset":
    return RangeDataset(*args, **kwargs)
  
  @staticmethod
  def choice(
    a: np.ndarray,
    size: Union[int, Tuple[int, ...]],
    replace: bool = False,
    p: np.ndarray = None
  ):
    return ChoiceDataset(a, size, replace, p)

  @staticmethod
  def from_generator(generator_fn: Callable) -> "Dataset":
    return GeneratorDataset(generator_fn)

  @staticmethod
  def from_element(element) -> "Dataset":
    return ElementDataset(element)

  @staticmethod
  def from_slices(elements) -> "Dataset":
    return SlicesDataset(elements)

  @staticmethod
  def zip(datasets) -> "Dataset":
    return ZipDataset(datasets)

  @staticmethod
  def choose_from_datasets(
    datasets: Iterable[IterableDataset],
    choose_dataset: IterableDataset,
    stop_on_empty_dataset: bool = True
  ) -> "Dataset":
    return ChooseFromDatasets(
      datasets,
      choose_dataset,
      stop_on_empty_dataset
    )

class BatchDataset(Dataset):
  def __init__(
    self,
    dataset: IterableDataset,
    batch_size: int,
    drop_remainder: bool = False
  ):
    assert isinstance(dataset, IterableDataset)
    assert batch_size > 0
    self.dataset = dataset
    self.batch_size = batch_size
    self.drop_remainder = drop_remainder

  def __iter__(self) -> Iterator:
    def _stack_op(batch_data):
      if torch.is_tensor(batch_data[0]):
        return torch.stack(batch_data, dim=0)
      else:
        batch_data = list(map(np.asarray, batch_data))
        return np.stack(batch_data, axis=0)
    # create dataset iterator
    batch = []
    for data in self.dataset:
      batch.append(data)
      if len(batch) == self.batch_size:
        yield rlchemy.utils.map_nested_tuple(tuple(batch), op=_stack_op)
        # reset batch
        batch = []
    # return the last batch
    if len(batch) > 0 and not self.drop_remainder:
      yield rlchemy.utils.map_nested_tuple(tuple(batch), op=_stack_op)


class ChooseFromDatasets(Dataset):
  def __init__(
    self,
    datasets: Iterable[IterableDataset],
    choose_dataset: IterableDataset,
    stop_on_empty_dataset: bool = True
  ):
    self.datasets = []
    for dataset in datasets:
      assert isinstance(dataset, IterableDataset)
      self.datasets.append(dataset)
    assert isinstance(choose_dataset, IterableDataset)
    self.datasets = tuple(self.datasets)
    self.choose_dataset = choose_dataset
    self.stop_on_empty_dataset = stop_on_empty_dataset
    self._num_datasets = len(self.datasets)

  def __iter__(self) -> Iterator:
    # create dataset iterators
    dataset_its = [
      iter(self.datasets[i])
      for i in range(self._num_datasets)
    ]
    for index in self.choose_dataset:
      if index >= self._num_datasets:
        raise IndexError(f"Index out of bounds: {index}")
      try:
        yield next(dataset_its[index])
      except StopIteration:
        if self.stop_on_empty_dataset:
          return


class ElementDataset(Dataset):
  def __init__(self, element: Any):
    self.element = element

  def __iter__(self) -> Iterator:
    yield self.element


class SlicesDataset(Dataset):
  def __init__(self, elements: Any):
    self.elements = elements
    self.n_slices = len(self.elements)
  
  def __iter__(self) -> Iterator:
    for i in range(self.n_slices):
      yield self.elements[i]

class RepeatDataset(Dataset):
  def __init__(
    self,
    dataset: IterableDataset,
    count: Optional[int] = None
  ):
    assert isinstance(dataset, IterableDataset)
    if count is None:
      count = -1
    assert isinstance(count, int), \
      f"`count` must be an integer, got {type(count)}"
    self.dataset = dataset
    self.count = count
  
  def __iter__(self) -> Iterator:
    if self.count < 0:
      while True:
        yield from self.dataset
    else:
      for _ in range(self.count):
        yield from self.dataset


class ShuffleDataset(Dataset):
  def __init__(
    self,
    dataset: IterableDataset,
    buffer_size: int,
    reshuffle_each_iteration: bool = True
  ):
    """This dataset fills a buffer with buffer_size elements, then randomly
    samples elements from this buffer, replacing the selected elements with
    new elements. For perfect shuffling, a buffer size greater than or equal
    to the full size of the dataset is required.

    Args:
        dataset (IterableDataset): dataset.
        buffer_size (int): buffer size.
    """
    assert isinstance(dataset, IterableDataset)
    assert buffer_size > 0
    self.dataset = dataset
    self.buffer_size = buffer_size
    self.reshuffle_each_iteration = reshuffle_each_iteration
    # ---
    self._seed_sequence = np.random.SeedSequence()
    self._buffer = None
    self._rng = None

  def reset(self):
    self._buffer = []
    if self.reshuffle_each_iteration:
      # reset random generator on each epoch
      self._rng = np.random.default_rng()
    else:
      self._rng = np.random.default_rng(
        np.random.PCG64(self._seed_sequence)
      )

  def random_pop(
    self,
    data_replace: Optional[Any] = None
  ) -> Any:
    index = self._rng.choice(len(self._buffer))
    if data_replace is None:
      pop_data = self._buffer.pop(index)
    else:
      pop_data = self._buffer[index]
      self._buffer[index] = data_replace
    return pop_data

  def __iter__(self) -> Iterator:
    self.reset()
    for data in self.dataset:
      # fill in the buffer
      if len(self._buffer) < self.buffer_size:
        self._buffer.append(data)
        continue
      # random pop data from buffer
      # and replace it with the new one
      yield self.random_pop(data_replace=data)
    # pop remaining data in the buffer
    while len(self._buffer) > 0:
      yield self.random_pop()


class ChoiceDataset(Dataset):
  def __init__(
    self,
    a: np.ndarray,
    size: Optional[Union[int, Tuple[int, ...]]] = None,
    replace: bool = False,
    p: np.ndarray = None
  ):
    """Randomly chooce samples from a given 1D-array
    Note that if the size is specified the dataset will
    iterate over the first dimension. For example, if (3, 5)
    then there are 3 slices of samples, each sample has shape
    (5,)

    Args:
        array (np.ndarray): _description_
        size (Union[int, Tuple[int, ...]]): _description_
        replace (bool, optional): _description_. Defaults to False.
        p (np.ndarray, optional): _description_. Defaults to None.
    """
    self.a = copy.deepcopy(a)
    self.size = copy.deepcopy(size)
    self.replace = replace
    self.p = copy.deepcopy(p)
    # ---
    self._seed_sequence = np.random.SeedSequence()
    self._rng = np.random.default_rng(
      np.random.PCG64(self._seed_sequence)
    )

  def __iter__(self) -> Iterator:
    choices = self._rng.choice(
      self.a,
      size = self.size,
      replace = self.replace,
      p = self.p
    )
    if np.isscalar(choices):
      yield choices
    else:
      for choice in choices:
        yield choice

class RangeDataset(Dataset):
  def __init__(self, *args, **kwargs):
    _slice = slice(*args, **kwargs)
    self._start = _slice.start or 0
    self._stop = _slice.stop or sys.maxsize
    self._step = _slice.step or 1

  def __iter__(self) -> Iterator:
    yield from range(self._start, self._stop, self._step)


class GeneratorDataset(Dataset):
  def __init__(self, generator_fn: Callable):
    self.generator_fn = generator_fn

  def __iter__(self) -> Iterator:
    yield from self.generator_fn()


class ZipDataset(Dataset):
  def __init__(
    self,
    datasets: Iterable[IterableDataset]
  ):
    self.datasets = []
    for dataset in datasets:
      assert isinstance(dataset, IterableDataset)
      self.datasets.append(dataset)
    self.datasets = tuple(self.datasets)

  def __iter__(self) -> Iterator:
    yield from zip(*self.datasets)


class StreamDataset(Dataset):
  def __init__(
    self,
    stream_producer: Union[
      BaseStreamProducer, Iterable[BaseStreamProducer]],
    num_workers: Optional[int] = None,
    group_size: Optional[int] = None,
    seq_len: Optional[int] = None,
    drop_remainder: bool = False
  ):
    # multiple workers
    if num_workers is not None:
      # vectorized stream producer
      if isinstance(stream_producer, collections.abc.Iterable):
        # convert iterable to list
        stream_producer = list(stream_producer)
        assert len(stream_producer) == num_workers
      else:
        assert isinstance(stream_producer, (
          BaseStreamProducer,
          StreamProducerWrapper
        ))
        # if a single stream producer is provided
        # copy the instance for `num_workers` times
        # note that in this case a single stream producer
        # is shared across multiple workers/threads
        # make sure your stream producers are thread-safe!!!!
        stream_producer = [stream_producer for i in range(num_workers)]
      cls = type(self)
      datasets = [
        Dataset.zip((
          Dataset.from_element(i).repeat(-1),
          cls(
            stream_producer = stream_producer[i],
            num_workers = None,
            group_size = group_size,
            seq_len = seq_len,
            drop_remainder = drop_remainder
          )
        ))
        for i in range(num_workers)
      ]
      group_size = group_size or num_workers
      choose_dataset = (
        Dataset.choice(range(num_workers), num_workers, replace=False)
               .repeat(-1)
      )
      dataset = Dataset.choose_from_datasets(datasets, choose_dataset)
    else:
      assert isinstance(stream_producer, (
        BaseStreamProducer,
        StreamProducerWrapper
      ))
      dataset = Dataset.from_generator(
        self.create_generator(
          stream_producer,
          seq_len,
          drop_remainder
        )
      )
      dataset = dataset.repeat(-1)
    self.dataset = dataset

  def __iter__(self) -> Iterator:
    yield from self.dataset

  @staticmethod
  def create_generator(
    stream_producer: BaseStreamProducer,
    seq_len: Optional[int],
    drop_remainder: bool
  ):
    def _wrapped_generator() -> Iterator:
      """Load streams from stream producer and generate batch data"""
      def _slice_op(x: Any, ind: int) -> Any:
        """slice sinple sample"""
        return x[ind]

      def _slice_chunk_op(x: Any, ind: int):
        if ind + seq_len > stream_length:
          return x[ind:]
        else:
          return x[ind:ind+seq_len]

      def _pad_chunk_op(x: Any, pad_size: int) -> Any:
        pad_size = [(0, pad_size)]
        for _ in range(1, len(x.shape)):
          pad_size.append((0, 0))
        return np.pad(x, pad_size)

      def _slice_and_pad_chunk_op(x: Any, ind: int) -> Any:
        data = _slice_chunk_op(x, ind)
        pad_size = seq_len - len(data)
        if pad_size == 0:
          return data
        # pad data to seq_len
        return _pad_chunk_op(data, pad_size)

      # recharge stream producer
      stream_producer.maybe_recharge()
      stream = stream_producer(iterate=False)
      first_data = next(iter(rlchemy.utils.iter_nested(stream.data)))
      stream_length = len(first_data)
      if seq_len is None:
        # slice one sample
        for i in range(0, stream_length):
          yield rlchemy.utils.map_nested(
            stream.data, _slice_op, i
          )
      else:
        for i in range(0, stream_length, seq_len):
          if drop_remainder:
            # ignore the last chunk that smaller than `seq_len`
            if i + seq_len > stream_length:
              break
          yield rlchemy.utils.map_nested(
            stream.data, _slice_and_pad_chunk_op, i
          )
    return _wrapped_generator
