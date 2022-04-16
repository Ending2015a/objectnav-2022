# --- built in ---
# --- 3rd party ---
import numpy as np
# --- my module ---
from kemono.data.dataset import Dataset, StreamDataset
from kemono.data import RlchemyDynamicStreamProducer
from kemono.data.wrap import ZeroPadChunkStreamWrapper

def test_range():
  dataset = Dataset.range(10)
  for idx, data in enumerate(dataset):
    assert data == idx
  assert data == 9

def test_batch():
  dataset = Dataset.range(10)
  dataset = dataset.batch(3, drop_remainder=True)
  for idx, data in enumerate(dataset):
    assert np.all(data == np.asarray([0, 1, 2]) + idx*3)
  assert idx == 2
  assert np.all(data == np.asarray([6, 7, 8]))

def test_shuffle():
  dataset = Dataset.range(10)
  dataset = dataset.shuffle(5, reshuffle_each_iteration=True)
  all_data = []
  for idx, data in enumerate(dataset):
    all_data.append(data)
  # must be shuffled
  assert np.any(np.arange(10) != np.asarray(all_data))
  all_data = sorted(all_data)
  assert np.all(np.arange(10) == np.asarray(all_data))

def test_zip():
  dataset1 = Dataset.range(10)
  dataset2 = Dataset.range(10, 20)
  dataset = Dataset.zip([dataset1, dataset2])
  for idx, data in enumerate(dataset):
    assert np.all(np.asarray(data) == (np.asarray([0, 10]) + idx))
  assert idx == 9
  assert np.all(np.asarray(data) == np.asarray([9, 19]))

def test_repeat():
  dataset = Dataset.range(10).repeat(2)
  for idx, data in enumerate(dataset):
    assert idx%10 == data
  assert idx == 19
  assert data == 9

def test_from_generator():
  def hello_generator():
    for i in range(10):
      yield i
  dataset = Dataset.from_generator(hello_generator)
  for idx, data in enumerate(dataset):
    assert data == idx
  assert idx == 9
  assert data == 9
  # repeat
  dataset = Dataset.from_generator(hello_generator).repeat(2)

def test_from_element():
  dataset = Dataset.from_element(100).repeat(10)
  for idx, data in enumerate(dataset):
    assert data == 100
  assert idx == 9

def test_choose_from_datasets():
  index_dataset = Dataset.range(2).repeat(5)
  dataset1 = Dataset.range(10)
  dataset2 = Dataset.range(10, 19)
  dataset = Dataset.choose_from_datasets(
    [dataset1, dataset2],
    index_dataset,
  )
  for idx, data in enumerate(dataset):
    assert data == 10 * (idx % 2) + (idx // 2)
  assert idx == 9
  assert data == 14

  index_dataset = Dataset.range(2).repeat(-1)
  dataset = Dataset.choose_from_datasets(
    [dataset1, dataset2],
    index_dataset,
    stop_on_empty_dataset = True
  )
  for idx, data in enumerate(dataset):
    assert data == 10 * (idx % 2) + (idx // 2)
  assert idx == 18
  assert data == 9

def test_choice_dataset():
  dataset = Dataset.choice(10, size=5, replace=False).repeat(100).batch(5)
  for idx, data in enumerate(dataset):
    assert len(data) == len(np.unique(data))

def test_stream_dataset():
  producer = RlchemyDynamicStreamProducer(
    data_root = '/src/logs/kemono_expert/trajs/',
    glob_paths = '*.trajectory.npz'
  )
  producer = ZeroPadChunkStreamWrapper(producer, chunk_size=10)
  dataset = StreamDataset(
    producer,
    num_workers = 3,
    group_size = 3,
    seq_len = 10,
    drop_remainder = False
  )
  for idx, data in enumerate(dataset):
    print(data[0], data[1]['done'])
    print(data[0], data[1]['mask'])
    if idx == 100:
      break

def example():
  test_range()
  test_batch()
  np.random.seed(10)
  test_shuffle()
  test_zip()
  test_repeat()
  test_from_generator()
  test_from_element()
  test_choose_from_datasets()
  test_choice_dataset()
  test_stream_dataset()
  



if __name__ == '__main__':
  example()