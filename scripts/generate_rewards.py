# --- built in ---
import os
import argparse
import glob
# --- 3rd party ---
import numpy as np
import rlchemy
from rlchemy import registry
# --- my module ---
from kemono.envs import rewards


def get_slice(data, index):
  def _slice_op(data):
    return data[index]
  return rlchemy.utils.map_nested(data, op=_slice_op)

def stack_slices(data):
  def _stack_op(data):
    return np.stack(data, axis=0)
  return rlchemy.utils.map_nested_tuple(tuple(data), op=_stack_op)

def example(args):
  path = os.path.join(args.root_path, args.glob_path)
  stream_paths = glob.glob(path, recursive=True)
  reward_class = registry.get.reward(args.reward)
  assert reward_class is not None
  reward_fn = reward_class()

  for stream_path in stream_paths:
    rel_stream_path = os.path.relpath(stream_path, start=args.root_path)
    output_stream_path = os.path.join(args.output_path, rel_stream_path)
    data = rlchemy.envs.load_trajectory(stream_path)
    new_data = []
    if 'rew' in data.keys():
      for step in range(len(data['rew'])):
        step_data = get_slice(data, step)
        del step_data['rew']
        new_rew = reward_fn(None, **step_data)
        step_data['rew'] = new_rew
        new_data.append(step_data)
      new_data = stack_slices(new_data)
    else:
      new_data = data
    os.makedirs(os.path.dirname(output_stream_path), exist_ok=True)
    rlchemy.envs.monitor.save_trajectory(output_stream_path, new_data)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--root_path', type=str, required=True)
  parser.add_argument('--output_path', type=str, required=True)
  parser.add_argument('--glob_path', type=str, default='*.trajectory.npz')
  parser.add_argument('--reward', type=str, default='v1')
  args = parser.parse_args()
  example(args)