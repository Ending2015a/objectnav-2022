# --- built in ---
import os
import glob
from typing import (
  Any,
  List,
  Optional
)
# --- 3rd party --
import tqdm
import numpy as np
import torch
from torch import nn
import einops
from omegaconf import OmegaConf
# --- my module --
from kemono.semantics.task import HabitatDataset
from kemono.semantics import utils

def main(args):
  # Create configurations
  configs = []
  if args.config is not None:
    configs.append(OmegaConf.load(args.config))
  configs.append(OmegaConf.from_dotlist(args.dot_list))
  conf = OmegaConf.merge(*configs)
  OmegaConf.resolve(conf)
  # Create dataset
  trainset = HabitatDataset(**conf.task.trainset)
  num_classes = conf["global"].num_classes
  num_samples = len(trainset)
  img_size = conf.task.trainset.img_size
  # Calculate pixel counts for all classes
  pixel_counts = np.zeros((num_classes,), dtype=np.int64)
  ratios = []
  for idx in tqdm.tqdm(range(num_samples)):
    rgb, depth, seg = trainset[idx]
    seg = torch.from_numpy(seg[0])
    seg_oh = nn.functional.one_hot(seg, num_classes)
    seg_oh = seg_oh.view((-1, num_classes)).sum(dim=0)
    seg_oh = seg_oh / seg_oh.sum()
    ratios.append(seg_oh.detach().numpy())
  ratio_mean = np.asarray(ratios).mean(axis=0)
  print("Class ratio:")
  for idx, ratio in enumerate(ratio_mean):
    name = utils.mpcat40categories[idx].mpcat40
    print(f'  {idx}. {name}: {ratio:.6f}')
  ratio_sum = np.asarray(ratios).sum(axis=0)
  weights = 1 / (np.log(ratio_sum) + np.log(np.prod(img_size)) + 1e-5)
  weights = num_classes * weights / sum(weights)
  print(weights.tolist())



if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(add_help=True)
  parser.add_argument('--config', type=str, required=True, help='configuration file')
  parser.add_argument('dot_list', nargs=argparse.REMAINDER)
  args = parser.parse_args()
  main(args)