# --- built in ---
import os
import copy
import glob
from typing import List
# --- 3rd party ---
import habitat
from omegaconf import OmegaConf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import einops
# --- my module ---
import kemono


CONFIG_PATH = '/src/configs/atlas/test_multi_atlas.yaml'

EP_GLOB_PATTERN = '**/ep-*/'
SAMPLE_GLOB_PATTERN = '**/sample_*.npz'


def glob_filenames(
  root_path: str,
  pattern: str
) -> List[str]:
  root_path = os.path.abspath(root_path)
  glob_path = os.path.join(root_path, pattern)
  # glob abspaths
  abspaths = glob.glob(glob_path, recursive=True)
  # convert to relpaths
  relpaths = [os.path.relpath(path, start=root_path) for path in abspaths]
  relpaths = sorted(relpaths)
  return relpaths


def plot_chart(ax, chart):
  ax.imshow(chart, cmap='hot', interpolation='nearest')
  ax.set_title('Ground truth chart', fontsize='xx-large')

def plot_chart_gt_distance(ax, chart_gt):
  dist_map = chart_gt[...,0]
  cmap = copy.copy(plt.cm.viridis)
  cmap.set_bad(color='black')
  ax.imshow(dist_map, cmap=cmap, interpolation='nearest')
  ax.set_title('Ground truth distance', fontsize='xx-large')

def plot_chart_gt_gradient(ax, chart, chart_gt):
  vector_map = chart_gt[..., 1:]
  vector_map[np.isinf(vector_map)] = 0
  height, width = vector_map.shape[:-1]
  mesh = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
  mesh = np.stack(mesh, axis=-1).astype(np.float32)
  mesh = mesh.reshape((-1, 2))
  vector_map = vector_map.reshape((-1, 2))
  mask = np.logical_and(vector_map[...,0]==0, vector_map[...,1]==0)
  mesh = mesh[~mask]
  vector_map = vector_map[~mask]
  ax.imshow(chart, cmap='hot', interpolation='nearest')
  theta = np.arctan2(vector_map[...,1], vector_map[...,0])

  ph = np.linspace(0, 2*np.pi, 13)
  u = np.cos(ph)
  v = np.sin(ph)
  colors = np.arctan2(v, u)
  norm = Normalize()
  norm.autoscale(colors)

  ax.quiver(mesh[...,1], mesh[...,0],
    vector_map[...,0], vector_map[...,1],
    color = plt.cm.hsv(norm(theta)),
    angles='xy',
    width = 0.001)
  ax.set_title('Ground truth gradient', fontsize='xx-large')


def plot_sampled_points(ax, chart, points):
  height, width = chart.shape
  rsize = np.asarray((height, width), dtype=np.float32)
  points = (points * 0.5 + 0.5) * rsize
  ax.imshow(chart, cmap='hot', interpolation='nearest')
  ax.scatter(points[..., 0], points[..., 1], color='r', s=1)
  ax.set_title('Sampled points', fontsize='xx-large')

def example(args):
  np.random.seed(1)
  conf = OmegaConf.load(CONFIG_PATH)
  ep_list = glob_filenames(conf.sample.dirpath, EP_GLOB_PATTERN)
  assert len(ep_list) > 0
  for ep in range(len(ep_list)):
    eppath = os.path.join(conf.sample.dirpath, ep_list[ep])
    epdata = kemono.atlas.AtlasData.load(eppath)
    num_samples = len(epdata.paths)
    if args.samples_per_ep is not None:
      num_samples = len(epdata.paths[:args.samples_per_ep])

    for sample_id in range(num_samples):
      sample_path = epdata.paths[sample_id]
      sample_path = os.path.join(eppath, sample_path)
      data = kemono.atlas.ChartData.load(sample_path)

      chart = data.chart
      chart_gt = data.chart_gt
      if args.resize is not None:
        chart = kemono.atlas.utils.resize_image(
          data.chart,
          size = (args.resize, args.resize),
          mode = 'nearest'
        )
        chart_gt = kemono.atlas.utils.resize_image(
          einops.rearrange(data.chart_gt, 'h w c -> c h w'),
          size = (args.resize, args.resize),
          mode = 'nearest'
        )
        chart_gt = einops.rearrange(chart_gt, 'c h w -> h w c')

      fig, axs = plt.subplots(
        figsize=(12, 12),
        ncols=2, nrows=2,
        dpi=300,
        sharex = True,
        sharey = True
      )
      plot_chart(axs[0][0], chart)
      plot_chart_gt_distance(axs[0][1], chart_gt)
      plot_chart_gt_gradient(axs[1][0], chart, chart_gt)
      plot_sampled_points(axs[1][1], chart, data.points)
      plt.tight_layout()
      plt.show()


if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--resize', type=int, default=None)
  parser.add_argument('--samples-per-ep', type=int, default=None)
  args = parser.parse_args()
  example(args)


