# --- built in ---
import os
import copy
from typing import (
  Optional
)
# --- 3rd party ---
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def get_color_norm():
  ph = np.linspace(0, 2*np.pi, 13)
  u = np.cos(ph)
  v = np.sin(ph)
  colors = np.arctan2(v, u)
  color_norm = Normalize()
  color_norm.autoscale(colors)
  return color_norm

COLOR_NORM = get_color_norm()



def plot_energy(
  ax,
  energy_map: np.ndarray,
  title: Optional[str] = None,
  cmap: Optional[matplotlib.colors.Colormap] = None,
  interpolation: str = 'nearest'
):
  """
  Args:
    ax (matplotlib.axes.Axes): axes to plot.
    energy_map (np.ndarray): expecting (h, w), inf for invalid areas.
    title (Optional[str], optional): chart title. Defaults to None.
    cmap (Optional[matplotlib.colors.Colormap]): color map.
  """
  if cmap is None:
    cmap = plt.cm.viridis
  cmap = copy.copy(cmap)
  cmap.set_bad(color='black')
  ax.imshow(energy_map, cmap=cmap, interpolation=interpolation)
  if title is not None:
    ax.set_title(title, fontsize='xx-large')

# def plot_score(
#   ax,
#   vector_map: np.ndarray,
#   chart: Optional[np.ndarray] = None,
#   title: Optional[str] = None
# ):
#   vector_map[np.isinf(vector_map)] = 0
#   height, width = vector_map.shape[:-1]
#   mesh = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
#   mesh = np.stack(mesh, axis=-1).astype(np.float32)
#   mesh = mesh.reshape((-1, 2))
#   vector_map = vector_map.reshape((-1, 2))
#   mask = np.logical_and(vector_map[...,0]==0, vector_map[...,1]==0)
#   mesh = mesh[~mask]
#   vector_map = vector_map[~mask]
#   ax.imshow(chart, cmap='hot', interpolation='nearest')
#   theta = np.arctan2(vector_map[...,1], vector_map[...,0])

#   ph = np.linspace(0, 2*np.pi, 13)
#   u = np.cos(ph)
#   v = np.sin(ph)
#   colors = np.arctan2(v, u)
#   norm = Normalize()
#   norm.autoscale(colors)

#   ax.quiver(mesh[...,1], mesh[...,0],
#     vector_map[...,0], vector_map[...,1],
#     color = plt.cm.hsv(norm(theta)),
#     angles='xy',
#     width = 0.001)
#   if title is not None:
#     ax.set_title(title, fontsize='xx-large')


def plot_score(
  ax,
  score_map: np.ndarray,
  chart: Optional[np.ndarray] = None,
  title: Optional[str] = None,
  cmap: Optional[matplotlib.colors.Colormap] = None,
  interpolation: str = 'nearest',
  line_width: float = 0.001,
  **quiver_kwargs
):
  """_summary_

  Args:
    ax (_type_): _description_
    score_map (np.ndarray): expecting (h, w, 2), [x, y]
    chart (np.ndarray): expecting (h, w) or (h, w, 3)
    title (Optional[str], optional): chart title. Defaults to None.
  """
  score_map[np.isinf(score_map)] = 0
  #score_map[np.isnan(score_map)] = 0
  height, width = score_map.shape[:-1]
  mesh = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
  mesh = np.stack(mesh, axis=-1).astype(np.float32)
  mesh = mesh.reshape((-1, 2))
  score_map = score_map.reshape((-1, 2))
  mask = np.logical_and(score_map[...,0]==0, score_map[...,1]==0)
  mesh = mesh[~mask]
  score_map = score_map[~mask]
  if chart is not None:
    if len(chart.shape) < 3 or chart.shape[-1] == 1:
      # binary mask (black, white)
      ax.imshow(chart, cmap='hot', interpolation=interpolation)
    else:
      ax.imshow(chart)
  theta = np.arctan2(score_map[...,1], score_map[...,0])
  if cmap is None:
    cmap = plt.cm.hsv
  else:
    cmap = plt.cm.get_cmap(cmap)
  # plot score field
  ax.quiver(
    mesh[...,1], mesh[...,0],
    score_map[...,0], score_map[...,1],
    color = cmap(COLOR_NORM(theta)),
    angles = 'xy',
    width = line_width,
    **quiver_kwargs
  )
  if title is not None:
    ax.set_title(title, fontsize='xx-large')


def plot_image(
  ax,
  image: np.ndarray,
  title: Optional[str] = None,
  cmap: Optional[matplotlib.colors.Colormap] = None,
  interpolation: str = 'nearest'
):
  if len(image.shape) < 3 or image.shape[-1] == 1:
    # binary mask (black, white)
    if cmap is None:
      cmap = plt.cm.get_cmap('hot')
    ax.imshow(image, cmap=cmap, interpolation=interpolation)
  else:
    ax.imshow(image)
  if title is not None:
    ax.set_title(title, fontsize='xx-large')