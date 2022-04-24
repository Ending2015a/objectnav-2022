# --- built in ---
# --- 3rd party ---
import numpy as np
# --- my module ---

EPS = 1e-4

def global_to_local(
  goal: np.ndarray,
  agent_pose: np.ndarray
) -> np.ndarray:
  """Convert goal in global space to local space

  Args:
    goal (np.ndarray): [x, z]
    agent_pose (np.ndarray): [x, z, yaw]
  """
  pos = agent_pose[:2]
  yaw = agent_pose[2]
  delta = goal - pos
  rot_x = np.array((np.cos(-yaw), -np.sin(-yaw)), dtype=np.float32)
  rot_z = np.array((np.sin(-yaw), np.cos(-yaw)), dtype=np.float32)
  dx = np.sum(delta * rot_x)
  dz = np.sum(delta * rot_z)
  return np.array((dx, dz), dtype=np.float32)

def cart_to_polar(goal: np.ndarray) -> np.ndarray:
  """Convert cartesian to polar

  Args:
    goal (np.ndarray): [x, z] local
  """
  r = np.linalg.norm(goal)
  if r < EPS:
    return np.array((0, 0), dtype=np.float32)
  x, z = goal[0], goal[1]
  th = np.arctan2(-x, z)
  return np.array((r, th), dtype=np.float32)