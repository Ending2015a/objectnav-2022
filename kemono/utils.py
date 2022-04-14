# --- built in ---
import os
# --- 3rd party ---
import ray
from omegaconf import OmegaConf
# --- my module ---

def format_resolver(string, *args):
  return string.format(*args)

OmegaConf.register_new_resolver(
  "format", format_resolver
)


def init_ray():
  ip = os.environ.get('RAY_SERVER_IP', None)
  port = os.environ.get('RAY_SERVER_PORT', 10001)
  assert ip is not None, "Server IP not specified"
  print(f"Connect to ray server ray://{ip}:{port}")
  ray.init(address=f"ray://{ip}:{port}")
