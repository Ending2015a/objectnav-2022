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

def load_config(path, resolve=False):
  conf = OmegaConf.load(path)
  conf = resolve_imports(conf)
  if resolve:
    OmegaConf.resolve(conf)
  return conf

def resolve_imports(conf):
  import_paths = conf.get('_import_', [])
  conf.pop('_import_', None)
  confs = []
  for path in import_paths:
    import_conf = load_config(path)
    confs.append(import_conf)
  confs.append(conf)
  conf = OmegaConf.merge(*confs)
  return conf

def init_ray():
  ip = os.environ.get('RAY_SERVER_IP', None)
  port = os.environ.get('RAY_SERVER_PORT', 10001)
  assert ip is not None, "Server IP not specified"
  print(f"Connect to ray server ray://{ip}:{port}")
  ray.init(address=f"ray://{ip}:{port}")
