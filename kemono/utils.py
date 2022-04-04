# --- built in ---
# --- 3rd party ---
from omegaconf import OmegaConf
# --- my module ---

def format_resolver(string, *args):
  return string.format(*args)

OmegaConf.register_new_resolver(
  "format", format_resolver
)
