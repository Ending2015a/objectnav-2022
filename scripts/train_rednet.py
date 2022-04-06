# --- built in ---
import time
import os
import glob
from typing import (
  Any,
  List,
  Optional
)
# --- 3rd party --
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.strategies.ddp import DDPStrategy
# --- my module --
from kemono.semantics import SemanticTask


def main(args):
  # Create configurations
  configs = []
  if args.config is not None:
    configs.append(OmegaConf.load(args.config))
  configs.append(OmegaConf.from_dotlist(args.dot_list))
  conf = OmegaConf.merge(*configs)
  OmegaConf.resolve(conf)
  # create model & trainer
  model = SemanticTask(conf.task)

  checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor = "validation/mIoU",
    **conf.checkpoint
  )
  if 'ddp' in conf:
    strategy = DDPStrategy(**conf.ddp)
  else:
    strategy = None
  trainer = pl.Trainer(
    callbacks = checkpoint_callback,
    strategy = strategy,
    **conf.trainer
  )
  # start training
  trainer.fit(model)



if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(add_help=True)
  parser.add_argument('--config', type=str, required=True, help='configuration file')
  parser.add_argument('dot_list', nargs=argparse.REMAINDER)
  args = parser.parse_args()
  main(args)