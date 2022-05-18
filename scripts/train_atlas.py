# --- built in ---
# --- 3rd party ---
from omegaconf import OmegaConf
import pytorch_lightning as pl
# --- my module ---
import kemono
from kemono.atlas.task import AtlasTask

def main(args):
  # create configurations
  configs = []
  if args.config is not None:
    configs.append(OmegaConf.load(args.config))
  configs.append(OmegaConf.from_dotlist(args.options))
  conf = OmegaConf.merge(*configs)
  OmegaConf.resolve(conf)
  # create model & trainer
  model = AtlasTask(conf.task)

  checkpoint_callback = pl.callbacks.ModelCheckpoint(
    **conf.checkpoint
  )
  trainer = pl.Trainer(
    callbacks = checkpoint_callback,
    **conf.trainer
  )

  model._preview_predictions(0)
  model._preview_predictions(1)
  model._preview_predictions(2)
  exit(0)

  # start training
  trainer.fit(model)


if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(add_help=True)
  parser.add_argument('--config', type=str, required=True, help='configuration path')
  parser.add_argument('options', nargs=argparse.REMAINDER)
  args = parser.parse_args()
  main(args)