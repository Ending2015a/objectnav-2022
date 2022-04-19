# --- built in ---
# --- 3rd party ---
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.strategies.ddp import DDPStrategy
from habitat.config.default import get_config
import rlchemy
# --- my module ---
from kemono.agents.models.sac import SAC
from kemono.envs import habitat_env
from kemono.envs.wrap import (
  SemanticMapBuilderWrapper,
  SemanticWrapper,
  CleanObsWrapper,
  RayRemoteEnv
)

rlchemy.envs.TrajectoryRecorder.trajectory_suffix = \
  'ep{episodes:010d}.{start_steps}-{steps}'

def create_env(habitat_config, config):
  config = OmegaConf.create(config)
  # Create base env
  env = habitat_env.make(
    config.envs.env_id,
    habitat_config,
    **config.envs.habitat_env
  )
  # semantic segmentations
  env = SemanticWrapper(env, **config.envs.semantic_wrapper)
  # semantic top-down maps
  env = SemanticMapBuilderWrapper(
    env,
    **config.envs.semantic_map_builder
  )
  # clean observations
  env = CleanObsWrapper(
    env,
    **config.envs.clean_obs
  )
  env = rlchemy.envs.Monitor(
    env,
    **config.envs.monitor
  )
  env.add_tool(rlchemy.envs.TrajectoryRecorder(
    **config.envs.trajectory_recorder
  ))
  return env

def main(args):
  # Load OmegaConf configurations
  configs = []
  if args.config is not None:
    configs.append(OmegaConf.load(args.config))
  configs.append(OmegaConf.from_dotlist(args.dot_list))
  config = OmegaConf.merge(*configs)
  OmegaConf.resolve(config)
  # Load habitat configurations
  habitat_config = get_config(config.habitat_config)
  # finetune configurations (change seeds...etc)
  # habitat_config.SEED = 10
  # habitat_config.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_EPISODES = 5
  # create env
  if 'ray_remote_env' in config.envs:
    print('Creating remote environments')
    env = RayRemoteEnv(
      create_env,
      (habitat_config, config),
      **config.envs.ray_remote_env
    )
  else:
    print('Creating environments')
    env = create_env(habitat_config, config)
  # create model & trainer
  print('Creating SAC model')
  model = SAC(config.sac, env=env)
  # Create trainer
  print('Creating trainer')
  checkpoint_callback = pl.callbacks.ModelCheckpoint(
    **config.checkpoint
  )
  if 'ddp' in config:
    strategy = DDPStrategy(**config.ddp)
  else:
    strategy = None
  trainer = pl.Trainer(
    callbacks = checkpoint_callback,
    strategy = strategy,
    **config.trainer
  )
  # start training
  trainer.fit(model)


if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(add_help=True)
  parser.add_argument('--config', type=str, required=True, help='Configuration file')
  parser.add_argument('dot_list', nargs=argparse.REMAINDER)
  args = parser.parse_args()
  main(args)