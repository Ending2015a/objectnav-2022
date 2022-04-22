# --- built in ---
# --- 3rd party ---
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.strategies.ddp import DDPStrategy
from habitat.config.default import get_config
import rlchemy
from rlchemy import registry
# --- my module ---
import kemono
from kemono.envs import habitat_env
from kemono.envs.wrap import (
  SemanticMapBuilderWrapper,
  SemanticWrapper,
  CleanObsWrapper,
  AutoResetWrapper,
  VecRayRemoteEnv,
)

def create_env(habitat_config, config, index=None):
  if index is not None:
    habitat_config.defrost()
    habitat_config.SEED = habitat_config.SEED + index
    habitat_config.freeze()
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
  # Monitor group
  env = rlchemy.envs.Monitor(
    env,
    **config.envs.monitor
  )
  rlchemy.envs.TrajectoryRecorder.trajectory_suffix = \
    "ep{episodes:010d}.{start_steps}-{steps}-" + f"id{index}"
  env.add_tool(rlchemy.envs.TrajectoryRecorder(
    **config.envs.trajectory_recorder
  ))
  # auto reset
  env = AutoResetWrapper(env)
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
  # create env
  if 'vec_ray_remote' in config.envs:
    print('Creaing VecRayRemoteEnv environment')
    kemono.utils.init_ray()
    env = VecRayRemoteEnv(
      create_env,
      (habitat_config, config),
      **config.envs.vec_ray_remote
    )
  else:
    print('Creaing environment')
    env = create_env(habitat_config, config, index=0)
  # create model & trainer
  agent_name = config.agent.pop('type')
  agent_class = registry.get.kemono_agent(agent_name)
  print(f'Creating {agent_name} model')
  model = agent_class(config.agent, env=env)
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
  env.close()


if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(add_help=True)
  parser.add_argument('--config', type=str, required=True, help='Configuration file')
  parser.add_argument('dot_list', nargs=argparse.REMAINDER)
  args = parser.parse_args()
  main(args)