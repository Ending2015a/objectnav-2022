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
  SemanticMapObserver,
  SemanticWrapper,
  PlannerWrapper,
  CleanObsWrapper,
  AutoResetWrapper,
  SubprocVecEnv,
)

def create_env(habitat_config, config, index=None):
  if index is not None:
    habitat_config.defrost()
    habitat_config.SEED = habitat_config.SEED + index
    habitat_config.freeze()
  if 'vec_envs' in config.envs:
    gpu = config.envs.vec_envs.gpus[index]
    import torch
    torch.cuda.set_device(gpu)
    habitat_config.defrost()
    habitat_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = gpu
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
    env, **config.envs.semantic_map_builder
  )
  env = SemanticMapObserver(
    env, **config.envs.semantic_map_observer
  )
  env = PlannerWrapper(
    env, **config.envs.planner
  )
  env = SemanticMapObserver(
    env, **config.envs.semantic_map_observer2
  )
  # clean observations
  env = CleanObsWrapper(
    env, **config.envs.clean_obs
  )
  # Monitor group
  env = rlchemy.envs.Monitor(
    env, **config.envs.monitor
  )
  rlchemy.envs.TrajectoryRecorder.trajectory_suffix = \
    "ep{episodes:010d}.{start_steps}-{steps}-" + f"id{index}"
  env.add_tool(rlchemy.envs.TrajectoryRecorder(
    **config.envs.trajectory_recorder
  ))
  # auto reset
  env = AutoResetWrapper(env)
  return env

def create_env_fn(index, args):
  def _fn():
    return create_env(*args, index)
  return _fn

def main(args):
  # Load OmegaConf configurations
  configs = []
  if args.config is not None:
    configs.append(kemono.utils.load_config(args.config, resolve=True))
  configs.append(OmegaConf.from_dotlist(args.dot_list))
  config = OmegaConf.merge(*configs)
  OmegaConf.resolve(config)
  # Load habitat configurations
  habitat_config = get_config(config.habitat_config)
  # create env
  if 'vec_envs' in config.envs:
    print('Create vectorized environments')
    env_fns = [
      create_env_fn(i, (habitat_config, config))
      for i in range(config.envs.vec_envs.n_envs)
    ]
    env = SubprocVecEnv(env_fns)
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