# --- built in ---
import functools
import collections
from dataclasses import dataclass
from typing import (
  Any,
  List,
  Dict,
  Tuple,
  Union,
  Callable,
  Optional
)
# --- 3rd party ---
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import rlchemy
from rlchemy import registry
from rlchemy.lib.nets import DelayedModule
import gym
from omegaconf import OmegaConf
# --- my module ---
from kemono.agents.nets import *
from kemono.envs.runner import Runner
from kemono.data.dataset import Dataset, StreamDataset
from kemono.data import RlchemyDynamicStreamProducer
from kemono.data.wrap import (
  ZeroPadChunkStreamWrapper,
  ResetMaskStreamWrapper,
  CombinedStreamProducer,
  MultiprocessStreamProducer
)
from kemono.data.callbacks import StreamManager, StreamDrawer
from kemono.data.sampler import StreamDatasetSampler


class ValueNet(DelayedModule):
  def __init__(
    self,
    out_dim: int,
    backbone: Dict[str, Any],
    hidden_units: int = 1024,
    mlp_units: List[int] = [256],
    skip_conn: bool = True,
    rnn_type: str = 'lstm'
  ):
    super().__init__()
    self.out_dim = out_dim
    self.backbone = AwesomeBackbone(**backbone)
    self.hidden_units = hidden_units
    self.mlp_units = mlp_units
    self.skip_conn = skip_conn
    self.rnn_type = rnn_type
    # ---
    self.output_dim = None
    self._model_rnn = None
    self._model_body = None

  def get_input_shapes(
    self, x: Any, *args, **kwargs
  ):
    return self.backbone.get_input_shapes(x)

  def build(self, input_shapes: torch.Size):
    # built backbone model
    self.backbone.build(input_shapes)
    out_dim = self.out_dim
    # create rnn model
    self._model_rnn = AwesomeRnn(
      self.backbone.output_dim,
      self.hidden_units,
      skip_conn = self.skip_conn,
      rnn_type = self.rnn_type,
      return_history = True
    )
    # create final mlp layers
    self._model_body = AwesomeMlp(
      self.hidden_units,
      mlp_units = self.mlp_units + [self.out_dim]
    )
    self.output_dim = out_dim
    self.mark_as_built()
  
  def forward(
    self,
    x: torch.Tensor,
    states: Tuple[torch.Tensor, ...] = None,
    reset: torch.Tensor = None,
  ):
    x = self.backbone(x)
    x, states, history = self._model_rnn(x, states, reset)
    x = self._model_body(x)
    return x, states, history

  def get_states(self, batch_size=1, device='cuda'):
    return self._model_rnn.get_states(batch_size=batch_size, device=device)

class Agent(nn.Module):
  value_key = 'value'
  def __init__(
    self,
    observation_space: gym.spaces.Space,
    action_space: gym.spaces.Space,
    value: Dict[str, Any]
  ):
    super().__init__()
    self.observation_space = observation_space
    self.action_space = action_space
    self.n_actions = self.action_space.n
    self.value_config = value
    # ---
    self.q1_value = None
    self.q2_value = None
    self.q1_value_tar = None
    self.q2_value_tar = None

  @property
  def device(self):
    p = next(iter(self.parameters()), None)
    return p.device if p is not None else torch.device('cuda')

  def setup_model(self):
    self.q1_value = ValueNet(
      out_dim = self.n_actions,
      **self.value_config
    )
    self.q2_value = ValueNet(
      out_dim = self.n_actions,
      **self.value_config
    )
    self.q1_value_tar = ValueNet(
      out_dim = self.n_actions,
      **self.value_config
    )
    self.q2_value_tar = ValueNet(
      out_dim = self.n_actions,
      **self.value_config
    )
    for param in self.q1_value_tar.parameters():
      param.requires_grad = False
    for param in self.q2_value_tar.parameters():
      param.requires_grad = False
    input_tensors = rlchemy.utils.input_tensor(self.observation_space)
    input_tensors = self.proc_observation(input_tensors)
    self.q1_value(input_tensors)
    self.q2_value(input_tensors)
    self.q1_value_tar(input_tensors)
    self.q2_value_tar(input_tensors)

  def setup(self):
    self.setup_model()
    return self

  def proc_observation(
    self,
    x: Dict[str, torch.Tensor]
  ):
    def preprocess_obs(obs_and_space):
      obs, space = obs_and_space
      # we only preprocess Box spaces, e.g. images, vectors
      if isinstance(space, gym.spaces.Box):
        obs = rlchemy.utils.preprocess_obs(obs, space)
      return obs

    with torch.no_grad():
      # process nestedly
      return rlchemy.utils.map_nested_tuple(
        (x, self.observation_space),
        op = preprocess_obs,
        sortkey = True
      )

  def forward(
    self,
    x: Dict[str, torch.Tensor],
    states = None,
    reset = None,
    proc_obs: bool = True,
    det: bool = False
  ):
    """Forward agent, predict actions
    the input tensor `x` expecting (seq, batch, ...)

    Args:
        x (Dict[str, torch.Tensor]): _description_
        states (_type_, optional): _description_. Defaults to None.
        reset (_type_, optional): _description_. Defaults to None.
        proc_obs (bool, optional): _description_. Defaults to True.
        det (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    if proc_obs:
      x = self.proc_observation(x)
    # rgb expecting (..., batch, c, h, w)
    rgb_tensor = x['rgb']
    batch = rgb_tensor.shape[-4]
    if states is None:
      states = self.get_states(batch, device=rgb_tensor.device)
    val, vf_states, vf_history = self.q1_value(
      x,
      states = states[self.value_key],
      reset = reset
    )
    dist = rlchemy.prob.Categorical(val)
    if det:
      act = dist.mode()
    else:
      act = dist.sample()
    states = {self.value_key: vf_states}
    history = {self.value_key: vf_history}
    return act, states, history

  @torch.no_grad()
  def _predict(
    self,
    x: np.ndarray,
    states = None,
    reset = None,
    **kwargs
  ):
    # expecting x (...) or (batch, ...)
    is_batch = len(x['rgb'].shape) > 3
    if not is_batch:
      expand_op = lambda x: np.expand_dims(x, axis=0)
      x = rlchemy.utils.map_nested(x, expand_op)
    # predict actions
    x = rlchemy.utils.nested_to_tensor(x, device=self.device)
    outputs, states, *_ = self(x, states, reset, **kwargs)
    outputs = rlchemy.utils.nested_to_numpy(outputs)
    if not is_batch:
      squeeze_op = lambda x: np.squeeze(x, axis=0)
      outputs = rlchemy.utils.map_nested(outputs, squeeze_op)
    # detach recurrent states
    states = rlchemy.utils.map_nested(states, lambda s: s.detach())
    return outputs, states

  def predict(
    self,
    x,
    states = None,
    reset = None,
    proc_obs: bool = True,
    det: bool = True
  ):
    return self._predict(
      x,
      states = states,
      reset = reset,
      proc_obs = proc_obs,
      det = det
    )

  def get_states(self, batch_size=1, device='cuda'):
    return {
      self.value_key: self.value.get_states(batch_size, device)
    }

  def update_target(self, tau=1.0):
    rlchemy.utils.soft_update(
      self.q1_value_tar.parameters(),
      self.q1_value.parameters(),
      tau = tau
    )
    rlchemy.utils.soft_update(
      self.q2_value_tar.parameters(),
      self.q2_value.parameters(),
      tau = tau
    )

@registry.register.kemono_agent('softq')
class SoftQ(pl.LightningModule):
  def __init__(
    self,
    config: Dict[str, Any],
    env: Optional[gym.Env] = None,
    observation_space: Optional[gym.spaces.Space] = None,
    action_space: Optional[gym.spaces.Space] = None,
  ):
    super().__init__()
    # initialize
    self.env = None
    self.observation_space = None
    self.action_space = None
    self.agnet = None
    self.log_alpha = None
    self._train_sampler = None
    self._train_runner = None

    # initialize lightning module
    self.config = OmegaConf.create(config)
    self.set_env(env=env)
    observation_space = self.observation_space
    action_space = self.action_space
    # initialize
    self.save_hyperparameters(ignore='env')
    self.automatic_optimization = False
    self.setup_model()

  def set_env(self, env):
    self.env = env
    self.observation_space = env.observation_space
    self.action_space = env.action_space

  def setup_model(self):
    self.agent = Agent(
      observation_space = self.observation_space,
      action_space = self.action_space,
      value = self.config.value
    )
    self.agent.setup()
    self.agent.to(device='cuda')
    self.agent.update_target(tau=1.0)
    # setup entropy coefficient
    self.register_buffer(
      'log_alpha',
      torch.tensor(np.log(self.config.init_alpha), dtype=torch.float32)
    )

  @staticmethod
  def make_stream_producer(stream_configs: Dict[str, Any]):
    producers = []
    for key, config in stream_configs.items():
      callbacks = []
      if 'manager' in config:
        callbacks.append(StreamManager(**config.manager))
      if 'drawer' in config:
        callbacks.append(StreamDrawer(**config.drawer))
      producer = RlchemyDynamicStreamProducer(
        callbacks = callbacks,
        **config.producer
      )
      producer = ResetMaskStreamWrapper(producer)
      producer = ZeroPadChunkStreamWrapper(
        producer,
        **config.zero_pad
      )
      producers.append(producer)
    producer = CombinedStreamProducer(producers, shuffle=True)
    return producer

  def setup_train(self):
    # setup stream producer & data sampler
    dataset_config = self.config.train_dataset
    producer = self.make_stream_producer(dataset_config.stream_producer)
    if 'multiprocess' in dataset_config:
      producer = MultiprocessStreamProducer(
        producer,
        functools.partial(
          self.make_stream_producer,
          dataset_config.stream_producer
        ),
        **dataset_config.multiprocess
      )
      if 'prefetch' in dataset_config:
        producer.prefetch(dataset_config.prefetch)
    dataset = StreamDataset(
      producer,
      **dataset_config.dataset
    )
    self._train_sampler = StreamDatasetSampler(
      dataset,
      producer,
      **dataset_config.sampler
    )
    # setup environment runner
    if hasattr(self.env, 'n_envs'):
      runner_class = VecRunner
    else:
      runner_class = Runner
    self._train_runner = runner_class(
      env = self.env,
      agent = self,
      **self.config.runner
    )

  def setup(self, stage: str):
    # setup env, model, config here
    # stage: either 'fit', 'validate'
    # 'test' or 'predict'
    if stage == 'fit':
      self.setup_train()

  def train_batch_fn(self):
    # sample n steps for every epoch
    self._train_runner.collect(
      n_steps = self.config.n_steps
    )
    # generate n batches for every epoch
    for _ in range(self.config.n_gradsteps):
      batch, caches = self._train_sampler.sample()
      if caches is None:
        states = self.get_states(
          batch_size = self.config.batch_size
        )
      else:
        states = caches['states']
      batch['states'] = states
      yield batch

  def get_states(self, batch_size=1, device='cuda'):
    return self.agent.get_states(batch_size, device=device)

  def configure_optimizers(self):
    value_optim = torch.optim.Adam(
      list(self.agent.q1_value.parameters()) +
        list(self.agent.q2_value.parameters()),
      lr = self.config.value_lr
    )
    return value_optim

  def forward(
    self,
    x: torch.Tensor,
    states: Optional[Any] = None,
    reset: Optional[Any] = None,
    proc_obs: bool = True,
    det: bool = False
  ):
    return self.agent.predict(
      x,
      states = states,
      reset = reset,
      proc_obs = proc_obs,
      det = det
    )

  def predict(
    self,
    x: Any,
    states: Optional[Any] = None,
    reset: Optional[Any] = None,
    proc_obs: bool = True,
    det: bool = True
  ) -> Tuple[Any, Tuple[torch.Tensor, ...]]:
    return self.agent.predict(
      x,
      states = states,
      reset = reset,
      proc_obs = proc_obs,
      det = det
    )

  def training_step(self, batch, batch_idx):
    value_optim = self.optimizers()
    loss_dict, next_states = self._compute_losses(batch)
    q1_loss = loss_dict['q1_loss']
    q2_loss = loss_dict['q2_loss']
    # update value, q-values
    value_optim.zero_grad()
    self.manual_backward(q1_loss+q2_loss)
    value_optim.step()
    # update caches
    self._train_sampler.cache(states=next_states)
    self.agent.update_target(self.config.tau)

    self.log_dict({
        f'train/{key}': value
        for key, value in loss_dict.items()
      },
      on_step = False,
      on_epoch = True,
      sync_dist = True
    )
    self._train_runner.log_dict(scope='train')
    return loss_dict

  def train_dataloader(self):
    dataset = Dataset.from_generator(self.train_batch_fn)
    # set batch_size to None to enable manual batching
    return DataLoader(dataset=dataset, batch_size=None)

  def _compute_target_value(
    self,
    next_q,
    rew,
    done
  ):
    """Compute target values
    SAC:
      V = E_a'~pi[Q(s', a') - alpha * log pi(a'|s')]
    
    Soft Q:
      V = logsumexp_a'(Q(s', a')/alpha) * alpha

    Args:
      next_q (torch.Tensor): Q values of next steps (seq, b, act)
      next_logp (torch.Tensor): Log probability of next steps (seq, b, act)
      rew (torch.Tensor): reward (seq, b)
      done (torch.Tensor): done (seq, b)

    Returns:
      target values (seq, b)
    """
    with torch.no_grad():
      alpha = self.log_alpha.exp()
      next_vf = alpha * torch.logsumexp(next_q/alpha, dim=-1) # (seq, b)
      y = rew + (1.-done) * self.config.gamma * next_vf
    return y

  def _compute_losses(self, batch):
    """
    Jq1 = 0.5 * mse(q1, rew + (1-done) * gamma * vf)
    Jq2 = 0.5 * mse(q1, rew + (1-done) * gamma * vf)

    Args:
        batch (_type_): _description_
    """
    # preprocess batch
    with torch.no_grad():
      obs = self.agent.proc_observation(batch['obs'])
      next_obs = self.agent.proc_observation(batch['next_obs'])
    act = batch['act'].to(dtype=torch.int64).unsqueeze(-1) # (seq, b, 1)
    rew = batch['rew'].to(dtype=torch.float32) # (seq, b)
    done = batch['done'].to(dtype=torch.float32) # (seq, b)
    reset = batch['reset'].to(dtype=torch.float32) # (seq, b)
    mask = batch['mask'].to(dtype=torch.float32) # (seq, b)
    states = batch['states']
    # [(b, rec), (b, rec)]
    states_vf = states[self.agent.value_key]
    # forward q values
    q1, next_states_vf, history_q1 = self.agent.q1_value(
      obs,
      states = states_vf,
      reset = reset
    ) # (seq, b, act)
    q2, _, history_q2 = self.agent.q2_value(
      obs,
      states = states_vf,
      reset = reset
    ) # (seq, b, act)
    # forward target values and policies
    with torch.no_grad():
      # slice next states
      _slice_op = lambda x: x[0]
      # [(b, rec), (b, rec)]
      # forward target values
      next_0 = rlchemy.utils.map_nested(history_q1, op=_slice_op)
      next_q1, _, _ = self.agent.q1_value_tar(
        next_obs,
        states = next_0,
        reset = done
      ) # (seq, b, act)
      next_0 = rlchemy.utils.map_nested(history_q2, op=_slice_op)
      next_q2, _, _ = self.agent.q2_value_tar(
        next_obs,
        states = next_0,
        reset = done
      ) # (seq, b, act)
      next_q = torch.min(next_q1, next_q2)
    # calculate q losses
    y = self._compute_target_value(next_q, rew, done)
    q1 = q1.gather(-1, act).squeeze(-1) # (seq, b)
    q2 = q2.gather(-1, act).squeeze(-1) # (seq, b)
    q1_loss = rlchemy.loss.regression(
      (y-q1) * mask,
      loss_type = self.config.loss_type,
      delta = self.config.huber_delta
    ).mean()
    q2_loss = rlchemy.loss.regression(
      (y-q2) * mask,
      loss_type = self.config.loss_type,
      delta = self.config.huber_delta
    ).mean()
    with torch.no_grad():
      # additional informations to logging
      q = torch.min(q1, q2)
      norm_q = torch.softmax(q, dim=-1)
      log_q = torch.log(norm_q + 1e-7)
      entropy_q = -1.0 * torch.sum(norm_q * log_q, dim=-1).mean()
      mean_q = torch.mean(q)
      alpha = self.log_alpha.exp()
    loss_dict = {
      'q1_loss': q1_loss,
      'q2_loss': q2_loss,
      'entropy_q': entropy_q,
      'mean_q': mean_q,
      'alpha': alpha
    }
    next_states = {
      self.agent.value_key: next_states_vf
    }
    return loss_dict, next_states

