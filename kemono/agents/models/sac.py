# --- built in ---
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
from rlchemy.lib.nets import DelayedModule
import gym
from omegaconf import OmegaConf
# --- my module ---
from kemono.agents.nets import *


class CategoricalPolicyNet(DelayedModule):
  def __init__(
    self,
    n_actions: int,
    backbone: Dict[str, Any],
    hidden_units: int = 1024,
    mlp_units: List[int] = [256],
    skip_conn: bool = True,
    rnn_type: str = 'lstm'
  ):
    """Categorical policy for discrete actions

    Args:
      dim (Optional[int], optional): input dimension. Defaults to None.
      n_actions (Optional[int], optional): number of discrete actions.
        Defaults to None.
    """
    super().__init__()
    self.n_actions = n_actions
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
    out_dim = self.n_actions
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
      mlp_units = self.mlp_units + [self.n_actions]
    )
    self.output_dim = out_dim
    self.mark_as_built()

  def forward(
    self,
    x: torch.Tensor,
    states: Tuple[torch.Tensor, ...] = None,
    reset: torch.Tensor = None
  ):
    x = self.backbone(x)
    x, states, history = self._model_rnn(x, states, reset)
    x = self._model_body(x)
    dist = rlchemy.prob.Categorical(x)
    return dist, states, history

  def get_states(self, batch_size=1, device='cuda'):
    return self._model_rnn.get_states(batch_size=batch_size, device=device)

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
  policy_key = 'policy'
  value_key = 'value'
  def __init__(
    self,
    observation_space: gym.spaces.Space,
    action_space: gym.spaces.Space,
    policy: Dict[str, Any],
    value: Dict[str, Any]
  ):
    super().__init__()
    self.observation_space = observation_space
    self.action_space = action_space
    self.n_actions = self.action_space.n
    self.policy_config = policy
    self.value_config = value
    # ---
    self.policy = None
    self.q_value1 = None
    self.q_value2 = None
    self.value = None
    self.value_tar = None

  @property
  def device(self):
    p = next(iter(self.parameters()), None)
    return p.device if p is not None else torch.device('cuda')

  def setup_model(self):
    self.policy = CategoricalPolicyNet(
      n_actions = self.n_actions,
      **self.policy_config
    )
    self.q_value1 = ValueNet(
      out_dim = self.n_actions,
      **self.value_config
    )
    self.q_value2 = ValueNet(
      out_dim = self.n_actions,
      **self.value_config
    )
    self.value = ValueNet(
      out_dim = 1,
      **self.value_config
    )
    self.value_tar = ValueNet(
      out_dim = 1,
      **self.value_config
    )
    input_tensors = rlchemy.utils.input_tensor(self.observation_space)
    self.forward_policy(input_tensors)
    self.forward_value(input_tensors)
    self.forward_q_values(input_tensors)


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

  def forward_policy(
    self,
    x: Dict[str, torch.Tensor],
    states = None,
    reset = None,
    proc_obs: bool = True
  ):
    if proc_obs:
      x = self.proc_observation(x)
    dist, states, history = self.policy(x, states=states, reset=reset)
    return dist, states, history

  def forward_value(
    self,
    x: Dict[str, torch.Tensor],
    states = None,
    reset = None,
    proc_obs: bool = True
  ):
    if proc_obs:
      x = self.proc_observation(x)
    x, states, history = self.value(x, states=states, reset=reset)
    return x, states, history

  def forward_q_values(
    self,
    x: Dict[str, torch.Tensor],
    states = None,
    reset = None,
    proc_obs: bool = True
  ):
    if proc_obs:
      x = self.proc_observation(x)
    q1, _, _ = self.q_value1(x, states=states, reset=reset)
    q2, _, _ = self.q_value2(x, states=states, reset=reset)
    return q1, q2

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
    dist, pi_states, pi_history = self.forward_policy(
      x,
      states = states[self.policy_key],
      reset = reset,
      proc_obs = False
    )
    val, vf_states, vf_history = self.forward_value(
      x,
      states = states[self.value_key],
      reset = reset,
      proc_obs = False
    )
    if det:
      act = dist.mode()
    else:
      act = dist.sample()
    states = {
      self.policy_key: pi_states,
      self.value_key: vf_states
    }
    history = {
      self.policy_key: pi_history,
      self.value_key: vf_history
    }
    return act, states, val, history

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
      self.policy_key: self.policy.get_states(batch_size, device),
      self.value_key: self.value.get_states(batch_size, device)
    }

  def update_target(self, tau=1.0):
    tar = list(self.value_tar.parameters())
    src = list(self.value.parameters())
    rlchemy.utils.soft_update(tar, src, tau=tau)

class SAC(pl.LightningModule):
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
    self.target_ent = None

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
      policy = self.config.policy,
      value = self.config.value
    )
    self.agent.setup()
    self.agent.to(device='cuda')
    self.agent.update_target(tau=1.0)
    # setup entropy coefficient
    self.log_alpha = nn.Parameter(
      torch.tensor(np.log(self.config.init_entcoef),
        dtype=torch.float32)
    )
    # setup target entropy: log(dim(A))
    if self.target_ent is None:
      self.target_ent = np.log(self.action_space.n)
    entropy_scale = self.config.entropy_scale
    self.target_ent = entropy_scale * float(np.asarray(self.target_ent).item())

  def setup_train(self):
    #TODO setup stream provider
    #TODO setup ray runner
    pass


  def setup(self, stage: str):
    # setup env, model, config here
    # stage: either 'fit', 'validate'
    # 'test' or 'predict'
    if stage == 'fit':
      self.setup_train()

  def train_batch_fn(self):
    # sample n steps for every epoch
    # TODO
    self._train_runner.collect(
      n_steps = self.config.n_steps
    )
    # generate n batches for every epoch
    for _ in range(self.config.n_gradsteps):
      batch = self.sampler()
      yield batch

  def get_states(self, batch_size=1, device='cuda'):
    return self.agent.get_states(batch_size, device=device)

  def configure_optimizers(self):
    policy_optim = torch.optim.Adam(
      self.agent.policy.parameters(),
      lr = self.config.policy_lr
    )
    value_optim = torch.optim.Adam(
      list(self.agent.value.parameters()) +
      list(self.agent.q_value1.parameters()) +
      list(self.agent.q_value2.parameters()),
      lr = self.config.value_lr
    )
    alpha_optim = torch.optim.Adam(
      [self.log_alpha],
      lr = self.config.alpha_lr
    )
    return policy_optim, value_optim, alpha_optim

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
    policy_optim, value_optim, alpha_optim = self.optimizers()
    loss_dict, next_states = self._compute_losses(batch, batch_idx)
    pi_loss = loss_dict['pi_loss']
    vf_loss = loss_dict['vf_loss']
    q1_loss = loss_dict['q1_loss']
    q2_loss = loss_dict['q2_loss']
    alpha_loss = loss_dict['alpha_loss']
    # update policy
    policy_optim.zero_grad()
    self.manual_backward(pi_loss)
    policy_optim.step()
    # update value, q-values
    value_optim.zero_grad()
    self.manual_backward(vf_loss+q1_loss+q2_loss)
    value_optim.step()
    # update alpha
    alpha_optim.zero_grad()
    self.manual_backward(alpha_loss)
    alpha_optim.step()
    # TODO: update caches
    self.sampler.update(next_states)
    self.agent.update_target(self.config.tau)
    return loss_dict

  def train_dataloader(self):
    # TODO stream
    # set batch_size to None to enable manual batching
    return DataLoader(dataset=dataset, batch_size=None)

  def _compute_losses(self, batch, states):
    """
    Jpi = alpha * p * logp - p * q
    Jq1 = 0.5 * mse(q1, rew + (1-done) * gamma * vf)
    Jq2 = 0.5 * mse(q1, rew + (1-done) * gamma * vf)
    Jvf = 0.5 * mse(vf, p * min(q1, q2) - alpha * p * logp)
    Jalpha = -alpha * (p * logp + target_ent)

    Jpi
    Jv = Jq1 + Jq2 + Jvf
    Jalpha

    Args:
        batch (_type_): _description_
    """
    # preprocess batch
    with torch.no_grad():
      obs = self.agent.proc_observations(batch['obs'])
      next_obs = self.agent.proc_observations(batch['next_obs'])
    act = batch['act'].to(dtype=torch.int64).unsqueeze(-1)
    rew = batch['rew'].to(dtype=torch.float32)
    done = batch['done'].to(dtype=torch.float32)
    reset = batch['reset'].to(dtype=torch.float32)
    states_pi = states[self.agent.policy_key]
    states_vf = states[self.agent.value_key]
    # forward networks
    alpha = self.log_alpha.exp()
    # forward policy
    dist, next_states_pi, _ = self.agent.policy(
      obs,
      states = states_pi,
      reset = reset
    )
    p = torch.softmax(dist.logits, dim=-1) # (seq, b, act)
    logp = torch.log(p + 1e-7) # (seq, b, act)
    # forward q values
    q1, _, _ = self.agent.q_value1(
      obs,
      states = states_vf,
      reset = reset
    ) # (seq, b, act)
    q2, _, _ = self.agent.q_value2(
      obs,
      states = states_vf,
      reset = reset
    )
    q = torch.min(q1, q2)
    # forward values
    vf, next_states_vf, history_vf = self.agent.value(
      obs,
      states = states_vf,
      reset = reset
    )
    vf = vf.squeeze(-1) # (seq, b)
    # forward target values
    with torch.no_grad():
      # slice next states
      _slice_op = lambda x: x[0]
      next_1 = rlchemy.utils.map_nested(history_vf, op=_slice_op)
      next_vf, _, _ = self.agent.value_tar(
        next_obs,
        states = next_1,
        reset = done
      ) # (seq, b, 1)
      next_vf = next_vf.squeeze(-1) # (seq, b)
    # calculate policy loss
    pi_loss = torch.mean((alpha * p * logp - p * q1.detach()).sum(dim=-1))
    # calculate alpha loss
    with torch.no_grad():
      tar = torch.sum((p * logp + self.target_ent), dim=-1)
    alpha_loss = -1.0 * torch.mean(alpha * tar)
    # calculate q losses
    with torch.no_grad():
      y = rew + (1.-done) * self.gamma * next_vf
    q1 = q1.gather(-1, act).squeeze(-1) # (seq, b)
    q2 = q2.gather(-1, act).squeeze(-1) # (seq, b)
    q1_loss = rlchemy.loss.regression(y-q1, loss_type=self.loss_type)
    q2_loss = rlchemy.loss.regression(y-q2, loss_type=self.loss_type)
    # calculate v loss
    with torch.no_grad():
      y = torch.sum(p * q - alpha * p * logp, dim=-1) # (seq, b)
    vf_loss = rlchemy.loss.regression(y-vf, loss_type=self.loss_type)
    loss_dict = {
      'pi_loss': pi_loss,
      'vf_loss': vf_loss,
      'q1_loss': q1_loss,
      'q2_loss': q2_loss,
      'alpha_loss': alpha_loss
    }
    next_states = {
      self.agent.policy_key: next_states_pi,
      self.agent.value_key: next_states_vf
    }
    return loss_dict, next_states

