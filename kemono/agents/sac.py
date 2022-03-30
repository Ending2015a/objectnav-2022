# --- built in ---
import collections
from dataclasses import dataclass
from re import L
from typing import (
  Any,
  List,
  Dict,
  Tuple,
  Union,
  Optional
)
# --- 3rd party ---
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.models
import pytorch_lightning as pl
import einops
import rlchemy
from rlchemy.nets import DelayedModule
# --- my module ---

class PretrainedResNet(nn.Module):
  def __init__(self):
    """Resnet 50
    | names   | output shape      | ratio                 |
    |---------|-------------------|-----------------------|
    | inputs  | (1, 3, 256, 256)  | (1, 3, H, W)          |
    | conv1   | (1, 64, 128, 128) | (1, 64, H/2, W/2)     |
    | bn1     | (1, 64, 128, 128) | (1, 64, H/2, W/2)     |
    | relu    | (1, 64, 128, 128) | (1, 64, H/2, W/2)     |
    | maxpool | (1, 64, 64, 64)   | (1, 64, H/2, W/2)     |
    | layer1  | (1, 256, 64, 64)  | (1, 256, H/4, W/4)    |
    | layer2  | (1, 512, 32, 32)  | (1, 512, H/8, W/8)    |
    | layer3  | (1, 1024, 16, 16) | (1, 1024, H/16, W/16) |
    | layer4  | (1, 2048, 8, 8)   | (1, 2048, H/32, W/32) |
    | avgpool | (1, 2048, 1, 1)   | (1, 2048, 1, 1)       |
    | fc      | (1, 1000)         |                       |
    """
    super().__init__()
    self.resnet = torchvision.models.resnet50(pretrained=True)

  def forward(self, x):
    return self.resnet(x)

class AwesomeFeatureExtractor(nn.Module):
  def __init__(
    self,
    pretrained: bool = False
  ):
    super().__init__()
    resnet = torchvision.models.resnet50(pretrained=pretrained)
    # rgb branch
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    self.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
    # depth branch
    self.conv1_d = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    self.bn1_d = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
    # remain layers
    self.relu = resnet.relu
    self.maxpool = resnet.maxpool
    self.layer1 = resnet.layer1
    self.layer2 = resnet.layer2
    self.layer3 = resnet.layer3
    self.layer4 = resnet.layer4
    self.avgpool = resnet.avgpool
    if pretrained:
      # load pretrained weights
      self.conv1.load_state_dict(resnet.conv1.state_dict())
      self.bn1.load_state_dict(resnet.bn1.state_dict())
      self.conv1_d.load_state_dict(
        {k: torch.mean(v, 1, keepdim=True).data}
        for k, v in resnet.conv1.state_dict().items()
      )
      self.bn1.load_state_dict(resnet.bn1.state_dict())
  
  def forward(self, x):
    def forward_net(x, conv, bn1):
      batch = x.shape[:-3]
      x = x.flatten(0, -4)
      x = conv(x)
      x = bn1(x)
      x = self.relu(x)
      x = self.maxpool(x)
      x = self.layer1(x)
      x = self.layer2(x)
      x = self.layer3(x)
      x = self.layer4(x)
      x = self.avgpool(x)
      x = x.flatten(-3, -1)
      x = x.reshape(batch + (-1,))
      return x
    # (..., 2048)
    x_rgb = forward_net(x['rgb'], self.conv1, self.bn1)
    x_depth = forward_net(x['depth'], self.conv1_d, self.bn1_d)
    # (..., 4096)
    x = torch.cat((x_rgb, x_depth), dim=-1)
    return x

class AwesomeLSTM(nn.Module):
  def __init__(
    self,
    dim: int = 1024,
    units: int = 1024
  ):
    super().__init__()
    self.units = units
    self.input_dim = dim
    self.output_dim = units
    self.cell = nn.LSTMCell(dim, units)
  
  def forward(
    self,
    x: torch.Tensor,
    states: Tuple[torch.Tensor, ...],
    reset: Optional[torch.Tensor]
  ):
    """Forward LSTM Cell

    Args:
      x (torch.Tensor): input tensor, expecting (seq, b, dim)
      states (Tuple[torch.Tensor, ...]): initial states
      reset (Optional[torch.Tensor]): sequence of reset signal to reset
        the input hidden/cell states.

    Returns:
      torch.Tensor: sequence outputs
      Tuple[Tuple[torch.Tensor, ...], ...]: sequence of states
    """
    seq = x.shape[0]
    hx, cx = states
    out = []
    states = []
    mask = 1.0 - reset
    for i in range(seq):
      # reset hidden/cell states to zero if reset = True
      hx, cx = self.cell(x[i], (hx * mask[i], cx * mask[i]))
      out.append(hx)
      states.append((hx, cx))
    return torch.stack(out, dim=0), tuple(states)

  def get_states(self, batch_size=1, device='cuda'):
    return (
      torch.zeros((batch_size, self.units), device=device),
      torch.zeros((batch_size, self.units), device=device)
    )


class AwesomeGRU(nn.Module):
  def __init__(
    self,
    dim: int = 1024,
    units: int = 1024
  ):
    super().__init__()
    self.units = units
    self.input_dim = dim
    self.output_dim = units
    self.cell = nn.GRUCell(dim, units)
  
  def forward(
    self,
    x: torch.Tensor,
    states: Tuple[torch.Tensor, ...],
    reset: Optional[torch.Tensor]
  ):
    """Forward GRU Cell

    Args:
      x (torch.Tensor): input tensor, expecting (seq, b, dim)
      states (Tuple[torch.Tensor, ...]): initial states
      reset (Optional[torch.Tensor]): sequence of reset signal to reset
        the input hidden/cell states, expecting (seq, b)

    Returns:
      torch.Tensor: sequence outputs
      Tuple[Tuple[torch.Tensor, ...], ...]: sequence of states
    """
    seq = x.shape[0]
    hx = states[0]
    # (seq, b) to (seq, b, 1)
    if len(reset.shape) < 3:
      reset = reset.unsqueeze(reset, -1)
    out = []
    states = []
    mask = 1.0 - reset
    for i in range(seq):
      hx = self.cell(x[i], hx * mask[i])
      out.append(hx)
      states.append((hx,))
    return torch.stack(out, dim=0), tuple(states)

  def get_states(self, batch_size=1, device='cuda'):
    return (
      torch.zeros((batch_size, self.units), device=device),
    )


class AwesomeRnn(nn.Module):
  def __init__(
    self,
    dim: int = 1024,
    units: int = 1024,
    skip_conn: bool = True,
    rnn_type: str = 'lstm',
    return_history: bool = True
  ):
    super().__init__()
    self.units = units
    self.skip_conn = skip_conn
    self.rnn_type = rnn_type.lower()
    self.return_history = return_history
    assert self.rnn_type in ['lstm', 'gru']
    if self.rnn_type == 'lstm':
      self.cell = AwesomeLSTM(dim, units)
    elif self.rnn_type == 'gru':
      self.cell = AwesomeGRU(dim, units)
    else:
      raise ValueError(f"Unknown rnn type: {self.rnn_type}")

  def forward(
    self,
    x: torch.Tensor,
    states: Tuple[torch.Tensor, ...] = None,
    reset: torch.Tensor = None
  ):
    """Forward rnn module

    Args:
      x (torch.Tensor): input tensor, expecting (seq, b, dim), (b, dim)
      states (Tuple[torch.Tensor, ...]): [(batch, out_dim), ...]
      reset (torch.Tensor, optional): reset signal, expecting (seq, b) or
        (b,). Defaults to None.

    Returns:
      Tuple[torch.Tensor, Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        torch.Tensor: output tensor
        Tuple[torch.Tensor]: next states
        Tuple[torch.Tensor]: final states
    """
    # padding x from (b, dim) to (seq, b, dim)
    orig_shape = x.shape
    if len(x.shape) < 3:
      x = torch.unsqueeze(x, 0)
    seq, batch = x.shape[0:2]
    # create states
    if states is None:
      states = self.get_states(batch, device=x.device)
    # create reset signal tensor
    if reset is None:
      reset = torch.tensor(0)
    reset = reset.type_as(x)
    seq, batch = x.shape[0:2]
    reset = torch.broadcast_to(reset, (seq, batch))
    # out: sequence of rnn outputs (seq, b, dim)
    # new_states: sequence of hidden/cell states
    #   Tuple[Tuple[(b, dim), ...], ...]
    out, states_seq = self.cell(
      x,
      states = states,
      reset = reset
    )
    if self.skip_conn:
      out = out + x
    # unpad seq dim
    if len(orig_shape) < 3:
      out = torch.squeeze(out, 0)
    final_states = states_seq[-1]
    if self.return_history:
      _stack_op = lambda xs: torch.stack(xs, dim=0)
      history = rlchemy.utils.map_nested_tuple(states_seq, op=_stack_op)
      return out, final_states, history
    return out, final_states

  def get_states(self, batch_size=1, device='cuda'):
    return self.cell.get_states(batch_size=batch_size, device=device)


class CategoricalPolicyNet(DelayedModule):
  def __init__(
    self,
    n_actions: Optional[int] = None,
    hidden_units: int = 1024,
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
    self.hidden_units = hidden_units
    self.skip_conn = skip_conn
    self.rnn_type = rnn_type
    # ---
    self.input_dim = None
    self.output_dim = None
    self._model = None

  def build(self, input_shape: torch.Size):
    # expecting input_shape = (seq, b, 4096)
    in_dim = input_shape[-1]
    out_dim = self.n_actions
    self._model_head = nn.Sequential(
      nn.Linear(in_dim, self.hidden_units)
    )
    # expecting (seq, b, 1024)
    self._model_rnn = AwesomeRnn(
      self.hidden_units,
      self.hidden_units,
      skip_conn = self.skip_conn,
      rnn_type = self.rnn_type,
      return_history = True
    )
    self._model_body = nn.Sequential(
      nn.Linear(self.hidden_units, self.hidden_units//4),
      nn.LeakyReLU(0.1, inplace=True),
      nn.Linear(self.hidden_units//4, out_dim)
    )
    self.input_dim = in_dim
    self.output_dim = out_dim
    self.mark_as_built()

  def forward(
    self,
    x: torch.Tensor,
    states: Tuple[torch.Tensor, ...],
    reset: torch.Tensor
  ):
    x = self._model_head(x)
    x, states, history = self._model_rnn(x, states, reset)
    x = self._model_body(x)
    dist = rlchemy.prob.Categorical(x)
    return dist, states, history

  def get_states(self, batch_size=1, device='cuda'):
    return self._model_rnn.get_states(batch_size=batch_size, device=device)

class ValueNet(DelayedModule):
  def __init__(
    self,
    out_dim: int = 1,
    hidden_units: int = 1024,
    skip_conn: bool = True,
    rnn_type: str = 'lstm'
  ):
    super().__init__()
    self.out_dim = out_dim
    self.hidden_units = hidden_units
    self.skip_conn = skip_conn
    self.rnn_type = rnn_type
    # ---
    self.input_dim = None
    self.output_dim = None
    self._model = None

  def build(self, input_shape: torch.Size):
    # expecting input_shape = (seq, batch, 4096)
    in_dim = input_shape[-1]
    out_dim = self.out_dim
    self._model_head = nn.Sequential(
      nn.Linear(in_dim, self.hidden_units)
    )
    # expecting (seq, b, 1024)
    self._model_rnn = AwesomeRnn(
      self.hidden_units,
      self.hidden_units,
      skip_conn = self.skip_conn,
      rnn_type = self.rnn_type,
      return_history = True
    )
    self._model_body = nn.Sequential(
      nn.Linear(self.hidden_units, self.hidden_units//4),
      nn.LeakyReLU(0.1, inplace=True),
      nn.Linear(self.hidden_units//4, out_dim)
    )
    self.input_dim = in_dim
    self.output_dim = out_dim
    self.mark_as_built()
  
  def forward(
    self,
    x: torch.Tensor,
    states: Tuple[torch.Tensor, ...],
    reset: torch.Tensor,
  ):
    x = self._model_head(x)
    x, states, history = self._model_rnn(x, states, reset)
    x = self._model_body(x)
    return x, states, history

  def get_states(self, batch_size=1, device='cuda'):
    return self._model_rnn.get_states(batch_size=batch_size, device=device)


class Agent(nn.Module):
  def __init__(
    self,
    input_shapes: Dict[str, Any],
    n_actions: int,
    pretrained_resnet: bool = False,
    hidden_units: int = 1024,
    skip_conn: bool = True,
    rnn_type: str = 'gru'
  ):
    super().__init__()
    self.input_shapes = input_shapes
    for expect in ['rgb', 'depth', 'goal', 'pose']:
      assert expect in self.input_shapes, f'Expecting key {expect}'
    self.n_actions = n_actions
    self.pretrained_resnet = pretrained_resnet
    self.hidden_units = hidden_units
    self.skip_conn = skip_conn
    self.rnn_type = rnn_type

    self.preprocess_net = None
    self.policy = None
    self.values = None
  
  @property
  def device(self):
    p = next(iter(self.parameters()), None)
    return p.device if p is not None else torch.device('cpu')

  def setup_model(self):
    self.preprocess_net = AwesomeFeatureExtractor(
      self.pretrained_resnet
    )
    self.policy = CategoricalPolicyNet(
      n_actions = self.n_actions,
      hidden_units = self.hidden_units,
      skip_conn = self.skip_conn,
      rnn_type = self.rnn_type
    )
    self.q_value1 = ValueNet(
      out_dim = self.n_actions,
      hidden_units = self.hidden_units,
      skip_conn = self.skip_conn,
      rnn_type = self.rnn_type
    )
    self.q_value2 = ValueNet(
      out_dim = self.n_actions,
      hidden_units = self.hidden_units,
      skip_conn = self.skip_conn,
      rnn_type = self.rnn_type
    )
    self.value = ValueNet(
      out_dim = 1,
      hidden_units = self.hidden_units,
      skip_conn = self.skip_conn,
      rnn_type = self.rnn_type
    )
    self.value_tar = ValueNet(
      out_dim = 1,
      hidden_units = self.hidden_units,
      skip_conn = self.skip_conn,
      rnn_type = self.rnn_type
    )
    _tensor_op = lambda shape: torch.zeros((1,) + shape)
    input_tensors = rlchemy.utils.map_nested(self.input_shapes, _tensor_op)
    self.forward_policy(input_tensors)
    self.forward_q_values(input_tensors)

  def setup(self):
    self.setup_model()

  def proc_observations(
    self,
    x: Dict[str, torch.Tensor]
  ):
    # expecting x: (..., c, h, w)
    with torch.no_grad():
      x = self.preprocess_net(x)
    return x

  def forward_policy(
    self,
    x: Dict[str, torch.Tensor],
    states = None,
    reset = None,
    proc_obs: bool = True
  ):
    if proc_obs:
      x = self.proc_observations(x)
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
      x = self.proc_observations(x)
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
      x = self.proc_observations(x)
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
    if proc_obs:
      x = self.proc_observations(x)
    # expecting (seq, batch, 4069)
    batch = x.shape[-2]
    if states is None:
      states = self.get_states(batch, device=x.device)
    dist, pi_states, pi_history = self.forward_policy(
      x,
      states = states['policy'],
      reset = reset,
      proc_obs = False
    )
    val, vf_states, vf_history = self.forward_value(
      x,
      states = states['value'],
      reset = reset,
      proc_obs = False
    )
    if det:
      act = dist.mode()
    else:
      act = dist.sample()
    states = {'policy': pi_states, 'value': vf_states}
    history = {'policy': pi_history, 'value': vf_history}
    return act, states, val, history

  @torch.no_grad()
  def _predict(
    self,
    x: np.ndarray,
    states = None,
    reset = None,
    det: bool = True
  ):
    is_batch = len(x['rgb'].shape) > 3
    if not is_batch:
      expand_op = lambda x: np.expand_dims(x, axis=0)
      x = rlchemy.utils.map_nested(x, expand_op)
    # predict actions
    x = rlchemy.utils.nested_to_tensor(x, device=self.device)
    outputs, states, *_ = self(x, states, reset, det)
    outputs = rlchemy.utils.nested_to_numpy(outputs)
    if not is_batch:
      squeeze_op = lambda x: np.squeeze(x, axis=0)
      outputs = rlchemy.utils.map_nested(outputs, squeeze_op)
    return outputs, states.detach()

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
      'policy': self.policy.get_states(batch_size, device),
      'values': self.value.get_states(batch_size, device)
    }

  def update_target(self, tau=1.0):
    tar = list(self.value_tar.parameters())
    src = list(self.value.parameters())
    rlchemy.utils.soft_update(tar, src, tau=tau)

class SAC(pl.LightningModule):
  def __init__(
    self,
    input_shapes: Dict[Any, Tuple[int]],
    n_actions: int,
    pretrained_resnet: bool = True,
    hidden_units: int = 1024,
    skip_conn: bool = True,
    rnn_type: str = 'gru'
  ):
    super().__init__()
    # initialize

    self.save_hyperparameters()
    self.automatic_optimization = False
    self.setup_model()

  def setup_model(self):
    # TODO: setup stream
    self.agent = Agent(
      input_shapes = self.input_shapes,
      n_actions = self.n_actions,
      pretrained_reset = self.pretrained_reset,
      hidden_units = self.hidden_units,
      skip_conn = self.skip_conn,
      rnn_type = self.rnn_type
    )
    self.agent.setup()
    self.agent.to(device='cuda')
    self.agent.update_target(tau=1.0)
    # setup entropy coefficient
    self.log_alpha = nn.Parameter(
      torch.Tensor(0.0, dtype=torch.float32)
    )
    # setup target entropy: log(dim(A))
    if self.target_ent is None:
      self.target_ent = np.log(self.n_actions)
    entropy_scale = 0.2
    self.target_ent = entropy_scale * float(np.asarray(self.target_ent).item())

  def setup(self, stage: str):
    # setup env, model, config here
    # stage: either 'fit', 'validate'
    # 'test' or 'predict'
    if stage == 'fit':
      self.setup_train()

  def train_batch_fn(self):
    # sample n steps for every epoch
    #TODO
    self._train_runner.collect(
      random = False,
      n_episodes = self.n_episodes,
      nowait = True
    )
    # generate n batches for every epoch
    for _ in range(self.n_gradsteps):
      batch = self.sampler()
      yield batch

  def get_states(self, batch_size=1, device='cuda'):
    return self.agent.get_states(batch_size, device=device)

  def configure_optimizers(self):
    policy_optim = torch.optim.Adam(
      self.agent.policy.parameters(),
      lr = self.policy_lr
    )
    value_optim = torch.optim.Adam(
      list(self.agent.value.parameters()) +
      list(self.agent.q_value1.parameters()) +
      list(self.agent.q_value2.parameters()),
      lr = self.value_lr
    )
    alpha_optim = torch.optim.Adam(
      [self.log_alpha],
      lr = self.alpha_lr
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
    return loss_dict

  def train_dataloader(self):
    # TODO stream
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
    states_pi = states['policy']
    states_vf = states['value']
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
      'policy': next_states_pi,
      'value': next_states_vf
    }
    return loss_dict, next_states

