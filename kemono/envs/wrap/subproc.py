# --- built in ---
import abc
import enum
import multiprocessing
from typing import (
  Any,
  Callable,
  List,
  Optional,
  Tuple,
  Union,
  Dict
)
# --- 3rd party ---
import gym
import numpy as np
import cloudpickle
# --- my module ---
from rlchemy.lib import utils as rl_utils

__all__ = [
  'SubprocVecEnv'
]

# The implementation mainly follows tianshou/venvs.py
# habitat-lab/vector_env.py and stable-baselines3/subproc_vec_env.py
class BaseEnvWorker(metaclass=abc.ABCMeta):
  def __init__(self, env_fn: Callable):
    self._env_fn = env_fn

  def setup(self):
    self.observation_space = self.getattr('observation_space')
    self.action_space      = self.getattr('action_space')
    self.metadata          = self.getattr('metadata')
    self.reward_range      = self.getattr('reward_range')
    self.spec              = self.getattr('spec')

  @abc.abstractmethod
  def getattr(self, attrname: str) -> Any:
    raise NotImplementedError

  @abc.abstractmethod
  def setattr(self, attrname: str, value: Any) -> Any:
    raise NotImplementedError

  @abc.abstractmethod
  def reset(self, **kwargs) -> Any:
    raise NotImplementedError

  @abc.abstractmethod
  def step_async(self, act):
    raise NotImplementedError

  @abc.abstractmethod
  def step_wait(self) -> Any:
    raise NotImplementedError

  @abc.abstractmethod
  def seed(self, seed: int) -> Any:
    self.action_space.seed(seed)

  @abc.abstractmethod
  def render(self) -> Any:
    raise NotImplementedError

  @abc.abstractmethod
  def close_async(self):
    raise NotImplementedError
  
  @abc.abstractmethod
  def close_wait(self):
    raise NotImplementedError

# The implementation mainly follows tianshou/venvs.py
# habitat-lab/vector_env.py and stable-baselines3/subproc_vec_env.py
class BaseVecEnv(gym.Env):
  def __init__(
    self,
    env_fns: List[Callable],
    worker_class: BaseEnvWorker
  ):
    self._worker_class = worker_class
    self.workers = [
      worker_class(fn, worker_id=idx)
      for idx, fn in enumerate(env_fns)
    ]
    self.n_envs = len(self.workers)
    self._closed = False
    # setup workers
    for w in self.workers:
      w.setup()
    # init properties
    self.observation_spaces = [w.observation_space for w in self.workers]
    self.action_spaces      = [w.action_space for w in self.workers]
    self.metadatas          = [w.metadata for w in self.workers]
    self.reward_ranges      = [w.reward_range for w in self.workers]
    self.specs              = [w.spec for w in self.workers]

    self.observation_space = self.observation_spaces[0]
    self.action_space      = self.action_spaces[0]
    self.metadata          = self.metadatas[0]
    self.reward_range      = self.reward_ranges[0]
    self.spec              = self.specs[0]

  @property
  def closed(self) -> bool:
    return self._closed

  def __len__(self) -> int:
    return self.n_envs

  def getattrs(
    self,
    attrname: str,
    id: Optional[List[int]] = None
  ) -> List[Any]:
    """Retrieve attributes from each environment

    Args:
      attrname (str): Attribute name
      id (List[int], optional): Environment indices. Defaults to None.

    Returns:
      list: A list of retrieved attributes
    """
    self._assert_not_closed()
    ids = self._get_ids(id)
    return [self.workers[id].getattr(attrname) for id in ids]

  def setattrs(
    self,
    attrname: str,
    value: Optional[Any] = None,
    values: Optional[List[Any]] = None,
    id: Optional[List[int]] = None
  ) -> List[Any]:
    """Set attributes for each environments

    Args:
      attrname (str): Attribute name
      value (Any, optional): Attribute value. This value will be passed
        to all envs specified by `id`. Defaults to None.
      values (List[Any], optional): A list of attribute values. Defaults to None.
      id (List[int], optional): Environment indices. Defaults to None.
    
    Returns:
      list: Returned values
    """
    self._assert_not_closed()
    if values is not None and value is not None:
      raise ValueError('Both `value` and `values` are set. '
        'Only one of them can be specified.')
    ids = self._get_ids(id)
    if values is not None:
      assert len(values) == len(ids)
      return [self.workers[id].setattr(attrname, v)
        for id, v in zip(ids, values)]
    return [self.workers[id].setattr(attrname, value) for id in ids]

  def reset(self, id: Optional[List[int]]=None, **kwargs) -> Any:
    self._assert_not_closed()
    ids = self._get_ids(id)
    obs_list = [self.workers[id].reset(**kwargs) for id in ids]
    obs = self._flatten_obs(obs_list)
    return obs

  def step(
    self,
    acts: Union[np.ndarray, List, Tuple],
    id: Optional[List[int]] = None
  ) -> Tuple[Any]:
    self._assert_not_closed()
    ids = self._get_ids(id)
    assert len(acts) == len(ids)
    # step workers
    for act, id in zip(acts, ids):
      self.workers[id].step_async(act)
    results = [self.workers[id].step_wait() for id in ids]
    obs_list, rew_list, done_list, info_list = zip(*results)
    # stack results
    obs = self._flatten_obs(obs_list)
    rew = np.stack(rew_list)
    done = np.stack(done_list)
    return obs, rew, done, info_list

  def seed(
    self,
    seed: Optional[List[int]]=None,
    id: Optional[List[int]]=None
  ) -> List[Any]:
    self._assert_not_closed()
    ids = self._get_ids(id)
    if seed is None:
      seed_list = [None] * self.n_envs
    elif isinstance(seed, int):
      seed_list = [seed+id for id in ids]
    else:
      # list, tuple, np.ndarray
      assert len(seed) == len(ids)
      seed_list = seed
    return [self.workers[id].seed(s) for id, s in zip(ids, seed_list)]

  def render(self, **kwargs) -> List[Any]:
    self._assert_not_closed()
    return [w.render(**kwargs) for w in self.workers]

  def close(self):
    if self.closed:
      return
    for w in self.workers:
      w.close_async()
    for w in self.workers:
      w.close_wait()
    self._closed = True

  def _flatten_obs(self, obs_list: List[Any]) -> Any:
    obs = rl_utils.map_nested_tuple(
      tuple(obs_list), lambda obs: np.stack(obs))
    return obs

  def _get_ids(self, id: Optional[List[int]]=None) -> List[int]:
    if id is None:
      return list(range(self.n_envs))
    return [id] if np.isscalar(id) else id

  def _assert_not_closed(self):
    assert not self.closed, "This env is already closed"


class CloudpickleWrapper():
  def __init__(self, **kwargs):
    self.kwargs = kwargs

  def __getattr__(self, key: Any) -> Any:
    return self.kwargs.get(key)

  def __getstate__(self) -> Any:
    return cloudpickle.dumps(self.kwargs)

  def __setstate__(self, kwargs: Dict):
    self.kwargs = cloudpickle.loads(kwargs)

# Commands
class CMD(enum.Enum):
  getattr = 1
  setattr = 2
  reset   = 3
  step    = 4
  seed    = 5
  render  = 6
  close   = 7

def _subproc_worker(_p, p, param_wrapper):
  _p.close()
  import setproctitle
  import multiprocessing
  setproctitle.setproctitle(
    multiprocessing.current_process().name
  )
  env = param_wrapper.fn()
  try:
    while True:
      try:
        cmd, data = p.recv()
      except EOFError: # the pipe has been closed
        p.close()
        break
      if cmd == CMD.getattr:
        p.send(getattr(env, data[0], None))
      elif cmd == CMD.setattr:
        p.send(setattr(env, data[0], data[1]))
      elif cmd == CMD.reset:
        p.send(env.reset(**data[0]))
      elif cmd == CMD.step:
        obs, rew, done, info = env.step(data[0])
        p.send((obs, rew, done, info))
      elif cmd == CMD.seed:
        p.send(env.seed(data[0]))
      elif cmd == CMD.render:
        p.send(env.render(**data[0]))
      elif cmd == CMD.close:
        p.send(env.close())
        p.close()
        break
      else:
        p.close()
        raise NotImplementedError
  except KeyboardInterrupt:
    p.close()

class SubprocEnvWorker(BaseEnvWorker):
  def __init__(
    self,
    env_fn: Callable,
    worker_id: int
  ):
    methods = multiprocessing.get_all_start_methods()
    start_method = 'spawn'
    if 'forkserver' in methods:
      start_method = 'forkserver'
    ctx = multiprocessing.get_context(start_method)
    self.p, _p = ctx.Pipe()
    args = (
      self.p, _p, CloudpickleWrapper(fn=env_fn)
    )
    self.worker_id = worker_id
    self.process = ctx.Process(
      target = _subproc_worker,
      args = args,
      name = f'SubprocEnvWorker-{worker_id}',
      daemon = True
    )
    self.process.start()
    self._waiting_cmd = None
    _p.close()
    super().__init__(env_fn)
    
  def getattr(self, attrname: str) -> Any:
    return self._exec(CMD.getattr, attrname)

  def setattr(self, attrname: str, value: Any) -> Any:
    return self._exec(CMD.setattr, attrname, value)

  def reset(self, **kwargs) -> Any:
    return self._exec(CMD.reset, kwargs)

  def step_async(self, act: Any):
    self._exec(CMD.step, act, block=False)

  def step_wait(self, timeout: Optional[int]=None) -> Any:
    return self._wait(CMD.step, timeout=timeout)

  def seed(self, seed: int) -> Any:
    super().seed(seed)
    return self._exec(CMD.seed, seed)

  def render(self, **kwargs) -> Any:
    return self._exec(CMD.render, kwargs)

  def close_async(self):
    self._exec(CMD.close, block=False)

  def close_wait(self) -> Any:
    return self._wait(CMD.close, timeout=1)

  def _exec(
    self,
    cmd: CMD,
    *args,
    block: bool = True,
    timeout: Optional[int] = None
  ):
    #TODO: find a more reliable way
    if self._waiting_cmd and cmd != CMD.close:
      raise RuntimeError(f"Another command {cmd} was sent when "
        f"waiting for the reply {self._waiting_cmd}.")
    self.p.send([cmd, args])
    self._waiting_cmd = cmd
    if block:
      return self._wait(cmd, timeout=timeout)

  def _wait(self, cmd: CMD, timeout: Optional[int] = None):
    if self._waiting_cmd != cmd:
      raise RuntimeError(f"Waiting for command {cmd} but another command "
        f"{self._waiting_cmd} is executing.")
    res = None
    if self.p.poll(timeout):
      res = self.p.recv()
    self._waiting_cmd = None #unmarked
    return res


class SubprocVecEnv(BaseVecEnv):
  def __init__(
    self,
    env_fns: List[Callable],
    **kwargs
  ):
    kwargs.pop('worker_class', None)
    super().__init__(env_fns, SubprocEnvWorker, **kwargs)