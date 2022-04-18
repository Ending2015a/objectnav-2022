# --- built in ---
from typing import Any, Optional, Dict, Deque
import collections
# --- 3rd party ---
import cv2
import numpy as np
import gym
import pytorch_lightning as pl
# --- my module ---

DEBUG = True

class LastNStatistics:
  def __init__(self, n_episodes: int=10):
    self.n_episodes = n_episodes
    self.lengths = collections.deque(maxlen=self.n_episodes)
    self.rewards = collections.deque(maxlen=self.n_episodes)
    self.spl = collections.deque(maxlen=self.n_episodes)
    self.softspl = collections.deque(maxlen=self.n_episodes)
    self.success = collections.deque(maxlen=self.n_episodes)
    self.dist2goal = collections.deque(maxlen=self.n_episodes)

  def append_metrics(self, metrics: Dict[str, float]):
    self.spl.append(metrics['spl'])
    self.softspl.append(metrics['softspl'])
    self.success.append(metrics['success'])
    self.dist2goal.append(metrics['dist2goal'])
  
  def append_length(self, length: int):
    self.lengths.append(length)
  
  def append_reward(self, reward: int):
    self.rewards.append(reward)

  def get_default_average(self, deque: Deque, default=0.0):
    if len(deque) == 0:
      return default
    else:
      return np.nanmean(deque)

  def get_average_dict(self) -> Dict[str, float]:
    return dict(
      avg_length = self.get_default_average(self.lengths),
      avg_reward = self.get_default_average(self.rewards),
      avg_spl = self.get_default_average(self.spl),
      avg_softspl = self.get_default_average(self.softspl),
      avg_success = self.get_default_average(self.success),
      avg_dist2goal = self.get_default_average(self.dist2goal)
    )


class Runner():
  def __init__(
    self,
    env: gym.Env,
    agent,
    goal_mapping: Dict[int, str],
    report_n_episodes: int = 10,
  ):
    assert isinstance(env, gym.Env)
    self.env = env
    self.agent = agent
    self.goal_mapping = goal_mapping
    self.report_n_episodes = report_n_episodes

    self._cached_obs = None
    self._cached_states = None
    self._cached_reset = None
    self._not_reset_yet = True

    self.episode_lengths = None
    self.episode_rewards = None
    self.completed_episodes = 0
    self.total_timesteps = 0
    self.last_n_statistics = LastNStatistics(self.report_n_episodes)
    # per goal statistics
    self.goal_last_n_statistics = [
      LastNStatistics(self.report_n_episodes)
      for n in range(len(goal_mapping.keys()))
    ]

  def reset_statistics(self):
    self.episode_lengths = 0
    self.episode_rewards = 0

  def reset(self):
    self._cached_obs = self.env.reset()
    self._cached_states = self.agent.get_states(batch_size=1)
    self._cached_makes = True
    self.reset_statistics()
    self._not_reset_yet = False

  def _sample_action(self, random: bool=False):
    # random sample actions
    act, next_states = self.agent.predict(
      self._cached_obs,
      states = self._cached_states,
      reset = self._cached_reset,
      det = False
    )
    if random:
      act = np.asarray(self.env.action_spaces.sample())
    return act, next_states

  def _collect_step(self, random: bool = False):
    if self._not_reset_yet:
      self.reset()
    act, next_states = self._sample_action(random=random)
    # step environment
    next_obs, rew, done, info = self.env.step(act)
    # cache rollouts
    self._cached_obs = next_obs
    self._cached_states = next_states
    self._cached_reset = done
    self._not_reset_yet = done
    return next_obs, rew, done, info

  def step(self, random: bool=False):
    next_obs, rew, done, info = self._collect_step(random=random)
    if DEBUG:
      self.env.render('human')
      cv2.waitKey(1)
    # make statistics
    metrics = info['metrics']
    self.episode_lengths += 1
    self.episode_rewards += rew
    self.total_timesteps += 1
    if done:
      objectgoal = np.asarray(next_obs['objectgoal']).item()
      self._update_last_n_statistics(
        self.last_n_statistics, metrics
      )
      # update per goal statistics
      self._update_last_n_statistics(
        self.goal_last_n_statistics[objectgoal], metrics
      )
      self.completed_episodes += 1
      # the statistics are reset in the `self.reset()`
    return next_obs, rew, done, info

  def _update_last_n_statistics(
    self,
    last_n: LastNStatistics,
    metrics: Dict[str, float]
  ):
    last_n.append_length(self.episode_lengths)
    last_n.append_reward(self.episode_rewards)
    last_n.append_metrics(metrics)

  def collect(
    self,
    n_steps: int = 64,
    random: bool = False
  ):
    """Collect samples
    The samples are stored to TrajectoryRecorders
    """
    for n in range(n_steps):
      self.step(random=random)

  def log_dict(self, agent=None, scope=''):
    if agent is None:
      agent = self.agent
    assert isinstance(agent, pl.LightningModule)
    scope_op = lambda *x: '/'.join(filter(None, [scope]+list(x)))
    # create log_dict
    log_dict = {}
    _log_dict = self.last_n_statistics.get_average_dict()
    for key, value in _log_dict.items():
      log_dict[scope_op(key)] = value
    for goal_id, goal_name in self.goal_mapping.items():
      _log_dict = self.goal_last_n_statistics[goal_id].get_average_dict()
      for key, value in _log_dict.items():
        log_dict[scope_op(goal_name, key)] = value
    # log to tensorboard through agent
    agent.log_dict(
      log_dict,
      reduce_fx = 'mean',
      sync_dist = True
    )
    agent.log_dict({
        scope_op('completed_episodes'): self.completed_episodes,
        scope_op('total_timesteps'): self.total_timesteps
      },
      reduce_fx = 'sum',
      sync_dist = True
    )
