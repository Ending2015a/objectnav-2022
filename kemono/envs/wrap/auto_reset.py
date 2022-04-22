# --- built in ---
# --- 3rd party ---
import gym
# --- my module ---


class AutoResetWrapper(gym.Wrapper):
  def __init__(self, env):
    super().__init__(env)

  def step(self, action):
    obs, rew, done, info = self.env.step(action)
    if done:
      obs = self.env.reset()
    return obs, rew, done, info
