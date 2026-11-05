"""
MAC observation wrappers for ablation experiments.

Both wrappers only modify what the MAC CONTROLLER sees — they do NOT touch
the master env, the specialists, or the env-side reward shaping. This isolates
the question "does MAC need the HMM signal / vol features to perform?"

  ShuffleRegimeMACObs  — replaces obs[:3] (HMM regime probs) with a random
                          Dirichlet sample each step. Tests whether the
                          learned controller relies on the actual HMM signal.

  NoVolMACObs          — strips obs[3:6] (vol features). MAC sees only 3D
                          regime probs. Tests whether vol features add value.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class ShuffleRegimeMACObs(gym.Wrapper):
    """Replace MAC's regime probs (first 3 obs dims) with random Dirichlet draws.

    Vol features (last 3 obs dims) pass through unchanged.
    Underlying env, specialists, and reward shaping are untouched — only the
    controller's OBSERVATION of regime is scrambled.
    """

    def __init__(self, env, seed=42):
        super().__init__(env)
        self.rng = np.random.default_rng(seed)

    def _scramble(self, obs):
        random_probs = self.rng.dirichlet([1.0, 1.0, 1.0]).astype(np.float32)
        obs = np.array(obs, copy=True)
        obs[:3] = random_probs
        return obs

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._scramble(obs), info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        return self._scramble(obs), r, term, trunc, info

    # Pass-through accessors used by training callbacks (val_env.get_portfolio_history,
    # val_env.portfolio_value etc. live on the underlying MAC env).
    def get_portfolio_history(self):
        return self.env.get_portfolio_history()

    @property
    def portfolio_value(self):
        return self.env.portfolio_value

    @property
    def current_step(self):
        return self.env.current_step

    @property
    def dates(self):
        return self.env.dates


class NoVolMACObs(gym.Wrapper):
    """Strip vol features from MAC's obs — controller sees only 3D regime probs.

    Underlying env and specialists are untouched.
    """

    def __init__(self, env):
        super().__init__(env)
        old = env.observation_space
        self.observation_space = spaces.Box(
            low=old.low[:3], high=old.high[:3],
            shape=(3,), dtype=old.dtype,
        )

    def _strip(self, obs):
        return np.array(obs[:3], copy=True)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._strip(obs), info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        return self._strip(obs), r, term, trunc, info

    # Pass-through accessors needed by training callbacks.
    def get_portfolio_history(self):
        return self.env.get_portfolio_history()

    @property
    def portfolio_value(self):
        return self.env.portfolio_value

    @property
    def current_step(self):
        return self.env.current_step

    @property
    def dates(self):
        return self.env.dates
