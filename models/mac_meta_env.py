"""
MAC Meta-Environment for MARS-Pareto v2.

Wraps a PortfolioEnv + 3 frozen v6 specialists. The "agent" is a small MAC
(Meta-Adaptive Controller) that outputs routing weights over the specialists.

At each step:
  1. MAC sees a small obs: regime probs (3) + broad-market vol features (3)
  2. MAC outputs 3 logits → softmax → (w_def, w_bal, w_agg)
  3. Each specialist runs on the FULL master-env obs and proposes an action
  4. Blended_action = w_def*action_def + w_bal*action_bal + w_agg*action_agg
  5. Master env steps with the blended action → reward goes to MAC
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

from portfolio_env import PortfolioEnv


class MACMetaEnv(gym.Env):
    """3-specialist meta-env. MAC's action is the routing distribution."""

    metadata = {'render_modes': []}

    def __init__(self, df, specialists_paths, env_kwargs, device='cpu'):
        super().__init__()
        self.master_env = PortfolioEnv(df=df, **env_kwargs)
        self.defensive  = PPO.load(specialists_paths['defensive'],  device=device)
        self.balanced   = PPO.load(specialists_paths['balanced'],   device=device)
        self.aggressive = PPO.load(specialists_paths['aggressive'], device=device)

        # MAC obs: regime probs (3) + broad-market vol features (3) = 6 dims.
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(6,), dtype=np.float32
        )
        # MAC action: 3 logits (env softmaxes → routing weights).
        self.action_space = spaces.Box(
            low=-5.0, high=5.0, shape=(3,), dtype=np.float32
        )

        self._master_obs = None

    def _extract_mac_obs(self):
        step = min(self.master_env.current_step, self.master_env.n_days - 1)
        regime_probs = self.master_env.regime_prob_array[step]
        vol_feat     = self.master_env.vol_features_array[step]
        obs = np.concatenate([regime_probs, vol_feat]).astype(np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    def reset(self, *, seed=None, options=None):
        master_obs, info = self.master_env.reset(seed=seed, options=options)
        self._master_obs = master_obs
        return self._extract_mac_obs(), info

    def step(self, mac_action):
        a = np.array(mac_action, dtype=np.float32)
        a = np.nan_to_num(a, nan=0.0, posinf=5.0, neginf=-5.0)
        a = a - a.max()
        w = np.exp(a)
        w_sum = w.sum()
        if w_sum < 1e-8:
            w = np.array([1.0, 1.0, 1.0], dtype=np.float32) / 3.0
        else:
            w = w / w_sum
        w_def, w_bal, w_agg = float(w[0]), float(w[1]), float(w[2])

        action_def, _ = self.defensive.predict(self._master_obs, deterministic=True)
        action_bal, _ = self.balanced.predict(self._master_obs, deterministic=True)
        action_agg, _ = self.aggressive.predict(self._master_obs, deterministic=True)

        blended = (w_def * action_def
                   + w_bal * action_bal
                   + w_agg * action_agg)

        new_obs, reward, term, trunc, info = self.master_env.step(blended)
        self._master_obs = new_obs

        info = dict(info)
        info['routing'] = {'w_def': w_def, 'w_bal': w_bal, 'w_agg': w_agg}

        return self._extract_mac_obs(), reward, term, trunc, info

    def render(self):
        pass

    @property
    def portfolio_value(self):
        return self.master_env.portfolio_value

    @property
    def current_step(self):
        return self.master_env.current_step

    @property
    def dates(self):
        return self.master_env.dates

    def get_portfolio_history(self):
        return self.master_env.get_portfolio_history()
