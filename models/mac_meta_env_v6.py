"""
MAC v6 — ABLATION: drops the DEFENDER specialist.

Companion to v5 (drop balanced) and v7 (drop hunter). Together v5/v6/v7 form
the complete hold-out-one-specialist ablation series, confirming each
specialist's load-bearing role.

This is an ablation, not a production candidate. Naming '_v6' for filesystem
isolation; production lineup stays v3.

Same plumbing as v3 / v5:
  - single master env (all specialists share obs shape)
  - per-specialist floor preconditioning preserved
  - MAC obs unchanged: 6 dims (regime probs + vol features)

Difference vs v3:
  - 2 specialists: balanced + aggressive (NO defender)
  - action space: 2 logits → softmax → (w_bal, w_agg)
  - on bear days, routing forced between steady (light bear floor 0.2) and
    hunter (no floor) — neither has CVaR-aware bear discipline

Predicted result (from personality grid):
  bear period: catastrophic (no defender → either steady -13.27% / -9.98%
                 or hunter -17.73% / -12.62%; weighted = far worse than v3's
                 -7.95% / -7.19%)
  bull period: ~similar to v3 (defender contributed negligibly on bull)
  2023 period: ~similar to v3 (defender contributed negligibly on 2023)
  net: dominantly worse due to bear catastrophe
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

from portfolio_env import PortfolioEnv
from mars_pareto_controller import _apply_training_floor


class MACMetaEnvV6(gym.Env):
    """MAC v6 meta-env — 2-way (bal/agg) router, no defender specialist."""

    metadata = {'render_modes': []}

    def __init__(self, df, specialists_paths, env_kwargs, training_floors,
                 device='cpu'):
        super().__init__()
        self.master_env = PortfolioEnv(df=df, **env_kwargs)

        self.balanced   = PPO.load(specialists_paths['balanced'],   device=device)
        self.aggressive = PPO.load(specialists_paths['aggressive'], device=device)
        self.training_floors = training_floors

        # MAC obs unchanged from v3: 6 dims
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(6,), dtype=np.float32
        )
        # 2 routing logits → (w_bal, w_agg)
        self.action_space = spaces.Box(
            low=-5.0, high=5.0, shape=(2,), dtype=np.float32
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
        ws = w.sum()
        if ws < 1e-8:
            w = np.array([0.5, 0.5], dtype=np.float32)
        else:
            w = w / ws
        w_bal, w_agg = float(w[0]), float(w[1])

        step = self.master_env.current_step
        regime_id = int(self.master_env.regime_id_array[step])

        action_bal, _ = self.balanced.predict(self._master_obs, deterministic=True)
        action_agg, _ = self.aggressive.predict(self._master_obs, deterministic=True)

        action_bal = _apply_training_floor(action_bal, regime_id,
                                           self.training_floors['balanced'])
        action_agg = _apply_training_floor(action_agg, regime_id,
                                           self.training_floors['aggressive'])

        blended = w_bal * action_bal + w_agg * action_agg

        new_obs, reward, term, trunc, info = self.master_env.step(blended)
        self._master_obs = new_obs

        info = dict(info)
        info['routing'] = {'w_def': 0.0, 'w_bal': w_bal, 'w_agg': w_agg}
        info['regime_id'] = regime_id

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


SPECIALISTS_DJI_V6_ABL = {
    'balanced':   'models/saved/ppo_steady_dji_v3.zip',
    'aggressive': 'models/saved/ppo_hunter_dji_v3.zip',
}

SPECIALISTS_SP100_V6_ABL = {
    'balanced':   'models/saved/ppo_steady_sp100_v3.zip',
    'aggressive': 'models/saved/ppo_hunter_sp100_v3.zip',
}

V6_ABL_TRAINING_FLOORS = {
    'balanced':   {0: 0.0, 1: 0.2, 2: 0.0},
    'aggressive': {0: 0.0, 1: 0.0, 2: 0.0},
}
