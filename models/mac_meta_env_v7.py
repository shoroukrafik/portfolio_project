"""
MAC v7 — ABLATION: drops the HUNTER (aggressive) specialist.

Companion to v5 (drop balanced) and v6 (drop defender). Completes the
hold-out-one-specialist series.

Difference vs v3:
  - 2 specialists: defensive + balanced (NO hunter)
  - action space: 2 logits → softmax → (w_def, w_bal)
  - on bull days, routing forced between defender (70% bear floor + CVaR-aware
    risk-averse) and steady (light risk, balanced) — neither captures hunter's
    bull-aggressive concentration

Predicted result (from personality grid):
  bull period: catastrophic loss vs v3 (no hunter → steady's +46.30/+17.54
                or defender's +14.25/+8.98; weighted = far below v3's
                +38.61% / +17.33%)
  bear period: ~similar to v3 (hunter contributed 2-9% on bear, removing it
                slightly improves bear)
  2023 period: ~similar or slightly worse (steady ran 2023 alone)
  net: dominantly worse due to bull catastrophe

This is an ablation, not a production candidate.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

from portfolio_env import PortfolioEnv
from mars_pareto_controller import _apply_training_floor


class MACMetaEnvV7(gym.Env):
    """MAC v7 meta-env — 2-way (def/bal) router, no hunter specialist."""

    metadata = {'render_modes': []}

    def __init__(self, df, specialists_paths, env_kwargs, training_floors,
                 device='cpu'):
        super().__init__()
        self.master_env = PortfolioEnv(df=df, **env_kwargs)

        self.defensive = PPO.load(specialists_paths['defensive'], device=device)
        self.balanced  = PPO.load(specialists_paths['balanced'],  device=device)
        self.training_floors = training_floors

        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(6,), dtype=np.float32
        )
        # 2 routing logits → (w_def, w_bal)
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
        w_def, w_bal = float(w[0]), float(w[1])

        step = self.master_env.current_step
        regime_id = int(self.master_env.regime_id_array[step])

        action_def, _ = self.defensive.predict(self._master_obs, deterministic=True)
        action_bal, _ = self.balanced.predict(self._master_obs, deterministic=True)

        action_def = _apply_training_floor(action_def, regime_id,
                                           self.training_floors['defensive'])
        action_bal = _apply_training_floor(action_bal, regime_id,
                                           self.training_floors['balanced'])

        blended = w_def * action_def + w_bal * action_bal

        new_obs, reward, term, trunc, info = self.master_env.step(blended)
        self._master_obs = new_obs

        info = dict(info)
        info['routing'] = {'w_def': w_def, 'w_bal': w_bal, 'w_agg': 0.0}
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


SPECIALISTS_DJI_V7_ABL = {
    'defensive': 'models/saved/ppo_defender_dji_v3.zip',
    'balanced':  'models/saved/ppo_steady_dji_v3.zip',
}

SPECIALISTS_SP100_V7_ABL = {
    'defensive': 'models/saved/ppo_defender_sp100_v3.zip',
    'balanced':  'models/saved/ppo_steady_sp100_v3.zip',
}

V7_ABL_TRAINING_FLOORS = {
    'defensive': {0: 0.0, 1: 0.7, 2: 0.2},
    'balanced':  {0: 0.0, 1: 0.2, 2: 0.0},
}
