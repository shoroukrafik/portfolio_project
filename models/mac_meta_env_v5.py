"""
MAC Meta-Environment v5 — ABLATION (drops the balanced specialist).

Path B from our v3-vs-v4 analysis: tests whether removing the steady specialist
sharpens MAC behavior on pure bull/bear periods (the user's intuition was that
specialists are strong individually and MAC's 3-way blending dilutes them).

Same plumbing as v3:
  - single master env (all specialists share obs shape)
  - per-specialist floor preconditioning preserved (defender bear=0.7,
    sideways=0.2; hunter no floor)
  - MAC obs unchanged: 6 dims (regime probs + vol features)

Difference vs v3:
  - 2 specialists instead of 3 (no balanced/steady)
  - action space: 2 logits → softmax → (w_def, w_agg)
  - on sideways days, routing falls to whichever of bear/bull has higher residual

Predicted result (from earlier math):
  bull period: routes ~92% to hunter → gains ~3-4pp toward hunter ceiling
  bear period: routes ~98% to defender → gains ~3pp toward defender ceiling
  2023 period: forced into hunter/defender mix → loses ~8-12pp vs steady's role-fit
  net: ~ -3pp on total (sideways loss > pure-regime gains)

This is an ablation, not a production candidate. Naming '_v5' for filesystem
isolation; production lineup stays v3.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

from portfolio_env import PortfolioEnv
from mars_pareto_controller import _apply_training_floor


class MACMetaEnvV5(gym.Env):
    """MAC v5 meta-env — 2-way (def/agg) router, no balanced specialist."""

    metadata = {'render_modes': []}

    def __init__(self, df, specialists_paths, env_kwargs, training_floors,
                 device='cpu'):
        super().__init__()
        self.master_env = PortfolioEnv(df=df, **env_kwargs)

        self.defensive  = PPO.load(specialists_paths['defensive'],  device=device)
        self.aggressive = PPO.load(specialists_paths['aggressive'], device=device)
        self.training_floors = training_floors

        # MAC obs unchanged from v3: 6 dims
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(6,), dtype=np.float32
        )
        # ← v5 difference: 2 routing logits instead of 3
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
        # 2-way softmax
        a = np.array(mac_action, dtype=np.float32)
        a = np.nan_to_num(a, nan=0.0, posinf=5.0, neginf=-5.0)
        a = a - a.max()
        w = np.exp(a)
        ws = w.sum()
        if ws < 1e-8:
            w = np.array([0.5, 0.5], dtype=np.float32)
        else:
            w = w / ws
        w_def, w_agg = float(w[0]), float(w[1])

        step = self.master_env.current_step
        regime_id = int(self.master_env.regime_id_array[step])

        action_def, _ = self.defensive.predict(self._master_obs, deterministic=True)
        action_agg, _ = self.aggressive.predict(self._master_obs, deterministic=True)

        # Pre-condition each action with its TRAINING cash floor
        action_def = _apply_training_floor(action_def, regime_id,
                                           self.training_floors['defensive'])
        action_agg = _apply_training_floor(action_agg, regime_id,
                                           self.training_floors['aggressive'])

        blended = w_def * action_def + w_agg * action_agg

        new_obs, reward, term, trunc, info = self.master_env.step(blended)
        self._master_obs = new_obs

        info = dict(info)
        info['routing'] = {'w_def': w_def, 'w_bal': 0.0, 'w_agg': w_agg}
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


# v5 ablation lineup — reuses the production v3 hunter + defender checkpoints.
SPECIALISTS_DJI_V5 = {
    'defensive':  'models/saved/ppo_defender_dji_v3.zip',
    'aggressive': 'models/saved/ppo_hunter_dji_v3.zip',
}

SPECIALISTS_SP100_V5 = {
    'defensive':  'models/saved/ppo_defender_sp100_v3.zip',
    'aggressive': 'models/saved/ppo_hunter_sp100_v3.zip',
}

V5_TRAINING_FLOORS = {
    'defensive':  {0: 0.0, 1: 0.7, 2: 0.2},
    'aggressive': {0: 0.0, 1: 0.0, 2: 0.0},
}
