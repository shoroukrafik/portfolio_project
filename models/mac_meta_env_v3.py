"""
MAC Meta-Environment for MARS v3 — final production architecture.

Structure inherits from v2 MAC's design (Paper 5 MARS-inspired meta-controller
over heterogeneous specialists), but SIMPLIFIED because all 3 v3 specialists
share the same observation shape (20 features + regime + probs + vol).

Key differences from v2 (v1c+ws lineup):
  v2 (v1c+ws): used 3 different obs shapes → needed MACMetaEnvV1C with
               per-role envs running in lockstep
  v3 (current): all specialists share 20-feature obs → ONE master env

What stays the same as v2:
  - MAC architecture: 6 → 32 → 16 → 3 MLP
  - MAC obs: regime probs (3) + broad-market vol features (3) = 6 dims
  - MAC action: 3 routing logits → softmax → (w_def, w_bal, w_agg)
  - Supervised warm-start on HMM regime probs
  - PPO fine-tune for 250k steps with 25k checkpointing
  - Per-specialist cash-floor preconditioning (defender trained with bear
    floor 0.7 + sideways floor 0.2; steady with bear floor 0.2; hunter no
    floor) — preserved by applying the floor to each specialist's action
    BEFORE blending so their behaviour matches training-time enforcement.

At each timestep:
  1. Master env produces one obs (shared by all specialists)
  2. Each specialist predicts action on that obs
  3. Each action is floor-preconditioned by the specialist's TRAINING floor
  4. MAC routes via regime probs + vol features
  5. Blended action steps the master env
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

from portfolio_env import PortfolioEnv
from mars_pareto_controller import _apply_training_floor


class MACMetaEnvV3(gym.Env):
    """MAC v3 meta-env — single master env + per-specialist floor preconditioning."""

    metadata = {'render_modes': []}

    def __init__(self, df, specialists_paths, env_kwargs, training_floors,
                 device='cpu'):
        super().__init__()
        # ONE master env — all specialists share its obs (same 20-feat + probs + vol).
        self.master_env = PortfolioEnv(df=df, **env_kwargs)

        self.defensive  = PPO.load(specialists_paths['defensive'],  device=device)
        self.balanced   = PPO.load(specialists_paths['balanced'],   device=device)
        self.aggressive = PPO.load(specialists_paths['aggressive'], device=device)
        # Each specialist's TRAINING cash-floor. When the master env runs with
        # zero floors, we precondition specialist actions with the floor they
        # were trained with so their behaviour matches training-time enforcement.
        self.training_floors = training_floors

        # MAC obs: regime probs (3) + broad-market vol features (3) = 6 dims.
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(6,), dtype=np.float32
        )
        # MAC action: 3 routing logits.
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
        # 1. MAC routing weights (softmax over 3 logits)
        a = np.array(mac_action, dtype=np.float32)
        a = np.nan_to_num(a, nan=0.0, posinf=5.0, neginf=-5.0)
        a = a - a.max()
        w = np.exp(a)
        ws = w.sum()
        if ws < 1e-8:
            w = np.array([1.0, 1.0, 1.0], dtype=np.float32) / 3.0
        else:
            w = w / ws
        w_def, w_bal, w_agg = float(w[0]), float(w[1]), float(w[2])

        step = self.master_env.current_step
        regime_id = int(self.master_env.regime_id_array[step])

        # 2. Each specialist predicts on the SAME master env obs
        action_def, _ = self.defensive.predict(self._master_obs, deterministic=True)
        action_bal, _ = self.balanced.predict(self._master_obs, deterministic=True)
        action_agg, _ = self.aggressive.predict(self._master_obs, deterministic=True)

        # 3. Pre-condition each action with its TRAINING cash floor
        # (defender bear=0.7 sideways=0.2; steady bear=0.2; hunter zero)
        action_def = _apply_training_floor(action_def, regime_id,
                                           self.training_floors['defensive'])
        action_bal = _apply_training_floor(action_bal, regime_id,
                                           self.training_floors['balanced'])
        action_agg = _apply_training_floor(action_agg, regime_id,
                                           self.training_floors['aggressive'])

        # 4. Blend
        blended = (w_def * action_def
                   + w_bal * action_bal
                   + w_agg * action_agg)

        # 5. Step master env
        new_obs, reward, term, trunc, info = self.master_env.step(blended)
        self._master_obs = new_obs

        info = dict(info)
        info['routing'] = {'w_def': w_def, 'w_bal': w_bal, 'w_agg': w_agg}
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


# Universal v3 specialist lineup (locked by methodology):
SPECIALISTS_DJI_V3 = {
    'defensive':  'models/saved/ppo_defender_dji_v3.zip',   # step 25k, bear -2.63%
    'balanced':   'models/saved/ppo_steady_dji_v3.zip',     # step 850k, 2023 +36.82%
    'aggressive': 'models/saved/ppo_hunter_dji_v3.zip',     # step 950k, bull +43.43%
}

SPECIALISTS_SP100_V3 = {
    'defensive':  'models/saved/ppo_defender_sp100_v3.zip', # step 50k, bear -3.91%
    'balanced':   'models/saved/ppo_steady_sp100_v3.zip',   # step 1M,  2023 +16.04%
    'aggressive': 'models/saved/ppo_hunter_sp100_v3.zip',   # step 1M,  bull +22.44%
}

# Per-specialist TRAINING cash floors (universal across both universes —
# each role used same recipe on both DJI and SP100).
V3_TRAINING_FLOORS = {
    'defensive':  {0: 0.0, 1: 0.7, 2: 0.2},  # defender recipe
    'balanced':   {0: 0.0, 1: 0.2, 2: 0.0},  # steady recipe
    'aggressive': {0: 0.0, 1: 0.0, 2: 0.0},  # hunter recipe (no floor)
}
