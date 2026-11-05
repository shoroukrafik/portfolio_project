"""
MAC Meta-Environment with v1c-style cross-variant specialists.

Wraps 3 frozen specialists with DIFFERENT training recipes:
  defensive  = v1 FULL  (19 feat, cash-floor env defaults)
  balanced   = v2       (20 feat + probs + vol, bear-floor 0.5)
  aggressive = v6       (20 feat + probs + vol, no floor)

Because each specialist has a different obs shape, we run one PortfolioEnv
PER role and step them in lockstep with the same blended action. Each
specialist's action is pre-conditioned with its TRAINING cash-floor before
blending (mirrors what the env would have enforced during training).

Routing is the MAC's responsibility — it sees a small obs (regime probs +
broad-market vol features) and outputs 3 logits → softmax → (w_def, w_bal,
w_agg).
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

from portfolio_env import PortfolioEnv
from mars_pareto_controller import _apply_training_floor


class MACMetaEnvV1C(gym.Env):
    """Per-role-env meta-env. Specialists have different obs shapes."""

    metadata = {'render_modes': []}

    def __init__(self, df, specialists_paths, env_kwargs_base,
                 role_env_configs, training_floors, device='cpu'):
        super().__init__()

        # Build one env per role with that specialist's obs config.
        self.envs = {}
        for role, cfg in role_env_configs.items():
            role_df = df.drop(columns=[c for c in cfg['drop']
                                       if c in df.columns])
            kwargs = dict(env_kwargs_base)
            kwargs.update(cfg['flags'])
            self.envs[role] = PortfolioEnv(df=role_df, **kwargs)
        self.master = self.envs['balanced']  # arbitrary — they're synced

        self.specialists = {
            role: PPO.load(specialists_paths[role], device=device)
            for role in ('defensive', 'balanced', 'aggressive')
        }
        self.training_floors = training_floors

        # MAC obs: regime probs (3) + broad-market vol features (3) = 6 dims.
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(6,), dtype=np.float32
        )
        # MAC action: 3 logits → softmax → routing weights.
        self.action_space = spaces.Box(
            low=-5.0, high=5.0, shape=(3,), dtype=np.float32
        )

        self._role_obs = {}

    def _extract_mac_obs(self):
        step = min(self.master.current_step, self.master.n_days - 1)
        regime_probs = self.master.regime_prob_array[step]
        vol_feat     = self.master.vol_features_array[step]
        obs = np.concatenate([regime_probs, vol_feat]).astype(np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    def reset(self, *, seed=None, options=None):
        info = {}
        for role, env in self.envs.items():
            o, info_role = env.reset(seed=seed, options=options)
            self._role_obs[role] = o
            info = info_role
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

        step = self.master.current_step
        regime_id = int(self.master.regime_id_array[step])

        action_def, _ = self.specialists['defensive'].predict(
            self._role_obs['defensive'], deterministic=True)
        action_bal, _ = self.specialists['balanced'].predict(
            self._role_obs['balanced'], deterministic=True)
        action_agg, _ = self.specialists['aggressive'].predict(
            self._role_obs['aggressive'], deterministic=True)

        # Pre-condition each action with the cash-floor it was TRAINED with.
        action_def = _apply_training_floor(action_def, regime_id,
                                           self.training_floors['defensive'])
        action_bal = _apply_training_floor(action_bal, regime_id,
                                           self.training_floors['balanced'])
        action_agg = _apply_training_floor(action_agg, regime_id,
                                           self.training_floors['aggressive'])

        blended = (w_def * action_def
                   + w_bal * action_bal
                   + w_agg * action_agg)

        # Step every env with the same blended action.
        master_reward, master_term, master_trunc, master_info = 0.0, False, False, {}
        for role, env in self.envs.items():
            new_obs, reward, term, trunc, info_role = env.step(blended)
            self._role_obs[role] = new_obs
            if role == 'balanced':
                master_reward = reward
                master_term, master_trunc = term, trunc
                master_info = dict(info_role)
        master_info['routing'] = {'w_def': w_def, 'w_bal': w_bal,
                                  'w_agg': w_agg}
        master_info['regime_id'] = regime_id

        return (self._extract_mac_obs(), master_reward, master_term,
                master_trunc, master_info)

    def render(self):
        pass

    @property
    def portfolio_value(self):
        return self.master.portfolio_value

    @property
    def current_step(self):
        return self.master.current_step

    @property
    def dates(self):
        return self.master.dates

    def get_portfolio_history(self):
        return self.master.get_portfolio_history()
