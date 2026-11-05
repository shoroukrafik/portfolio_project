"""
MARS-Pareto Controller — soft-routed multi-agent ensemble.

Architecture (Paper 5 MARS-inspired):
  3 frozen specialists, each at a distinct point on the v6 Pareto frontier:
    Defensive   — early-training checkpoint (best bear MDD)
    Balanced    — mid-training checkpoint (best benchmark wins)
    Aggressive  — late-training checkpoint (best bull capture)

  At each timestep, the controller blends the three specialists' actions
  weighted by the HMM's regime probabilities for that day:

      action_t = bull_prob_t      × action_aggressive(obs_t)
               + bear_prob_t      × action_defensive(obs_t)
               + sideways_prob_t  × action_balanced(obs_t)

  Since bull_prob + bear_prob + sideways_prob = 1 by construction,
  the blended action is a convex combination of the specialists'
  pre-softmax logits. The env's softmax then converts to weights.

Novelty vs Paper 5:
  Their specialists were trained from scratch with different (θ, λ)
  risk parameters. Ours are EXTRACTED from a single training's
  checkpoint sweep — risk profiles emerge naturally as PPO migrates
  along the Pareto frontier during training.
"""

import os
import numpy as np
from stable_baselines3 import PPO


class MarsParetoController:
    """Three-specialist soft-routed ensemble. v1: rule-based MAC."""

    def __init__(self, defensive_path, balanced_path, aggressive_path,
                 max_bull_path=None,
                 device='cpu', temperature=1.0, blend_mode='logits'):
        self.defensive  = PPO.load(defensive_path,  device=device)
        self.balanced   = PPO.load(balanced_path,   device=device)
        self.aggressive = PPO.load(aggressive_path, device=device)
        # Optional 4th specialist: max-bull, engaged only when bull confidence
        # exceeds 0.5. Routing splits the bull-prob weight between aggressive
        # and max-bull based on bull confidence (linear ramp above 0.5).
        self.max_bull = (PPO.load(max_bull_path, device=device)
                         if max_bull_path is not None else None)

        self.specialist_paths = {
            'defensive':  defensive_path,
            'balanced':   balanced_path,
            'aggressive': aggressive_path,
            'max_bull':   max_bull_path,
        }
        # Routing temperature for softmax(regime_probs / T):
        #   T = 1.0  → use HMM probs directly (v1 default)
        #   T < 1.0  → sharpen routing (v1a: T=0.5)
        #   T > 1.0  → flatten (more mixed routing)
        self.temperature = float(temperature)
        # Blend mode:
        #   'logits'  → average the specialists' raw action logits, env softmaxes (v1)
        #   'weights' → softmax each specialist FIRST → average weights → output
        #               log(weights) so env's downstream softmax recovers them.
        #               TRUE convex combination of portfolios (v1d).
        if blend_mode not in ('logits', 'weights'):
            raise ValueError(f"blend_mode must be 'logits' or 'weights', got {blend_mode}")
        self.blend_mode = blend_mode
        # Track strategy usage for analysis
        self.usage_log = []

    def predict(self, obs, regime_probs, deterministic=True):
        """
        regime_probs: np.array([bull_prob, bear_prob, sideways_prob])
                      typically from env.regime_prob_array[step]
        Returns: (blended_action, None) — matches SB3 predict() signature.
        """
        bull_p, bear_p, sideways_p = regime_probs

        action_def, _ = self.defensive.predict(obs, deterministic=deterministic)
        action_bal, _ = self.balanced.predict(obs, deterministic=deterministic)
        action_agg, _ = self.aggressive.predict(obs, deterministic=deterministic)
        action_max = None
        if self.max_bull is not None:
            action_max, _ = self.max_bull.predict(obs, deterministic=deterministic)

        # Apply temperature to sharpen / flatten routing.
        # Mapping: bull → aggressive, bear → defensive, sideways → balanced.
        # Order in the softmax: [aggressive, defensive, balanced]
        raw = np.array([bull_p, bear_p, sideways_p], dtype=np.float32)
        if self.temperature != 1.0 and self.temperature > 0:
            # Add tiny epsilon for numerical stability when probs are near 0
            log_p = np.log(np.maximum(raw, 1e-8))
            scaled = log_p / self.temperature
            scaled -= scaled.max()
            sharp = np.exp(scaled)
            sharp /= sharp.sum()
            w_agg, w_def, w_bal = float(sharp[0]), float(sharp[1]), float(sharp[2])
        else:
            w_agg = float(bull_p)
            w_def = float(bear_p)
            w_bal = float(sideways_p)

        # Optional max-bull split: when bull confidence > 0.5, ramp the bull
        # weight from aggressive → max-bull linearly. At bull_prob=0.5 the
        # max-bull gets 0; at bull_prob=1.0 it gets the full bull weight.
        # Total weights still sum to 1.
        w_max = 0.0
        if self.max_bull is not None and w_agg > 0.0:
            confidence_above_half = max(0.0, float(bull_p) - 0.5) * 2.0
            confidence_above_half = min(confidence_above_half, 1.0)
            w_max = w_agg * confidence_above_half
            w_agg = w_agg - w_max

        if self.blend_mode == 'logits':
            # v1 baseline: weighted average of raw logits.
            blended = (w_def * action_def
                       + w_bal * action_bal
                       + w_agg * action_agg)
            if action_max is not None and w_max > 0.0:
                blended = blended + w_max * action_max
        else:
            # v1d: weighted average of the POST-softmax portfolio weights.
            def to_weights(a):
                a = np.array(a, dtype=np.float64)
                a = a - a.max()             # numerical stability
                e = np.exp(a)
                return e / e.sum()
            w_d = to_weights(action_def)
            w_b = to_weights(action_bal)
            w_a = to_weights(action_agg)
            blended_weights = (w_def * w_d + w_bal * w_b + w_agg * w_a)
            if action_max is not None and w_max > 0.0:
                w_m = to_weights(action_max)
                blended_weights = blended_weights + w_max * w_m
            blended_weights = np.maximum(blended_weights, 1e-12)
            blended_weights = blended_weights / blended_weights.sum()
            blended = np.log(blended_weights).astype(np.float32)

        # Log for usage analysis
        self.usage_log.append({
            'w_defensive':  w_def,
            'w_balanced':   w_bal,
            'w_aggressive': w_agg,
            'w_max_bull':   w_max,
        })

        return blended, None

    def reset_log(self):
        self.usage_log = []

    def usage_summary(self):
        """Return mean weight given to each specialist over the trajectory."""
        if not self.usage_log:
            return None
        n = len(self.usage_log)
        out = {
            'defensive_mean_weight':  sum(r['w_defensive']  for r in self.usage_log) / n,
            'balanced_mean_weight':   sum(r['w_balanced']   for r in self.usage_log) / n,
            'aggressive_mean_weight': sum(r['w_aggressive'] for r in self.usage_log) / n,
            'days': n,
        }
        if self.max_bull is not None:
            out['max_bull_mean_weight'] = (
                sum(r.get('w_max_bull', 0.0) for r in self.usage_log) / n
            )
        return out


# Universe-specific specialist paths (relative to project root)
SPECIALISTS_DJI = {
    'defensive':  'models/saved/ppo_universal_v6_dji_defensive.zip',   # step 25k
    'balanced':   'models/saved/ppo_universal_v6_dji.zip',              # step 250k (now main)
    'aggressive': 'models/saved/ppo_universal_v6_dji_aggressive.zip',   # step 600k
    'max_bull':   'models/saved/ppo_universal_v6_dji_maxbull.zip',      # step 700k (38.71% bull)
}

SPECIALISTS_SP100 = {
    'defensive':  'models/saved/ppo_universal_v6_sp100_defensive.zip',  # step 25k
    'balanced':   'models/saved/ppo_universal_v6_sp100_balanced.zip',   # step 1M
    'aggressive': 'models/saved/ppo_universal_v6_sp100.zip',            # step 900k (now main)
    'max_bull':   'models/saved/ppo_universal_v6_sp100.zip',            # = aggressive (no
                                                                          # meaningful higher-bull
                                                                          # v6 SP100 checkpoint exists)
}


def build_controller(universe, temperature=1.0, blend_mode='logits',
                      use_max_bull=False):
    """Convenience: build a MARS-Pareto controller for a universe."""
    if universe == 'DJI 30':
        s = SPECIALISTS_DJI
    elif universe == 'SP100':
        s = SPECIALISTS_SP100
    else:
        raise ValueError(f"Unknown universe: {universe}")
    return MarsParetoController(
        defensive_path=s['defensive'],
        balanced_path=s['balanced'],
        aggressive_path=s['aggressive'],
        max_bull_path=(s['max_bull'] if use_max_bull else None),
        temperature=temperature,
        blend_mode=blend_mode,
    )


# ============================================================================
# v1c: Cross-variant specialists (different obs shapes per role)
# ============================================================================
#
# Each role's specialist comes from a different training run with a different
# feature set. We load three PPOs that each have their own observation_space;
# the caller hands us per-role obs dicts at predict time.
#
# Routing math identical to v1 — bull→aggressive, bear→defensive, sideways→
# balanced.
# ============================================================================

def _apply_training_floor(action, regime_id, floor_dict):
    """Mirror PortfolioEnv's cash-floor overlay on a raw action.

    v1c specialists trained against different cash_floors (env defaults for
    v1 FULL, half-floor for v2, zero for v6/v5). When we run them in a
    zero-floor master env, their actions need to be PRE-CONDITIONED with
    their training floor — otherwise v1 FULL goes naked in bear because it
    assumes the env will force cash on its behalf.
    """
    floor = float(floor_dict.get(int(regime_id), 0.0))
    if floor <= 0.0:
        return action
    a = np.array(action, dtype=np.float32)
    a = np.nan_to_num(a, nan=0.0, posinf=10.0, neginf=-10.0)
    a = a - a.max()
    w = np.exp(a)
    w = w / w.sum()
    if w[-1] < floor:
        stocks_total = w[:-1].sum()
        if stocks_total > 1e-8:
            target = 1.0 - floor
            w[:-1] *= (target / stocks_total)
        w[-1] = floor
    w = np.maximum(w, 1e-12)
    w = w / w.sum()
    return np.log(w).astype(np.float32)


class MarsParetoControllerV1c:
    """v1c: cross-variant specialists. Each role can have a different obs."""

    def __init__(self, defensive_path, balanced_path, aggressive_path,
                 cash_floors_per_role=None,
                 device='cpu', temperature=1.0, blend_mode='logits'):
        self.defensive  = PPO.load(defensive_path,  device=device)
        self.balanced   = PPO.load(balanced_path,   device=device)
        self.aggressive = PPO.load(aggressive_path, device=device)
        self.specialist_paths = {
            'defensive':  defensive_path,
            'balanced':   balanced_path,
            'aggressive': aggressive_path,
        }
        # Per-role training cash_floors. Each specialist's action gets
        # clamped to its OWN training floor before blending.
        self.cash_floors_per_role = cash_floors_per_role or {
            'defensive':  {0:0.0, 1:0.0, 2:0.0},
            'balanced':   {0:0.0, 1:0.0, 2:0.0},
            'aggressive': {0:0.0, 1:0.0, 2:0.0},
        }
        self.temperature = float(temperature)
        if blend_mode not in ('logits', 'weights'):
            raise ValueError(f"blend_mode must be 'logits' or 'weights'")
        self.blend_mode = blend_mode
        self.usage_log = []

    def predict(self, obs_dict, regime_probs, regime_id, deterministic=True):
        """
        obs_dict: {'defensive': obs_d, 'balanced': obs_b, 'aggressive': obs_a}
                  each obs may have a different shape (per-specialist envs).
        regime_probs: np.array([bull_p, bear_p, sideways_p])
        """
        bull_p, bear_p, sideways_p = regime_probs

        action_def, _ = self.defensive.predict(obs_dict['defensive'],
                                               deterministic=deterministic)
        action_bal, _ = self.balanced.predict(obs_dict['balanced'],
                                              deterministic=deterministic)
        action_agg, _ = self.aggressive.predict(obs_dict['aggressive'],
                                                deterministic=deterministic)

        # Pre-condition each action with the cash-floor it was trained
        # against. The master env runs at zero floor, so this restores
        # the per-specialist defensive bias that env-level enforcement
        # would have provided during training.
        action_def = _apply_training_floor(action_def, regime_id,
                                           self.cash_floors_per_role['defensive'])
        action_bal = _apply_training_floor(action_bal, regime_id,
                                           self.cash_floors_per_role['balanced'])
        action_agg = _apply_training_floor(action_agg, regime_id,
                                           self.cash_floors_per_role['aggressive'])

        raw = np.array([bull_p, bear_p, sideways_p], dtype=np.float32)
        if self.temperature != 1.0 and self.temperature > 0:
            log_p = np.log(np.maximum(raw, 1e-8))
            scaled = log_p / self.temperature
            scaled -= scaled.max()
            sharp = np.exp(scaled)
            sharp /= sharp.sum()
            w_agg, w_def, w_bal = float(sharp[0]), float(sharp[1]), float(sharp[2])
        else:
            w_agg = float(bull_p)
            w_def = float(bear_p)
            w_bal = float(sideways_p)

        if self.blend_mode == 'logits':
            blended = (w_def * action_def
                       + w_bal * action_bal
                       + w_agg * action_agg)
        else:
            def to_weights(a):
                a = np.array(a, dtype=np.float64)
                a = a - a.max()
                e = np.exp(a)
                return e / e.sum()
            w_d = to_weights(action_def)
            w_b = to_weights(action_bal)
            w_a = to_weights(action_agg)
            blended_weights = (w_def * w_d + w_bal * w_b + w_agg * w_a)
            blended_weights = np.maximum(blended_weights, 1e-12)
            blended_weights = blended_weights / blended_weights.sum()
            blended = np.log(blended_weights).astype(np.float32)

        self.usage_log.append({
            'w_defensive':  w_def,
            'w_balanced':   w_bal,
            'w_aggressive': w_agg,
        })
        return blended, None

    def reset_log(self):
        self.usage_log = []

    def usage_summary(self):
        if not self.usage_log:
            return None
        n = len(self.usage_log)
        return {
            'defensive_mean_weight':  sum(r['w_defensive']  for r in self.usage_log) / n,
            'balanced_mean_weight':   sum(r['w_balanced']   for r in self.usage_log) / n,
            'aggressive_mean_weight': sum(r['w_aggressive'] for r in self.usage_log) / n,
            'days': n,
        }


# v1c SYMMETRIC UNIVERSAL — 3 DISTINCT training recipes per role (not just
# step-checkpoints of one variant). Each role uses the SAME variant on both
# universes, so the design is universal even though the saved weights differ.
#
#   defensive  → v1 FULL  (legacy + 19 features + cash-floor env defaults)
#   balanced   → v2       (regime-λ + 20 features + bear-floor 0.5)
#   aggressive → v6       (no-floor + 20 features + regime-λ + vol features)
SPECIALISTS_DJI_V1C = {
    'defensive':  'models/saved/ppo_best_extended.zip',                 # v1 FULL DJI
    'balanced':   'models/saved/ppo_universal_v2_dji.zip',              # v2 DJI
    'aggressive': 'models/saved/ppo_universal_v6_dji_aggressive.zip',   # v6 DJI step 600k
}

SPECIALISTS_SP100_V1C = {
    'defensive':  'models/saved/ppo_best_extended_sp100.zip',           # v1 FULL SP100
    'balanced':   'models/saved/ppo_universal_v2_sp100.zip',            # v2 SP100
    'aggressive': 'models/saved/ppo_universal_v6_sp100.zip',            # v6 SP100 step 900k
}

# Per-role env configs.  Each role-key holds:
#   drop_cols: feature columns to remove from df (env auto-uses what remains)
#   env_flags: include_regime_probs / include_vol_features overrides
#
# Decoded from each model's observation_space.shape:
#   v1 FULL:   19 feat/stock, no probs, no vol   → drop mom_60d + v3 lags + trends
#   v2:        20 feat/stock, probs + vol         → drop v3 lags + trends
#   v5 agg:    29 feat/stock, probs + vol         → keep ALL cols
#   v6:        20 feat/stock, probs + vol         → drop v3 lags + trends
V3_COLS = ['r_lag_1_norm','r_lag_5_norm','r_lag_10_norm',
           'r_lag_20_norm','r_lag_40_norm','r_lag_60_norm',
           'is_uptrend','is_sideways','is_downtrend']

# With the symmetric universal lineup, BOTH universes use the same per-role
# env config (since the role-variants are universal):
#   defensive (v1 FULL):  19 feat/stock, NO probs, NO vol
#   balanced  (v2):       20 feat/stock, probs + vol
#   aggressive (v6):      20 feat/stock, probs + vol
_V1C_ENV_CONFIG_SYMMETRIC = {
    'defensive':  {'drop': V3_COLS + ['mom_60d_norm'],
                   'flags': dict(include_regime_probs=False, include_vol_features=False)},
    'balanced':   {'drop': V3_COLS,
                   'flags': dict(include_regime_probs=True,  include_vol_features=True)},
    'aggressive': {'drop': V3_COLS,
                   'flags': dict(include_regime_probs=True,  include_vol_features=True)},
}
V1C_ENV_CONFIG_DJI   = _V1C_ENV_CONFIG_SYMMETRIC
V1C_ENV_CONFIG_SP100 = _V1C_ENV_CONFIG_SYMMETRIC

# Per-role TRAINING cash_floors — universal under symmetric lineup:
#   v1 FULL:  env default = {0:0.0, 1:0.5, 2:0.1}
#   v2:       {0:0.0, 1:0.5, 2:0.0}  (bear-only floor)
#   v6:       {0:0.0, 1:0.0, 2:0.0}  (no floor)
_V1C_TRAINING_FLOORS_SYMMETRIC = {
    'defensive':  {0:0.0, 1:0.5, 2:0.1},   # v1 FULL recipe
    'balanced':   {0:0.0, 1:0.5, 2:0.0},   # v2 recipe (bear floor)
    'aggressive': {0:0.0, 1:0.0, 2:0.0},   # v6 recipe (no floor)
}
V1C_TRAINING_FLOORS_DJI   = _V1C_TRAINING_FLOORS_SYMMETRIC
V1C_TRAINING_FLOORS_SP100 = _V1C_TRAINING_FLOORS_SYMMETRIC


def build_controller_v1c(universe, temperature=1.0, blend_mode='logits'):
    """Build a v1c (cross-variant) controller for a universe."""
    if universe == 'DJI 30':
        s = SPECIALISTS_DJI_V1C
        floors = V1C_TRAINING_FLOORS_DJI
    elif universe == 'SP100':
        s = SPECIALISTS_SP100_V1C
        floors = V1C_TRAINING_FLOORS_SP100
    else:
        raise ValueError(f"Unknown universe: {universe}")
    return MarsParetoControllerV1c(
        defensive_path=s['defensive'],
        balanced_path=s['balanced'],
        aggressive_path=s['aggressive'],
        cash_floors_per_role=floors,
        temperature=temperature,
        blend_mode=blend_mode,
    )
