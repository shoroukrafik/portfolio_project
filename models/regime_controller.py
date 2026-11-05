# ============================================
# REGIME CONTROLLER (hard switch)
# ============================================
# At each timestep, look at the HMM regime label for that day and
# route the observation to the matching specialist agent. The
# selected specialist returns the action; the others sit idle.
#
# Hard-switch design:
#   regime_id == 0 (bull)     → bull specialist
#   regime_id == 1 (bear)     → bear specialist
#   regime_id == 2 (sideways) → sideways specialist
#
# Soft-blending (weighted mixture of all 3) is intentionally NOT
# implemented here — that would be a follow-up Multi-Agent v2 study.
# ============================================

import numpy as np

REGIME_NAMES = {0: 'bull', 1: 'bear', 2: 'sideways'}


class RegimeController:
    """
    Hard-switch controller. Holds a dict of {regime_id: PPO model} and
    forwards predictions to the specialist matching today's regime.

    Usage:
        ctrl = RegimeController({
            0: bull_model,
            1: bear_model,
            2: sideways_model,
        })
        action, regime_used = ctrl.predict(obs, regime_id_today)
    """

    def __init__(self, specialists):
        for k in (0, 1, 2):
            if k not in specialists:
                raise ValueError(f"Missing specialist for regime {k} ({REGIME_NAMES[k]})")
        self.specialists = specialists

        # Track which regime got called each step — useful for thesis plots
        # ("how often was each specialist active in each test period?")
        self.regime_call_counts = {0: 0, 1: 0, 2: 0}

    def predict(self, obs, regime_id, deterministic=True):
        """
        HARD switch: pick the specialist for `regime_id` and ask it.

        Returns (action, regime_used_int).
        """
        regime_id = int(regime_id)
        if regime_id not in self.specialists:
            raise ValueError(f"Unknown regime_id {regime_id}")
        action, _ = self.specialists[regime_id].predict(
            obs, deterministic=deterministic
        )
        self.regime_call_counts[regime_id] += 1
        return action, regime_id

    def predict_soft(self, obs, regime_probs, deterministic=True):
        """
        SOFT blend: query all specialists, return weighted average action.

        regime_probs: array-like of length 3, in order [bull, bear, sideways],
                      sums to 1 (HMM posterior probabilities).

        Each specialist outputs RAW action scores (before softmax). We blend
        them linearly by regime_probs; the env applies softmax inside step()
        so the resulting action is automatically a valid portfolio.

        We also bookkeep specialist usage proportionally — a day with probs
        [0.4, 0.1, 0.5] adds 0.4 to bull's call count, 0.1 to bear's, 0.5 to
        sideways's. So the usage_summary still reports total specialist
        contribution percentage.
        """
        regime_probs = np.asarray(regime_probs, dtype=np.float32)
        if regime_probs.shape[0] != 3:
            raise ValueError(f"regime_probs must have length 3, got "
                             f"{regime_probs.shape}")
        # Renormalize defensively (should already sum to 1)
        s = regime_probs.sum()
        if s <= 0:
            raise ValueError("regime_probs sums to <= 0")
        regime_probs = regime_probs / s

        # Query all 3 specialists
        actions = []
        for rid in (0, 1, 2):
            a, _ = self.specialists[rid].predict(obs, deterministic=deterministic)
            actions.append(np.asarray(a, dtype=np.float32))

        # Linear blend on raw action scores
        blended = (regime_probs[0] * actions[0]
                 + regime_probs[1] * actions[1]
                 + regime_probs[2] * actions[2])

        # Fractional usage tracking — preserves "total contribution" semantics
        for rid in (0, 1, 2):
            self.regime_call_counts[rid] += float(regime_probs[rid])

        return blended, regime_probs.tolist()

    def reset_counts(self):
        self.regime_call_counts = {0: 0, 1: 0, 2: 0}

    def usage_summary(self):
        """Return per-regime % of calls. Useful after one episode."""
        total = sum(self.regime_call_counts.values())
        if total == 0:
            return {0: 0.0, 1: 0.0, 2: 0.0}
        return {k: round(100.0 * v / total, 2) for k, v in self.regime_call_counts.items()}
