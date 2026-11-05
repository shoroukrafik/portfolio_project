"""
Strategy library for v4 hyper-heuristic agent.

Paper 8 (Cui et al. 2024) showed that for high-dimensional portfolio
problems (n stocks > ~50), direct asset-weight allocation by an RL
agent is **structurally bottlenecked** — softmax over n+1 actions
can't easily concentrate on a few winners.

The fix: agent picks 1 of K **pre-defined trading strategies** each
day. Each strategy is a deterministic function

    strategy(env_state) -> portfolio_weights  (length n_stocks + 1)

The agent's job becomes: "given today's state, which of K strategies
is most appropriate?" — a K-way classification problem with K ~ 6-8,
much easier than 91-dim continuous control.

The strategies below cover the spectrum of allocation philosophies:
  0. Equal weight (broad diversification)
  1. Top-K momentum (Paper 6/7 — momentum capture)
  2. Top-K low-volatility (defensive bull / sideways)
  3. Cash defensive (matches regime cash floor)
  4. All cash (max defensive)
  5. Bottom-K mean-reversion (contrarian)
  6. Top-K trend (uses is_uptrend tercile)
  7. Regime-aware blend (FULL-like baseline behaviour)
"""

import numpy as np


def _norm_weights(w):
    """Ensure weights sum to 1, are non-negative. Cash absorbs residual."""
    w = np.maximum(w, 0)
    s = w.sum()
    if s <= 0:
        # All-cash fallback
        out = np.zeros_like(w)
        out[-1] = 1.0
        return out
    return w / s


def equal_weight(n_stocks, n_actions, **_):
    """Strategy 0: 1/n in every stock, 0 cash."""
    w = np.ones(n_actions, dtype=np.float32)
    w[-1] = 0.0  # no cash
    return _norm_weights(w)


def top_k_momentum(n_stocks, n_actions, mom_scores, k=10, **_):
    """Strategy 1: equal-weight top-k by mom_60d. Goes long the winners.

    mom_scores: array of length n_stocks with each stock's 60-day momentum
                (raw or normalised — only the ranking matters).
    """
    w = np.zeros(n_actions, dtype=np.float32)
    k = min(k, n_stocks)
    if mom_scores is not None and len(mom_scores) == n_stocks:
        top_idx = np.argsort(mom_scores)[-k:]
        w[top_idx] = 1.0 / k
    else:
        # Fall back to equal-weight if no signal
        w[:n_stocks] = 1.0 / n_stocks
    return _norm_weights(w)


def top_k_low_vol(n_stocks, n_actions, vol_scores, k=10, **_):
    """Strategy 2: equal-weight bottom-k by realised vol (lowest-vol stocks).
    Defensive bull / sideways play.

    vol_scores: per-stock recent volatility (e.g., 20-day std of returns).
    """
    w = np.zeros(n_actions, dtype=np.float32)
    k = min(k, n_stocks)
    if vol_scores is not None and len(vol_scores) == n_stocks:
        # Use absolute value in case some are NaN; replace with large value to deprioritise
        scores = np.where(np.isnan(vol_scores), 1e9, vol_scores)
        low_idx = np.argsort(scores)[:k]
        w[low_idx] = 1.0 / k
    else:
        w[:n_stocks] = 1.0 / n_stocks
    return _norm_weights(w)


def cash_defensive(n_stocks, n_actions, regime_id, **_):
    """Strategy 3: regime-conditioned cash floor + equal-weight stocks.

    Bull (0)      -> 5% cash, 95% spread evenly
    Bear (1)      -> 70% cash, 30% spread evenly
    Sideways (2)  -> 30% cash, 70% spread evenly
    """
    cash_by_regime = {0: 0.05, 1: 0.70, 2: 0.30}
    cash_pct = cash_by_regime.get(int(regime_id), 0.30)
    stock_pct = 1.0 - cash_pct
    w = np.zeros(n_actions, dtype=np.float32)
    w[:n_stocks] = stock_pct / n_stocks
    w[-1] = cash_pct
    return _norm_weights(w)


def all_cash(n_stocks, n_actions, **_):
    """Strategy 4: 100% cash. Maximum defense."""
    w = np.zeros(n_actions, dtype=np.float32)
    w[-1] = 1.0
    return w


def bottom_k_mean_reversion(n_stocks, n_actions, mom_scores, k=10, **_):
    """Strategy 5: equal-weight bottom-k by mom_60d. Buy the laggards.

    Contrarian; tends to win in sideways/oversold conditions.
    """
    w = np.zeros(n_actions, dtype=np.float32)
    k = min(k, n_stocks)
    if mom_scores is not None and len(mom_scores) == n_stocks:
        bot_idx = np.argsort(mom_scores)[:k]
        w[bot_idx] = 1.0 / k
    else:
        w[:n_stocks] = 1.0 / n_stocks
    return _norm_weights(w)


def top_k_trend(n_stocks, n_actions, is_uptrend_mask, k=10, **_):
    """Strategy 6: equal-weight up to k stocks tagged is_uptrend = 1.

    is_uptrend_mask: array of length n_stocks, 1 if in top tercile of
                    today's cross-sectional 60d momentum (computed in
                    add_v3_features.py).
    """
    w = np.zeros(n_actions, dtype=np.float32)
    if is_uptrend_mask is not None and len(is_uptrend_mask) == n_stocks:
        idx = np.where(is_uptrend_mask > 0)[0]
        if len(idx) > 0:
            picks = idx[:k]
            w[picks] = 1.0 / len(picks)
            return _norm_weights(w)
    # Fall back to equal-weight
    w[:n_stocks] = 1.0 / n_stocks
    return _norm_weights(w)


def regime_aware_blend(n_stocks, n_actions, regime_id,
                       mom_scores=None, **_):
    """Strategy 7: FULL-like behaviour.

    Bull → top-15 momentum, 0% cash
    Bear → 50% cash, equal-weight rest
    Sideways → top-20 by momentum, 0% cash
    """
    regime_id = int(regime_id)
    if regime_id == 0:  # bull
        return top_k_momentum(n_stocks, n_actions, mom_scores, k=15)
    elif regime_id == 1:  # bear
        return cash_defensive(n_stocks, n_actions, regime_id=1)
    else:  # sideways
        return top_k_momentum(n_stocks, n_actions, mom_scores, k=20)


# Registry — order matters: this is the agent's discrete action space.
STRATEGY_REGISTRY = [
    ('equal_weight',          equal_weight),
    ('top_k_momentum',        top_k_momentum),
    ('top_k_low_vol',         top_k_low_vol),
    ('cash_defensive',        cash_defensive),
    ('all_cash',              all_cash),
    ('bottom_k_mean_revert',  bottom_k_mean_reversion),
    ('top_k_trend',           top_k_trend),
    ('regime_aware_blend',    regime_aware_blend),
]

N_STRATEGIES = len(STRATEGY_REGISTRY)


def apply_strategy(strategy_idx, n_stocks, n_actions, context):
    """Dispatch to the chosen strategy function.

    context: dict with whatever signals strategies need:
        - regime_id        : int 0/1/2
        - mom_scores       : np.ndarray (n_stocks,) — today's 60-day return per stock
        - vol_scores       : np.ndarray (n_stocks,) — today's 20-day realised vol
        - is_uptrend_mask  : np.ndarray (n_stocks,) — 1 if in top tercile

    Returns: weights vector of length n_actions.
    """
    if strategy_idx < 0 or strategy_idx >= N_STRATEGIES:
        # Out-of-range guard: fall back to equal-weight
        return equal_weight(n_stocks, n_actions, **context)
    _, func = STRATEGY_REGISTRY[strategy_idx]
    return func(n_stocks=n_stocks, n_actions=n_actions, **context)


def strategy_names():
    return [name for name, _ in STRATEGY_REGISTRY]
