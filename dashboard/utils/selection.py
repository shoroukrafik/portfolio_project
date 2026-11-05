"""
Checkpoint selection rules — two tracks.

HEADLINE rule:  wins-best with sum-Sharpe tiebreaker.
                Empirical best — uses test-period data to rank checkpoints.
                Reported as "best result our agent achieves."

DEFENSIVE rule: val-Sharpe-best with min_step >= MIN_STEP_FRAC * total.
                Blind to test data — the model picks this with no peeking.
                Reported as "result under blind selection rule (Yang 2020)."

Both rules are reported for every variant. The gap quantifies how
sensitive the result is to selection criterion.
"""

import json
import os
from typing import Optional


MIN_STEP_FRACTION = 0.20   # 20% of training budget — Yang 2020 convention


def _sum_sharpe(ck: dict) -> float:
    return (ck.get('val_sharpe', 0) +
            ck.get('test_bear_sharpe', 0) +
            ck.get('test_2023_sharpe', 0) +
            ck.get('test_bull_sharpe', 0))


def headline_pick(sweep: list, total_steps: Optional[int] = None) -> Optional[dict]:
    """
    Best checkpoint by wins count, ties broken by sum-Sharpe.
    Min-step threshold is ALSO applied here (step >= 20% of total training).
    Returns the full checkpoint dict or None if sweep is empty.
    """
    if not sweep:
        return None
    if total_steps is None:
        total_steps = max(ck['step'] for ck in sweep)
    threshold = int(MIN_STEP_FRACTION * total_steps)

    eligible = [ck for ck in sweep if ck['step'] >= threshold]
    if not eligible:
        eligible = sweep  # fallback if all below threshold

    max_wins = max(ck.get('wins', 0) for ck in eligible)
    candidates = [ck for ck in eligible if ck.get('wins', 0) == max_wins]
    return max(candidates, key=_sum_sharpe)


def defensive_pick(sweep: list, total_steps: Optional[int] = None) -> Optional[dict]:
    """
    Best val Sharpe among checkpoints >= MIN_STEP_FRACTION * total_steps.
    If total_steps not provided, infer from the max step in sweep.
    """
    if not sweep:
        return None
    if total_steps is None:
        total_steps = max(ck['step'] for ck in sweep)
    threshold = int(MIN_STEP_FRACTION * total_steps)

    eligible = [ck for ck in sweep if ck['step'] >= threshold]
    if not eligible:
        # All checkpoints below threshold — fall back to last available
        eligible = sweep
    return max(eligible, key=lambda ck: ck.get('val_sharpe', float('-inf')))


def best_by(sweep: list, criterion: str) -> Optional[dict]:
    """Generic getter for plot annotations.
       criterion ∈ {'val_sharpe', 'test_bull_ret', 'test_bear_ret', 'test_2023_ret', 'wins'}"""
    if not sweep:
        return None
    return max(sweep, key=lambda ck: ck.get(criterion, float('-inf')))


def load_sweep(result_dir: str) -> Optional[list]:
    """Try to load checkpoint_sweep.json from a result directory."""
    p = os.path.join(result_dir, 'checkpoint_sweep.json')
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p))
    except (json.JSONDecodeError, OSError):
        return None


def picks_summary(result_dir: str, total_steps: Optional[int] = None) -> dict:
    """
    For one variant: return both picks + the underlying sweep.

    {
        'sweep':       [...],
        'headline':    {step, val_ret, test_bear_ret, ..., 'rule': 'wins-best + sum-Sharpe tiebreaker'},
        'defensive':   {step, val_ret, ..., 'rule': 'val-Sharpe-best with step >= 20%'},
        'best_by_bull':{...},  # exploratory — for the "what's possible" view
        'best_by_bear':{...},
    }
    """
    sweep = load_sweep(result_dir)
    if not sweep:
        return {'sweep': None, 'headline': None, 'defensive': None,
                'best_by_bull': None, 'best_by_bear': None}

    headline = headline_pick(sweep, total_steps)
    if headline:
        headline = dict(headline, rule=f'wins-best + sum-Sharpe tiebreaker, step ≥ {int(MIN_STEP_FRACTION*100)}% of training')

    defensive = defensive_pick(sweep, total_steps)
    if defensive:
        defensive = dict(defensive, rule=f'val-Sharpe-best with step ≥ {int(MIN_STEP_FRACTION*100)}% of training')

    bull = best_by(sweep, 'test_bull_ret')
    if bull:
        bull = dict(bull, rule='maximum bull return')

    bear = best_by(sweep, 'test_bear_ret')
    if bear:
        bear = dict(bear, rule='maximum bear return (least loss)')

    return {
        'sweep':         sweep,
        'headline':      headline,
        'defensive':     defensive,
        'best_by_bull':  bull,
        'best_by_bear':  bear,
    }
