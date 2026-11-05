"""
Phase 2 — Drawdown-aware soft-switching for the MTL agent.

Idea: take the honest early-plateau MTL checkpoint (the one that already
matches FULL on bear and only loses on bull) and add a portfolio-state-
aware adjustment to the HMM probabilities BEFORE feeding the observation
to the model.

Mechanism (inference-only, no retraining):
  - Track running portfolio peak and current drawdown
  - When drawdown exceeds a threshold (default 3%), shift weight away
    from the bull/sideways heads toward the bear head
  - Replace the regime-prob slice of the observation (obs[573:576]) with
    the adjusted probabilities and forward through the model normally

This is a fast safety signal that complements the (slow) HMM regime
gate. The HMM says "what regime do I think we're in?"; the drawdown
adjustment says "regardless of what HMM says, the portfolio is
bleeding — defend."

We evaluate three variants for comparison:
  - Standard MTL (no DD adjustment)        ← sanity baseline
  - DD-aware (threshold=3%)                ← Phase 2 main
  - DD-aware aggressive (threshold=1.5%)   ← sensitivity check

Results are saved to results/mtl_extended/dd_aware_eval.json.
"""

import json
import os
import sys
import copy

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from portfolio_env import PortfolioEnv
from mtl_components import AttentionMTLExtractor
from train_final_extended import calculate_metrics


# Observation slice positions (must match the MTL extractor)
PROBS_SLICE = slice(573, 576)


def dd_adjust_probabilities(probs, drawdown_pct, threshold=3.0,
                            scaling=7.0, max_shift=0.5):
    """
    Shift mass from bull/sideways to bear based on current drawdown.

    Args:
        probs        : np.array shape (3,) — [p_bull, p_bear, p_sideways]
        drawdown_pct : current drawdown as a percentage (e.g. 5.0 means -5%)
        threshold    : drawdown % at which to start shifting (default 3%)
        scaling      : drawdown % at which full shift is reached (default 10%)
        max_shift    : maximum fraction of bull+sideways weight to move
                       into bear (default 0.5)

    Returns: np.array shape (3,) — adjusted probabilities (sum to 1)
    """
    if drawdown_pct <= threshold:
        return probs.copy()  # no adjustment

    # Linear ramp from 0 at threshold to max_shift at threshold+scaling
    excess = drawdown_pct - threshold
    shift_factor = min(excess / scaling, 1.0) * max_shift

    # Move shift_factor of (bull + sideways) mass into bear
    p_bull, p_bear, p_sideways = probs[0], probs[1], probs[2]
    move = shift_factor * (p_bull + p_sideways)
    p_bull_new     = p_bull * (1 - shift_factor)
    p_sideways_new = p_sideways * (1 - shift_factor)
    p_bear_new     = p_bear + move

    new_probs = np.array([p_bull_new, p_bear_new, p_sideways_new],
                         dtype=np.float32)
    # Renormalize (should already sum to 1 but float math)
    new_probs /= new_probs.sum()
    return new_probs


def evaluate_dd_aware(model, env, threshold=3.0, scaling=7.0, max_shift=0.5,
                     initial_balance=1_000_000):
    """
    Run an episode with drawdown-aware soft-switching applied to the
    observation BEFORE each model.predict call.
    """
    obs, _ = env.reset()
    done = False
    portfolio_peak = float(initial_balance)
    dd_engagement_count = 0
    n_steps = 0
    dd_history = []
    shift_history = []

    while not done:
        # Read current portfolio value from env
        port_val = float(env.portfolio_value)
        if port_val > portfolio_peak:
            portfolio_peak = port_val
        drawdown_pct = (1.0 - port_val / portfolio_peak) * 100.0

        # Adjust the HMM-prob slice of the observation
        original_probs = obs[PROBS_SLICE].copy()
        new_probs = dd_adjust_probabilities(
            original_probs, drawdown_pct,
            threshold=threshold, scaling=scaling, max_shift=max_shift,
        )
        obs_modified = obs.copy()
        obs_modified[PROBS_SLICE] = new_probs

        # Track engagement statistics
        dd_history.append(drawdown_pct)
        shift = float(new_probs[1] - original_probs[1])  # how much extra into bear
        shift_history.append(shift)
        if drawdown_pct > threshold:
            dd_engagement_count += 1
        n_steps += 1

        action, _ = model.predict(obs_modified, deterministic=True)
        obs, _, term, trunc, _ = env.step(action)
        done = term or trunc

    history = env.get_portfolio_history()
    return {
        'history':              history,
        'engagement_pct':       round(100.0 * dd_engagement_count / n_steps, 1),
        'max_drawdown_seen':    round(max(dd_history), 2),
        'mean_bear_shift':      round(float(np.mean(shift_history)), 4),
        'max_bear_shift':       round(float(np.max(shift_history)), 4),
    }


def evaluate_standard(model, env):
    """Plain MTL inference — no DD adjustment. For sanity baseline."""
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(action)
        done = term or trunc
    return {'history': env.get_portfolio_history()}


def main():
    base_dir    = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)
    data_dir    = os.path.join(project_dir, 'data')
    saved_dir   = os.path.join(base_dir, 'saved')
    ckpt_dir    = os.path.join(saved_dir, 'checkpoints', 'mtl_extended')
    results_dir = os.path.join(project_dir, 'results', 'mtl_extended')

    print("=" * 72)
    print("   Phase 2 — Drawdown-aware soft-switching for MTL")
    print("=" * 72)

    # ---- Pick the honest checkpoint (early plateau, step 400k) ----
    target_step = 400_000
    ckpt_path = os.path.join(ckpt_dir, f'step_{target_step:07d}')
    if not os.path.exists(ckpt_path + '.zip'):
        print(f"❌ Missing checkpoint: {ckpt_path}.zip")
        return
    print(f"\nUsing early-plateau MTL checkpoint: step {target_step:,}")
    model = PPO.load(ckpt_path, device='cpu',
                     custom_objects={'features_extractor_class':
                                     AttentionMTLExtractor})

    # ---- Load test data ----
    val_df       = pd.read_csv(os.path.join(data_dir, 'val_data_ext.csv'))
    test_bear_df = pd.read_csv(os.path.join(data_dir, 'test_bear_data_ext.csv'))
    test_bull_df = pd.read_csv(os.path.join(data_dir, 'test_bull_data_ext.csv'))
    for df in (val_df, test_bear_df, test_bull_df):
        df['date'] = pd.to_datetime(df['date'])

    env_kw = dict(initial_balance=1_000_000, include_regime_probs=True)

    # ---- DD variants to test ----
    variants = [
        ('Standard MTL (no DD)',        None),
        ('DD-aware  threshold=3%',      dict(threshold=3.0,  scaling=7.0, max_shift=0.5)),
        ('DD-aware  threshold=1.5%',    dict(threshold=1.5,  scaling=5.0, max_shift=0.5)),
    ]

    all_results = {}
    for label, dd_params in variants:
        print(f"\n--- {label} ---")
        per_period = {}
        for name, df in [('val', val_df),
                         ('test_bear', test_bear_df),
                         ('test_bull', test_bull_df)]:
            env = PortfolioEnv(df=df, **env_kw)
            if dd_params is None:
                out = evaluate_standard(model, env)
                m = calculate_metrics(out['history'])
                extra = {}
            else:
                out = evaluate_dd_aware(model, env, **dd_params)
                m = calculate_metrics(out['history'])
                extra = {k: out[k] for k in
                         ('engagement_pct', 'max_drawdown_seen',
                          'mean_bear_shift', 'max_bear_shift')}
            per_period[name] = {**m, **extra}
            if extra:
                print(f"  {name:<10}: Return={m['cumulative_return']:>7.2f}%  "
                      f"Sharpe={m['sharpe_ratio']:>7.4f}  "
                      f"MDD={m['max_drawdown']:>7.2f}%  "
                      f"DD-engage={extra['engagement_pct']}%  "
                      f"avg_shift={extra['mean_bear_shift']:+.3f}")
            else:
                print(f"  {name:<10}: Return={m['cumulative_return']:>7.2f}%  "
                      f"Sharpe={m['sharpe_ratio']:>7.4f}  "
                      f"MDD={m['max_drawdown']:>7.2f}%")
        all_results[label] = per_period

    # ---- Save ----
    out_path = os.path.join(results_dir, 'dd_aware_eval.json')
    with open(out_path, 'w') as f:
        json.dump({'checkpoint_step': target_step,
                   'variants': all_results}, f, indent=2)
    print(f"\n✅ Saved → {out_path}")

    # ---- Comparison table vs FULL ----
    full_path = os.path.join(project_dir, 'results', 'extended',
                             'final_results.json')
    full = json.load(open(full_path)) if os.path.exists(full_path) else None

    print("\n" + "=" * 90)
    print("   Phase 2 results — FULL vs early-MTL vs DD-aware MTL")
    print("=" * 90)
    print(f"  {'Period':<10} {'System':<32} "
          f"{'Return%':>9} {'Sharpe':>8} {'MDD%':>9}")
    print("  " + "-" * 78)
    for period in ['val', 'test_bear', 'test_bull']:
        if full is not None:
            b = full[f'agent_{period}']
            print(f"  {period:<10} {'FULL (single agent)':<32} "
                  f"{b['cumulative_return']:>9.2f} "
                  f"{b['sharpe_ratio']:>8.4f} "
                  f"{b['max_drawdown']:>9.2f}")
        for label, per_period in all_results.items():
            m = per_period[period]
            print(f"  {period:<10} {f'MTL: {label}':<32} "
                  f"{m['cumulative_return']:>9.2f} "
                  f"{m['sharpe_ratio']:>8.4f} "
                  f"{m['max_drawdown']:>9.2f}")
        print()

    print("=" * 72)
    print("   Look at bear-2022 — that's the metric DD-aware was designed for.")
    print("=" * 72 + "\n")


if __name__ == '__main__':
    main()
