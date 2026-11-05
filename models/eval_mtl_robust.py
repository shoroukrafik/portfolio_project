"""
Apply rolling-mean checkpoint selection to the MTL training run.

The single-eval val-Sharpe peak selection in the trainer is prone to
late-training spikes (as we saw in the first MTL run: peak at step 950k
with val Sharpe 3.47, but training reward was crashing simultaneously
— textbook overfitting).

This script:
  1. Loads val_sharpe_history.json from the MTL training output
  2. Computes a 5-eval trailing rolling mean
  3. Picks the step whose rolling-mean Sharpe is highest
  4. Loads THAT checkpoint from models/saved/checkpoints/mtl_extended/
  5. Evaluates it on val / bear / bull
  6. Saves results next to the original final_results.json
  7. Prints comparison: spike-selected vs rolling-mean-selected vs FULL

Requires that the trainer was run WITH checkpoint_dir enabled
(per-eval checkpoints saved in models/saved/checkpoints/mtl_extended/).

No retraining — operates on existing artifacts.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from portfolio_env import PortfolioEnv
from mtl_components import AttentionMTLExtractor
from train_final_extended import calculate_metrics, evaluate_agent


WINDOW = 5   # number of consecutive evals to average


def trailing_rolling_mean(values, window):
    v = np.asarray(values, dtype=float)
    out = np.full_like(v, np.nan)
    for i in range(window - 1, len(v)):
        out[i] = v[i - window + 1: i + 1].mean()
    return out


def main():
    base_dir    = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)
    data_dir    = os.path.join(project_dir, 'data')
    saved_dir   = os.path.join(base_dir, 'saved')
    ckpt_dir    = os.path.join(saved_dir, 'checkpoints', 'mtl_extended')
    results_dir = os.path.join(project_dir, 'results', 'mtl_extended')

    print("=" * 70)
    print("   MTL — rolling-mean checkpoint selection")
    print("=" * 70)

    # ---- Load training history ----
    hist_path = os.path.join(results_dir, 'val_sharpe_history.json')
    if not os.path.exists(hist_path):
        print(f"❌ Missing {hist_path}. Run train_multi_agent_mtl.py first.")
        return
    raw = json.load(open(hist_path))
    h = raw['history']
    steps   = np.array([x[0] for x in h])
    sharpes = np.array([x[1] for x in h])

    # ---- Identify candidates ----
    single_idx = int(np.argmax(sharpes))
    single_step = int(steps[single_idx])

    rolling = trailing_rolling_mean(sharpes, WINDOW)
    valid = ~np.isnan(rolling)
    roll_idx = int(np.argmax(np.where(valid, rolling, -np.inf)))
    roll_step = int(steps[roll_idx])

    # Also report the early plateau peak (≤ 500k) — common diagnostic
    early_mask = steps <= 500_000
    early_best = int(np.argmax(np.where(early_mask, sharpes, -np.inf)))
    early_step = int(steps[early_best])

    print(f"\nSelection candidates:")
    print(f"  Saved (single-eval peak)     step={single_step:>7,}  "
          f"Sharpe={sharpes[single_idx]:.4f}")
    print(f"  Rolling-mean peak ({WINDOW}-eval)   step={roll_step:>7,}  "
          f"Sharpe={sharpes[roll_idx]:.4f}  rolling_mean={rolling[roll_idx]:.4f}")
    print(f"  Early-plateau peak (≤500k)   step={early_step:>7,}  "
          f"Sharpe={sharpes[early_best]:.4f}")

    # ---- Load test envs ----
    val_df       = pd.read_csv(os.path.join(data_dir, 'val_data_ext.csv'))
    test_bear_df = pd.read_csv(os.path.join(data_dir, 'test_bear_data_ext.csv'))
    test_bull_df = pd.read_csv(os.path.join(data_dir, 'test_bull_data_ext.csv'))
    for df in (val_df, test_bear_df, test_bull_df):
        df['date'] = pd.to_datetime(df['date'])
    env_kw = dict(initial_balance=1_000_000, include_regime_probs=True)
    val_env       = PortfolioEnv(df=val_df,       **env_kw)
    test_bear_env = PortfolioEnv(df=test_bear_df, **env_kw)
    test_bull_env = PortfolioEnv(df=test_bull_df, **env_kw)

    # ---- Evaluate each candidate ----
    candidates = [
        ('Saved (single-peak spike)', single_step),
        ('Rolling-mean peak',         roll_step),
        ('Early-plateau peak (≤500k)', early_step),
    ]
    all_results = {}
    for label, step in candidates:
        ckpt = os.path.join(ckpt_dir, f'step_{step:07d}')
        if not os.path.exists(ckpt + '.zip'):
            print(f"\n⚠ missing checkpoint for step {step}: {ckpt}.zip")
            continue
        print(f"\n--- Evaluating: {label}  (step {step:,}) ---")
        model = PPO.load(ckpt, device='cpu',
                         custom_objects={'features_extractor_class':
                                         AttentionMTLExtractor})
        per_period = {}
        for name, env in [('val', val_env),
                          ('test_bear', test_bear_env),
                          ('test_bull', test_bull_env)]:
            h_run = evaluate_agent(model, env)
            m = calculate_metrics(h_run)
            per_period[name] = m
            print(f"  {name:<10}: Return={m['cumulative_return']:>7.2f}%  "
                  f"Sharpe={m['sharpe_ratio']:>7.4f}  "
                  f"MDD={m['max_drawdown']:>7.2f}%")
        all_results[label] = {'step': step, 'metrics': per_period}

    # ---- Save ----
    out_path = os.path.join(results_dir, 'rolling_mean_eval.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Saved → {out_path}")

    # ---- Comparison table vs FULL ----
    full_path = os.path.join(project_dir, 'results', 'extended',
                             'final_results.json')
    full = json.load(open(full_path)) if os.path.exists(full_path) else None

    print("\n" + "=" * 90)
    print("   MTL — checkpoint comparison")
    print("=" * 90)
    print(f"  {'Period':<10} {'Selection':<32} "
          f"{'Return%':>9} {'Sharpe':>8} {'MDD%':>9}")
    print("  " + "-" * 78)
    for period in ['val', 'test_bear', 'test_bull']:
        if full is not None:
            b = full[f'agent_{period}']
            print(f"  {period:<10} {'FULL (single agent)':<32} "
                  f"{b['cumulative_return']:>9.2f} "
                  f"{b['sharpe_ratio']:>8.4f} "
                  f"{b['max_drawdown']:>9.2f}")
        for label, info in all_results.items():
            m = info['metrics'][period]
            print(f"  {period:<10} {f'MTL: {label}':<32} "
                  f"{m['cumulative_return']:>9.2f} "
                  f"{m['sharpe_ratio']:>8.4f} "
                  f"{m['max_drawdown']:>9.2f}")
        print()

    print("=" * 70)
    print("   Pick the row that survives the rolling-mean robustness check.")
    print("   That is the honest MTL result for thesis comparison.")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
