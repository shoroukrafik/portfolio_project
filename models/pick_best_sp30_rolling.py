"""
Rolling-mean checkpoint diagnostic for FULL-on-SP30.

The training-time val-best was saved at step 25,000 with val Sharpe 3.45.
That's suspiciously high (DJI 30 FULL plateaus at 2.39) and the test-set
performance is much weaker than DJI 30, which is the classic val-overfit
signature we saw with MTL v1.

This script applies the same diagnostic that pick_best_bear / MTL v1
used:

  1. Load val_sharpe_history from the SP30 training run.
  2. Compute 5-eval trailing rolling mean of val Sharpe.
  3. Identify three candidate selection points:
        a) Single-eval peak  — what was saved (step 25k, Sharpe 3.45)
        b) Rolling-mean peak — most stable plateau
        c) Early-plateau peak (≤500k) — first half of training
  4. Load each candidate checkpoint from
       models/saved/checkpoints/full_extended_sp30/
  5. Evaluate each on val 2021, bear 2022, 2023, bull 2024.
  6. Print a comparison table.

If rolling-mean / early-plateau perform substantially better on test
than the saved spike → selection artifact, replace the canonical save.
If all three give similar (poor) test results → genuine external
validity failure, document and accept.

NO TEST DATA used for selection — only val.

Outputs:
  results/extended_sp30/rolling_mean_eval.json    — per-candidate test metrics
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from portfolio_env import PortfolioEnv
from train_final_extended import calculate_metrics, evaluate_agent


WINDOW = 5


def trailing_mean(values, window):
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
    ckpt_dir    = os.path.join(saved_dir, 'checkpoints', 'full_extended_sp30')
    results_dir = os.path.join(project_dir, 'results', 'extended_sp30')

    print("=" * 78)
    print("   SP30 FULL — rolling-mean checkpoint diagnostic")
    print("=" * 78)

    # ---- Load training trajectory ----
    hist_path = os.path.join(results_dir, 'val_sharpe_history.json')
    if not os.path.exists(hist_path):
        print(f"❌ Missing {hist_path}. Did training run?")
        return
    h = json.load(open(hist_path))['history']
    steps   = np.array([x[0] for x in h])
    sharpes = np.array([x[1] for x in h])

    # ---- Identify three candidates ----
    single_idx = int(np.argmax(sharpes))
    single_step = int(steps[single_idx])

    rolling = trailing_mean(sharpes, WINDOW)
    valid = ~np.isnan(rolling)
    roll_idx = int(np.argmax(np.where(valid, rolling, -np.inf)))
    roll_step = int(steps[roll_idx])

    early_mask = steps <= 500_000
    early_best = int(np.argmax(np.where(early_mask, sharpes, -np.inf)))
    early_step = int(steps[early_best])

    print(f"\nCandidates from val_sharpe_history:")
    print(f"  Saved (single-eval peak)     step={single_step:>7,}  "
          f"Sharpe={sharpes[single_idx]:.4f}")
    print(f"  Rolling-mean peak ({WINDOW}-eval)   step={roll_step:>7,}  "
          f"single-Sharpe={sharpes[roll_idx]:.4f}  rolling={rolling[roll_idx]:.4f}")
    print(f"  Early-plateau peak (≤500k)   step={early_step:>7,}  "
          f"Sharpe={sharpes[early_best]:.4f}")

    # ---- Load test envs ----
    print("\n--- Loading SP30 test data ---")
    val_df       = pd.read_csv(os.path.join(data_dir, 'val_data_sp30.csv'))
    test_bear_df = pd.read_csv(os.path.join(data_dir, 'test_bear_data_sp30.csv'))
    test_2023_df = pd.read_csv(os.path.join(data_dir, 'test_2023_data_sp30.csv'))
    test_bull_df = pd.read_csv(os.path.join(data_dir, 'test_bull_data_sp30.csv'))
    for df in (val_df, test_bear_df, test_2023_df, test_bull_df):
        df['date'] = pd.to_datetime(df['date'])
    val_env       = PortfolioEnv(df=val_df,       initial_balance=1_000_000)
    test_bear_env = PortfolioEnv(df=test_bear_df, initial_balance=1_000_000)
    test_2023_env = PortfolioEnv(df=test_2023_df, initial_balance=1_000_000)
    test_bull_env = PortfolioEnv(df=test_bull_df, initial_balance=1_000_000)

    # ---- Evaluate each candidate ----
    candidates = [
        ('Saved (single-peak)',        single_step),
        ('Rolling-mean peak',          roll_step),
        ('Early-plateau peak (≤500k)', early_step),
    ]
    # De-duplicate (some candidates may pick the same step)
    seen = set()
    unique_candidates = []
    for label, step in candidates:
        if step in seen:
            print(f"  (note: '{label}' resolves to step {step:,} — same as a prior candidate)")
            continue
        seen.add(step)
        unique_candidates.append((label, step))

    all_results = {}
    for label, step in unique_candidates:
        ckpt_path = os.path.join(ckpt_dir, f'step_{step:07d}')
        if not os.path.exists(ckpt_path + '.zip'):
            print(f"\n  ⚠ missing checkpoint for step {step}: {ckpt_path}.zip")
            continue
        print(f"\n--- Evaluating: {label}  (step {step:,}) ---")
        model = PPO.load(ckpt_path, device='cpu')
        per_period = {}
        for name, env in [('val', val_env),
                          ('test_bear', test_bear_env),
                          ('test_2023', test_2023_env),
                          ('test_bull', test_bull_env)]:
            hist = evaluate_agent(model, env)
            m = calculate_metrics(hist)
            per_period[name] = m
            print(f"  {name:<10}: Return={m['cumulative_return']:>7.2f}%  "
                  f"Sharpe={m['sharpe_ratio']:>7.4f}  "
                  f"MDD={m['max_drawdown']:>7.2f}%")
        all_results[label] = {'step': step, 'metrics': per_period}

    # ---- Save JSON ----
    out_path = os.path.join(results_dir, 'rolling_mean_eval.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Saved → {out_path}")

    # ---- Comparison vs DJI 30 FULL across all 4 periods ----
    dji_path = os.path.join(project_dir, 'results', 'extended',
                             'final_results.json')
    t2023_path = os.path.join(project_dir, 'results', 'test_2023_summary.json')
    dji = json.load(open(dji_path)) if os.path.exists(dji_path) else None
    dji_2023 = {}
    if os.path.exists(t2023_path):
        dji_2023 = json.load(open(t2023_path)).get('results', {}).get(
            'FULL (baseline)', {})

    print("\n" + "=" * 96)
    print("   SP30 candidates vs DJI 30 FULL (reference)")
    print("=" * 96)
    print(f"  {'Period':<10} {'System':<32} "
          f"{'Return%':>9} {'Sharpe':>8} {'MDD%':>9}")
    print("  " + "-" * 78)
    period_map = {
        'val':       ('agent_val', dji),
        'test_bear': ('agent_test_bear', dji),
        'test_2023': (None, dji_2023),
        'test_bull': ('agent_test_bull', dji),
    }
    for period in ['val', 'test_bear', 'test_2023', 'test_bull']:
        key, src = period_map[period]
        if src:
            dji_m = src.get(key) if key else src
            if dji_m and 'cumulative_return' in dji_m:
                print(f"  {period:<10} {'DJI 30 FULL':<32} "
                      f"{dji_m['cumulative_return']:>9.2f} "
                      f"{dji_m['sharpe_ratio']:>8.4f} "
                      f"{dji_m['max_drawdown']:>9.2f}")
        for label, info in all_results.items():
            m = info['metrics'][period]
            print(f"  {period:<10} {f'SP30: {label}':<32} "
                  f"{m['cumulative_return']:>9.2f} "
                  f"{m['sharpe_ratio']:>8.4f} "
                  f"{m['max_drawdown']:>9.2f}")
        print()

    print("=" * 78)
    print(" If rolling-mean/early-plateau candidates beat the saved spike on")
    print(" test periods → selection artifact, swap the canonical save.")
    print(" If all SP30 candidates lose to DJI 30 → genuine external-validity")
    print(" finding, document as universe-specific calibration limit.")
    print("=" * 78 + "\n")


if __name__ == '__main__':
    main()
