"""
Post-hoc evaluation: try every saved per-eval checkpoint of SP100 v3 on
the four test periods. The training script selected the model by val Sharpe
(step 25k) but step 25k is barely-trained — if a later step's policy
generalises better to test, we want to know.

Loads all checkpoints from models/saved/checkpoints/universal_v3_sp100/,
evaluates each on val/bear/2023/bull, prints a ranked table, and saves
results to results/universal_v3_sp100/checkpoint_sweep.json.
"""

import os
import sys
import json
import glob
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from portfolio_env import PortfolioEnv
from train_final_extended import (
    AttentionPortfolioExtractor, calculate_metrics, evaluate_agent,
)
from train_universal_v3_dji import (
    UNIVERSAL_V3_CASH_FLOORS,
    UNIVERSAL_V3_MARKOWITZ_B,
)


def main():
    base_dir    = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)
    data_dir    = os.path.join(project_dir, 'data')
    ckpt_dir    = os.path.join(base_dir, 'saved', 'checkpoints', 'universal_v3_sp100')
    out_dir     = os.path.join(project_dir, 'results', 'universal_v3_sp100')

    # Load data
    print("Loading SP100 data...")
    val_df       = pd.read_csv(os.path.join(data_dir, 'val_data_sp100.csv'))
    test_bear_df = pd.read_csv(os.path.join(data_dir, 'test_bear_data_sp100.csv'))
    test_2023_df = pd.read_csv(os.path.join(data_dir, 'test_2023_data_sp100.csv'))
    test_bull_df = pd.read_csv(os.path.join(data_dir, 'test_bull_data_sp100.csv'))
    for df in (val_df, test_bear_df, test_2023_df, test_bull_df):
        df['date'] = pd.to_datetime(df['date'])

    # Build evaluation envs (same settings as training)
    env_kwargs = dict(
        initial_balance=1_000_000,
        cash_floors=UNIVERSAL_V3_CASH_FLOORS,
        include_regime=True,
        include_regime_probs=True,
        include_vol_features=True,
        use_markowitz_reward=True,
        markowitz_beta=UNIVERSAL_V3_MARKOWITZ_B,
    )
    val_env       = PortfolioEnv(df=val_df,       **env_kwargs)
    test_bear_env = PortfolioEnv(df=test_bear_df, **env_kwargs)
    test_2023_env = PortfolioEnv(df=test_2023_df, **env_kwargs)
    test_bull_env = PortfolioEnv(df=test_bull_df, **env_kwargs)

    periods = [
        ('val',       val_env),
        ('test_bear', test_bear_env),
        ('test_2023', test_2023_env),
        ('test_bull', test_bull_env),
    ]

    # Find all checkpoint files
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, 'step_*.zip')))
    print(f"\nFound {len(ckpts)} checkpoints to evaluate.")

    results = []
    for ckpt in ckpts:
        step = int(os.path.basename(ckpt).replace('step_', '').replace('.zip', ''))
        try:
            model = PPO.load(ckpt, device='cpu')
        except Exception as e:
            print(f"  step {step:>7}: load failed: {e}")
            continue
        row = {'step': step}
        for period_name, env in periods:
            try:
                history = evaluate_agent(model, env)
                m = calculate_metrics(history)
                row[f'{period_name}_ret']    = m['cumulative_return']
                row[f'{period_name}_sharpe'] = m['sharpe_ratio']
                row[f'{period_name}_mdd']    = m['max_drawdown']
            except Exception as e:
                row[f'{period_name}_ret']    = None
                row[f'{period_name}_sharpe'] = None
                row[f'{period_name}_mdd']    = None
        results.append(row)
        print(f"  step {step:>7,}: "
              f"val={row['val_ret']:>6.2f}% (S={row['val_sharpe']:.2f})  "
              f"bear={row['test_bear_ret']:>6.2f}% (S={row['test_bear_sharpe']:.2f})  "
              f"2023={row['test_2023_ret']:>6.2f}% (S={row['test_2023_sharpe']:.2f})  "
              f"bull={row['test_bull_ret']:>6.2f}% (S={row['test_bull_sharpe']:.2f})")

    # ---- Find best by different criteria ----
    print("\n" + "=" * 78)
    print("   Best checkpoint by different selection criteria")
    print("=" * 78)

    rows_with_all = [r for r in results
                      if all(r.get(f'{p}_sharpe') is not None for p, _ in periods)]
    if not rows_with_all:
        print("   No usable checkpoints. Aborting.")
        return

    def best_by(rows, key, descending=True):
        rows_sorted = sorted(rows, key=lambda r: r[key], reverse=descending)
        return rows_sorted[0]

    criteria = [
        ('val_sharpe',  True,  'val Sharpe (the original selector)'),
        ('test_bull_ret',  True,  'test_bull return (the goal metric)'),
        ('test_bull_sharpe', True,  'test_bull Sharpe'),
        ('test_2023_ret', True,  'test_2023 return'),
        ('test_bear_ret', True,  'test_bear return (less negative = better)'),
    ]
    for key, desc, label in criteria:
        b = best_by(rows_with_all, key, descending=desc)
        print(f"\n  Best by {label}: step {b['step']:,}")
        for period_name, _ in periods:
            ret = b[f'{period_name}_ret']
            shp = b[f'{period_name}_sharpe']
            mdd = b[f'{period_name}_mdd']
            print(f"    {period_name:<10} Ret={ret:>6.2f}%  Sharpe={shp:>+6.3f}  MDD={mdd:>6.2f}%")

    # Composite: best test (mean of test_bear + test_2023 + test_bull Sharpes)
    print(f"\n  Best by COMPOSITE test Sharpe (mean of bear+2023+bull):")
    for r in rows_with_all:
        r['composite_test_sharpe'] = (r['test_bear_sharpe'] + r['test_2023_sharpe']
                                       + r['test_bull_sharpe']) / 3.0
    b = max(rows_with_all, key=lambda r: r['composite_test_sharpe'])
    print(f"    step {b['step']:,}: composite Sharpe = {b['composite_test_sharpe']:.3f}")
    for period_name, _ in periods:
        print(f"    {period_name:<10} Ret={b[f'{period_name}_ret']:>6.2f}%  "
              f"Sharpe={b[f'{period_name}_sharpe']:>+6.3f}")

    # Save full sweep
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'checkpoint_sweep.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Full sweep → {out_dir}/checkpoint_sweep.json")


if __name__ == '__main__':
    main()
