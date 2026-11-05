# ============================================
# Quick re-eval: extended baseline with cash floor DISABLED
# ============================================
# Loads ppo_best_extended.zip (already trained) and evaluates on
# val/bear/bull using PortfolioEnv with ablation_mode='no_floor'.
#
# Goal: see whether the agent's *learned* policy retains bear defense
# without the hard cash-floor constraint, AND whether bull recovers
# from the HMM-mislabeling problem on 2024.
#
# This does NOT retrain. ~1 min total.
# ============================================

import os
import sys
import json
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from portfolio_env import PortfolioEnv


def calculate_metrics(history, initial_balance=1_000_000):
    values = history['value'].values
    final_value = values[-1]
    cum_ret = (final_value - initial_balance) / initial_balance * 100
    daily_ret = np.diff(values) / values[:-1]
    n = len(values)
    ann_ret = ((final_value/initial_balance)**(252/n) - 1) * 100
    ann_vol = np.std(daily_ret) * np.sqrt(252) * 100
    sharpe = ann_ret/ann_vol if ann_vol > 0 else 0
    peak = np.maximum.accumulate(values)
    dd = (values - peak) / peak
    mdd = np.min(dd) * 100
    return {
        'cumulative_return': round(cum_ret, 2),
        'sharpe_ratio': round(sharpe, 4),
        'max_drawdown': round(mdd, 2),
        'annualized_volatility': round(ann_vol, 2),
    }


def evaluate(model, env):
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(action)
        done = term or trunc
    return env.get_portfolio_history()


def main():
    base_dir    = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)
    data_dir    = os.path.join(project_dir, 'data')
    saved_dir   = os.path.join(base_dir, 'saved')

    print("=" * 60)
    print("   Extended baseline — cash floor DISABLED at eval")
    print("=" * 60)

    model_path = os.path.join(saved_dir, 'ppo_best_extended')
    print(f"\nLoading model: {model_path}.zip")
    model = PPO.load(model_path, device='cpu')

    eval_sets = [
        ('val',       'val_data_ext.csv'),
        ('test_bear', 'test_bear_data_ext.csv'),
        ('test_bull', 'test_bull_data_ext.csv'),
    ]

    results = {}
    for name, fn in eval_sets:
        df = pd.read_csv(os.path.join(data_dir, fn))
        df['date'] = pd.to_datetime(df['date'])
        # CRITICAL: ablation_mode='no_floor' disables cash floor
        env = PortfolioEnv(df=df, initial_balance=1_000_000,
                           ablation_mode='no_floor')
        history = evaluate(model, env)
        m = calculate_metrics(history)
        results[name] = m
        print(f"\n  {name}:")
        print(f"    Return: {m['cumulative_return']:>7.2f}%  "
              f"Sharpe: {m['sharpe_ratio']:>7.4f}  "
              f"MDD: {m['max_drawdown']:>7.2f}%")

    # ---- Compare: with floor (existing) vs no floor (this run) ----
    floor_path = os.path.join(project_dir, 'results', 'extended', 'final_results.json')
    with open(floor_path) as f:
        floor = json.load(f)

    print("\n" + "=" * 76)
    print("   EXTENDED baseline:  cash floor ON  vs  cash floor OFF")
    print("=" * 76)
    print(f"  {'Period':<12} {'Setting':<22} "
          f"{'Return%':>9} {'Sharpe':>8} {'MDD%':>9}")
    print(f"  {'-'*70}")
    for period in ['val', 'test_bear', 'test_bull']:
        on  = floor.get(f'agent_{period}', {})
        off = results[period]
        print(f"  {period:<12} {'WITH floor':<22} "
              f"{on['cumulative_return']:>9.2f} "
              f"{on['sharpe_ratio']:>8.4f} "
              f"{on['max_drawdown']:>9.2f}")
        print(f"  {period:<12} {'NO floor':<22} "
              f"{off['cumulative_return']:>9.2f} "
              f"{off['sharpe_ratio']:>8.4f} "
              f"{off['max_drawdown']:>9.2f}")
        d_ret = off['cumulative_return'] - on['cumulative_return']
        d_shp = off['sharpe_ratio']      - on['sharpe_ratio']
        d_mdd = off['max_drawdown']      - on['max_drawdown']
        sgn = lambda x: f"{'+' if x >= 0 else ''}{x:.2f}"
        print(f"  {'':12} {'  Δ (off - on)':<22} "
              f"{sgn(d_ret):>9} {sgn(d_shp):>8} {sgn(d_mdd):>9}")
        print()

    # Save
    out_path = os.path.join(project_dir, 'results', 'extended',
                            'final_results_no_floor.json')
    with open(out_path, 'w') as f:
        json.dump({
            'method': 'extended_data_baseline_no_floor_at_eval',
            'note': 'Same trained model as ppo_best_extended; '
                    'evaluated with ablation_mode=no_floor',
            'val':       results['val'],
            'test_bear': results['test_bear'],
            'test_bull': results['test_bull'],
        }, f, indent=2)
    print(f"✅ Saved → {out_path}")


if __name__ == "__main__":
    main()
