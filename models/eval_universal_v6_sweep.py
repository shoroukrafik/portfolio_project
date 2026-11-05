"""
Sweep all checkpoints of a universal v6 single-agent run.

Mirrors eval_mac_sweep.py but for the single-agent v6 universal trainers
(DJI / SP100 / NDX). Evaluates every step_*.zip on val / test_bear /
test_2023 / test_bull and reports per-checkpoint metrics so we can find
the best-by-{val_sharpe, bull_return, bear_return, total_wins}.

Run:
  python3 models/eval_universal_v6_sweep.py --universe ndx
  python3 models/eval_universal_v6_sweep.py --universe dji
  python3 models/eval_universal_v6_sweep.py --universe sp100
"""

import os
import sys
import json
import glob
import argparse
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from portfolio_env import PortfolioEnv
from train_final_extended import calculate_metrics, evaluate_agent
from train_universal_v6_sp100 import V3_COLS_TO_DROP


PERIOD_CSVS = {
    'dji':   {
        'val':       'val_data_ext.csv',
        'test_bear': 'test_bear_data_ext.csv',
        'test_2023': 'test_2023_data_ext.csv',
        'test_bull': 'test_bull_data_ext.csv',
    },
    'sp100': {
        'val':       'val_data_sp100.csv',
        'test_bear': 'test_bear_data_sp100.csv',
        'test_2023': 'test_2023_data_sp100.csv',
        'test_bull': 'test_bull_data_sp100.csv',
    },
    'dji_mom': {
        'val':       'val_data_ext.csv',
        'test_bear': 'test_bear_data_ext.csv',
        'test_2023': 'test_2023_data_ext.csv',
        'test_bull': 'test_bull_data_ext.csv',
    },
    'sp100_mom': {
        'val':       'val_data_sp100.csv',
        'test_bear': 'test_bear_data_sp100.csv',
        'test_2023': 'test_2023_data_sp100.csv',
        'test_bull': 'test_bull_data_sp100.csv',
    },
    'sp100_mom_2m': {
        'val':       'val_data_sp100.csv',
        'test_bear': 'test_bear_data_sp100.csv',
        'test_2023': 'test_2023_data_sp100.csv',
        'test_bull': 'test_bull_data_sp100.csv',
    },
    'dji_mom_2m': {
        'val':       'val_data_ext.csv',
        'test_bear': 'test_bear_data_ext.csv',
        'test_2023': 'test_2023_data_ext.csv',
        'test_bull': 'test_bull_data_ext.csv',
    },
    'ndx_mom_2m': {
        'val':       'val_data_ndx.csv',
        'test_bear': 'test_bear_data_ndx.csv',
        'test_2023': 'test_2023_data_ndx.csv',
        'test_bull': 'test_bull_data_ndx.csv',
    },
    'ndx_2017_mom_bullconc': {
        'val':       'val_data_ndx_2017.csv',
        'test_bear': 'test_bear_data_ndx_2017.csv',
        'test_2023': 'test_2023_data_ndx_2017.csv',
        'test_bull': 'test_bull_data_ndx_2017.csv',
    },
    'sp100_2015_mom': {
        'val':       'val_data_sp100_2015.csv',
        'test_bear': 'test_bear_data_sp100_2015.csv',
        'test_2023': 'test_2023_data_sp100_2015.csv',
        'test_bull': 'test_bull_data_sp100_2015.csv',
    },
    'sp100_2015': {
        'val':       'val_data_sp100_2015.csv',
        'test_bear': 'test_bear_data_sp100_2015.csv',
        'test_2023': 'test_2023_data_sp100_2015.csv',
        'test_bull': 'test_bull_data_sp100_2015.csv',
    },
    'dji_2015': {
        'val':       'val_data_ext_2015.csv',
        'test_bear': 'test_bear_data_ext_2015.csv',
        'test_2023': 'test_2023_data_ext_2015.csv',
        'test_bull': 'test_bull_data_ext_2015.csv',
    },
    'dji_2015_mom': {
        'val':       'val_data_ext_2015.csv',
        'test_bear': 'test_bear_data_ext_2015.csv',
        'test_2023': 'test_2023_data_ext_2015.csv',
        'test_bull': 'test_bull_data_ext_2015.csv',
    },
    'dji_h2soft': {
        'val':       'val_data_ext.csv',
        'test_bear': 'test_bear_data_ext.csv',
        'test_2023': 'test_2023_data_ext.csv',
        'test_bull': 'test_bull_data_ext.csv',
    },
    'sp100_h2soft': {
        'val':       'val_data_sp100.csv',
        'test_bear': 'test_bear_data_sp100.csv',
        'test_2023': 'test_2023_data_sp100.csv',
        'test_bull': 'test_bull_data_sp100.csv',
    },
    'dji_lr3e4': {
        'val':       'val_data_ext.csv',
        'test_bear': 'test_bear_data_ext.csv',
        'test_2023': 'test_2023_data_ext.csv',
        'test_bull': 'test_bull_data_ext.csv',
    },
    'dji_lrsched': {
        'val':       'val_data_ext.csv',
        'test_bear': 'test_bear_data_ext.csv',
        'test_2023': 'test_2023_data_ext.csv',
        'test_bull': 'test_bull_data_ext.csv',
    },
    'dji_sb3default': {
        'val':       'val_data_ext.csv',
        'test_bear': 'test_bear_data_ext.csv',
        'test_2023': 'test_2023_data_ext.csv',
        'test_bull': 'test_bull_data_ext.csv',
    },
    'ndx': {
        'val':       'val_data_ndx.csv',
        'test_bear': 'test_bear_data_ndx.csv',
        'test_2023': 'test_2023_data_ndx.csv',
        'test_bull': 'test_bull_data_ndx.csv',
    },
    'ndx_2017_mom_bear': {
        'val':       'val_data_ndx_2017.csv',
        'test_bear': 'test_bear_data_ndx_2017.csv',
        'test_2023': 'test_2023_data_ndx_2017.csv',
        'test_bull': 'test_bull_data_ndx_2017.csv',
    },
    'ndx_2017_mom': {
        'val':       'val_data_ndx_2017.csv',
        'test_bear': 'test_bear_data_ndx_2017.csv',
        'test_2023': 'test_2023_data_ndx_2017.csv',
        'test_bull': 'test_bull_data_ndx_2017.csv',
    },
    'ndx_2017': {
        'val':       'val_data_ndx_2017.csv',
        'test_bear': 'test_bear_data_ndx_2017.csv',
        'test_2023': 'test_2023_data_ndx_2017.csv',
        'test_bull': 'test_bull_data_ndx_2017.csv',
    },
}

BENCH = {
    'dji': {
        'val':       {'DJI BH': 20.23, 'EW DJI': 25.09},
        'test_bear': {'DJI BH': -9.40, 'EW DJI': -10.25},
        'test_2023': {'DJI BH': 13.74, 'EW DJI': 18.05},
        'test_bull': {'DJI BH': 12.88, 'EW DJI': 22.04},
    },
    'sp100': {
        'val':       {'SP500 BH': 28.79, 'OEX BH': 29.33, 'EW SP100': 33.11},
        'test_bear': {'SP500 BH': -19.95, 'OEX BH': -22.99, 'EW SP100': -9.48},
        'test_2023': {'SP500 BH': 24.73, 'OEX BH': 31.39, 'EW SP100': 21.17},
        'test_bull': {'SP500 BH': 24.54, 'OEX BH': 30.90, 'EW SP100': 18.16},
    },
    'dji_mom': {
        'val':       {'DJI BH': 20.23, 'EW DJI': 25.09},
        'test_bear': {'DJI BH': -9.40, 'EW DJI': -10.25},
        'test_2023': {'DJI BH': 13.74, 'EW DJI': 18.05},
        'test_bull': {'DJI BH': 12.88, 'EW DJI': 22.04},
    },
    'sp100_mom': {
        'val':       {'SP500 BH': 28.79, 'OEX BH': 29.33, 'EW SP100': 33.11},
        'test_bear': {'SP500 BH': -19.95, 'OEX BH': -22.99, 'EW SP100': -9.48},
        'test_2023': {'SP500 BH': 24.73, 'OEX BH': 31.39, 'EW SP100': 21.17},
        'test_bull': {'SP500 BH': 24.54, 'OEX BH': 30.90, 'EW SP100': 18.16},
    },
    'sp100_mom_2m': {
        'val':       {'SP500 BH': 28.79, 'OEX BH': 29.33, 'EW SP100': 33.11},
        'test_bear': {'SP500 BH': -19.95, 'OEX BH': -22.99, 'EW SP100': -9.48},
        'test_2023': {'SP500 BH': 24.73, 'OEX BH': 31.39, 'EW SP100': 21.17},
        'test_bull': {'SP500 BH': 24.54, 'OEX BH': 30.90, 'EW SP100': 18.16},
    },
    'dji_mom_2m': {
        'val':       {'DJI BH': 20.23, 'EW DJI': 25.09},
        'test_bear': {'DJI BH': -9.40, 'EW DJI': -10.25},
        'test_2023': {'DJI BH': 13.74, 'EW DJI': 18.05},
        'test_bull': {'DJI BH': 12.88, 'EW DJI': 22.04},
    },
    'ndx_mom_2m': {
        'val':       {'NDX BH': 26.63, 'EW NDX': 31.00},
        'test_bear': {'NDX BH': -33.71, 'EW NDX': -15.86},
        'test_2023': {'NDX BH': 54.90, 'EW NDX': 37.00},
        'test_bull': {'NDX BH': 28.13, 'EW NDX': 10.38},
    },
    'ndx_2017_mom_bullconc': {                              # 90-stock universe EW (same as ndx_2017_mom)
        'val':       {'NDX BH': 26.63, 'EW NDX': 31.31},
        'test_bear': {'NDX BH': -33.71, 'EW NDX': -17.44},
        'test_2023': {'NDX BH': 54.90, 'EW NDX': 40.27},
        'test_bull': {'NDX BH': 28.13, 'EW NDX': 13.67},
    },
    'sp100_2015_mom': {
        'val':       {'SP500 BH': 28.79, 'OEX BH': 29.33, 'EW SP100': 33.11},
        'test_bear': {'SP500 BH': -19.95, 'OEX BH': -22.99, 'EW SP100': -9.48},
        'test_2023': {'SP500 BH': 24.73, 'OEX BH': 31.39, 'EW SP100': 21.17},
        'test_bull': {'SP500 BH': 24.54, 'OEX BH': 30.90, 'EW SP100': 18.16},
    },
    'sp100_2015': {
        'val':       {'SP500 BH': 28.79, 'OEX BH': 29.33, 'EW SP100': 33.11},
        'test_bear': {'SP500 BH': -19.95, 'OEX BH': -22.99, 'EW SP100': -9.48},
        'test_2023': {'SP500 BH': 24.73, 'OEX BH': 31.39, 'EW SP100': 21.17},
        'test_bull': {'SP500 BH': 24.54, 'OEX BH': 30.90, 'EW SP100': 18.16},
    },
    'dji_2015': {
        'val':       {'DJI BH': 20.23, 'EW DJI': 25.09},
        'test_bear': {'DJI BH': -9.40, 'EW DJI': -10.25},
        'test_2023': {'DJI BH': 13.74, 'EW DJI': 18.05},
        'test_bull': {'DJI BH': 12.88, 'EW DJI': 22.04},
    },
    'dji_2015_mom': {
        'val':       {'DJI BH': 20.23, 'EW DJI': 25.09},
        'test_bear': {'DJI BH': -9.40, 'EW DJI': -10.25},
        'test_2023': {'DJI BH': 13.74, 'EW DJI': 18.05},
        'test_bull': {'DJI BH': 12.88, 'EW DJI': 22.04},
    },
    'dji_h2soft': {
        'val':       {'DJI BH': 20.23, 'EW DJI': 25.09},
        'test_bear': {'DJI BH': -9.40, 'EW DJI': -10.25},
        'test_2023': {'DJI BH': 13.74, 'EW DJI': 18.05},
        'test_bull': {'DJI BH': 12.88, 'EW DJI': 22.04},
    },
    'sp100_h2soft': {
        'val':       {'SP500 BH': 28.79, 'OEX BH': 29.33, 'EW SP100': 33.11},
        'test_bear': {'SP500 BH': -19.95, 'OEX BH': -22.99, 'EW SP100': -9.48},
        'test_2023': {'SP500 BH': 24.73, 'OEX BH': 31.39, 'EW SP100': 21.17},
        'test_bull': {'SP500 BH': 24.54, 'OEX BH': 30.90, 'EW SP100': 18.16},
    },
    'dji_lr3e4': {
        'val':       {'DJI BH': 20.23, 'EW DJI': 25.09},
        'test_bear': {'DJI BH': -9.40, 'EW DJI': -10.25},
        'test_2023': {'DJI BH': 13.74, 'EW DJI': 18.05},
        'test_bull': {'DJI BH': 12.88, 'EW DJI': 22.04},
    },
    'dji_lrsched': {
        'val':       {'DJI BH': 20.23, 'EW DJI': 25.09},
        'test_bear': {'DJI BH': -9.40, 'EW DJI': -10.25},
        'test_2023': {'DJI BH': 13.74, 'EW DJI': 18.05},
        'test_bull': {'DJI BH': 12.88, 'EW DJI': 22.04},
    },
    'dji_sb3default': {
        'val':       {'DJI BH': 20.23, 'EW DJI': 25.09},
        'test_bear': {'DJI BH': -9.40, 'EW DJI': -10.25},
        'test_2023': {'DJI BH': 13.74, 'EW DJI': 18.05},
        'test_bull': {'DJI BH': 12.88, 'EW DJI': 22.04},
    },
    'ndx': {
        'val':       {'NDX BH': 26.63, 'EW NDX': 31.00},   # 68-stock universe EW
        'test_bear': {'NDX BH': -33.71, 'EW NDX': -15.86},
        'test_2023': {'NDX BH': 54.90, 'EW NDX': 37.00},
        'test_bull': {'NDX BH': 28.13, 'EW NDX': 10.38},
    },
    'ndx_2017_mom_bear': {                                  # 90-stock universe EW
        'val':       {'NDX BH': 26.63, 'EW NDX': 31.31},
        'test_bear': {'NDX BH': -33.71, 'EW NDX': -17.44},
        'test_2023': {'NDX BH': 54.90, 'EW NDX': 40.27},
        'test_bull': {'NDX BH': 28.13, 'EW NDX': 13.67},
    },
    'ndx_2017_mom': {                                       # 90-stock universe EW
        'val':       {'NDX BH': 26.63, 'EW NDX': 31.31},
        'test_bear': {'NDX BH': -33.71, 'EW NDX': -17.44},
        'test_2023': {'NDX BH': 54.90, 'EW NDX': 40.27},
        'test_bull': {'NDX BH': 28.13, 'EW NDX': 13.67},
    },
    'ndx_2017': {                                           # 90-stock universe EW
        'val':       {'NDX BH': 26.63, 'EW NDX': 31.31},
        'test_bear': {'NDX BH': -33.71, 'EW NDX': -17.44},
        'test_2023': {'NDX BH': 54.90, 'EW NDX': 40.27},
        'test_bull': {'NDX BH': 28.13, 'EW NDX': 13.67},
    },
}


def load_df(period_csv, drop_long_mom=False):
    df = pd.read_csv(f'data/{period_csv}')
    df['date'] = pd.to_datetime(df['date'])
    drop_cols = [c for c in V3_COLS_TO_DROP if c in df.columns]
    if drop_long_mom:
        # For "no long-mom" ablation variants — drop the long-momentum
        # columns so PortfolioEnv builds a 20-feature-per-stock obs that
        # matches the trained model.
        drop_cols += [c for c in ('mom_120d_norm', 'mom_252d_norm') if c in df.columns]
    return df.drop(columns=drop_cols)


def env_kwargs():
    return dict(
        initial_balance=1_000_000,
        cash_floors={0: 0.0, 1: 0.0, 2: 0.0},
        risk_penalties={0: 0.1, 1: 0.5, 2: 0.3},
        include_regime=True, include_regime_probs=True, include_vol_features=True,
        use_markowitz_reward=False, use_hyperheuristic=False,
    )


def evaluate_one(model_path, periods, drop_long_mom=False):
    """Run one checkpoint over all periods, return per-period metrics."""
    model = PPO.load(model_path, device='cpu')
    out = {}
    for p, csv in periods.items():
        df = load_df(csv, drop_long_mom=drop_long_mom)
        env = PortfolioEnv(df=df, **env_kwargs())
        history = evaluate_agent(model, env)
        out[p] = calculate_metrics(history)
    return out


def count_wins(results, bench):
    wins, total = 0, 0
    for p, bs in bench.items():
        ret = results[p]['cumulative_return']
        for _, val in bs.items():
            total += 1
            if ret > val:
                wins += 1
    return wins, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--universe', choices=['dji','sp100','ndx','ndx_2017','ndx_2017_mom','ndx_2017_mom_bear','ndx_2017_mom_bullconc','dji_mom','dji_mom_2m','sp100_mom','sp100_mom_2m','ndx_mom_2m','sp100_2015_mom','sp100_2015','dji_2015','dji_2015_mom','dji_h2soft','sp100_h2soft','dji_lr3e4','dji_lrsched','dji_sb3default'], required=True)
    parser.add_argument('--no_regime', action='store_true',
                        help='Sweep the H2 no-regime variant (suffix _noregime)')
    args = parser.parse_args()

    suffix = '_noregime' if args.no_regime else ''
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ck_dir = os.path.join(base, 'models', 'saved', 'checkpoints',
                           f'universal_v6_{args.universe}{suffix}')
    out_dir = os.path.join(base, 'results', f'universal_v6_{args.universe}{suffix}')
    os.makedirs(out_dir, exist_ok=True)

    periods = PERIOD_CSVS[args.universe]
    bench = BENCH[args.universe]

    checkpoints = sorted(glob.glob(os.path.join(ck_dir, 'step_*.zip')))
    print(f"\n{'='*100}")
    print(f"   Universal v6 {args.universe.upper()}{suffix} — sweeping {len(checkpoints)} checkpoints")
    print(f"{'='*100}\n")
    print(f"  {'step':>8} | {'val':>7} {'bear':>7} {'2023':>7} {'bull':>7} | {'v.shp':>5} | wins")
    print(f"  {'-'*8} + {'-'*7} {'-'*7} {'-'*7} {'-'*7} + {'-'*5} + ----")

    # Variants whose model was trained WITHOUT long-momentum features.
    # The sweep must drop mom_120d / mom_252d when loading these so obs shape matches.
    # NOTE: dji_h2soft / sp100_h2soft / dji_lr3e4 / dji_lrsched are NOT here — those
    # trainers don't drop long-mom from the dataframe, so the env auto-includes them.
    # Models expect 22 features.
    NO_LONG_MOM_VARIANTS = {'dji', 'sp100', 'ndx', 'ndx_2017', 'sp100_2015', 'dji_2015'}
    drop_long_mom = args.universe in NO_LONG_MOM_VARIANTS

    sweep = []
    for ck in checkpoints:
        step = int(os.path.basename(ck).replace('step_','').replace('.zip',''))
        print(f"  {step:>8,}", end='', flush=True)
        try:
            results = evaluate_one(ck, periods, drop_long_mom=drop_long_mom)
        except Exception as e:
            print(f"  ❌ failed: {e}")
            continue
        wins, total = count_wins(results, bench)
        v = results['val']['cumulative_return']
        b = results['test_bear']['cumulative_return']
        t = results['test_2023']['cumulative_return']
        u = results['test_bull']['cumulative_return']
        print(f" | {v:>+7.2f} {b:>+7.2f} {t:>+7.2f} {u:>+7.2f} | {results['val']['sharpe_ratio']:>5.3f} | {wins:>2}/{total:>2}")
        sweep.append({
            'step': step,
            'val_ret': v, 'test_bear_ret': b,
            'test_2023_ret': t, 'test_bull_ret': u,
            'val_sharpe': results['val']['sharpe_ratio'],
            'test_bear_sharpe': results['test_bear']['sharpe_ratio'],
            'test_2023_sharpe': results['test_2023']['sharpe_ratio'],
            'test_bull_sharpe': results['test_bull']['sharpe_ratio'],
            'val_mdd': results['val']['max_drawdown'],
            'test_bear_mdd': results['test_bear']['max_drawdown'],
            'test_2023_mdd': results['test_2023']['max_drawdown'],
            'test_bull_mdd': results['test_bull']['max_drawdown'],
            'wins': wins, 'total': total,
        })

    print(f"\n{'='*100}")
    print(f"   BEST checkpoints by criterion:")
    print(f"{'='*100}")
    best_val   = max(sweep, key=lambda r: r['val_sharpe'])
    best_wins  = max(sweep, key=lambda r: r['wins'])
    best_bull  = max(sweep, key=lambda r: r['test_bull_ret'])
    best_bear  = max(sweep, key=lambda r: r['test_bear_ret'])
    best_2023  = max(sweep, key=lambda r: r['test_2023_ret'])
    for label, r in [('val Sharpe', best_val), ('total wins', best_wins),
                      ('bull return', best_bull), ('bear return', best_bear),
                      ('2023 return', best_2023)]:
        print(f"  by {label:<13} step {r['step']:>9,}  val={r['val_ret']:+6.2f} bear={r['test_bear_ret']:+6.2f} "
              f"2023={r['test_2023_ret']:+6.2f} bull={r['test_bull_ret']:+6.2f}  vSh={r['val_sharpe']:.3f}  wins={r['wins']}/{r['total']}")

    out_path = os.path.join(out_dir, 'checkpoint_sweep.json')
    with open(out_path, 'w') as f:
        json.dump(sweep, f, indent=2)
    print(f"\n✅ → {out_path}")


if __name__ == '__main__':
    main()
