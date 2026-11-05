"""
Sweep ALL checkpoints for NDX experimental variants (exp1, exp2, exp3, exp4).

Each variant uses the production NDX test data (test_*_data_ndx.csv) but with
its own env_kwargs (defined in train_universal_v6_ndx_exp.variant_config).

Saves per-variant: results/universal_v6_ndx_{variant}/checkpoint_sweep.json
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
from train_universal_v6_ndx_exp import variant_config, PROD_CASH_FLOORS, PROD_RISK_PENALTIES


PERIODS = {
    'val':       'val_data_ndx.csv',
    'test_bear': 'test_bear_data_ndx.csv',
    'test_2023': 'test_2023_data_ndx.csv',
    'test_bull': 'test_bull_data_ndx.csv',
}


def load_df(name):
    df = pd.read_csv(f'data/{name}')
    df['date'] = pd.to_datetime(df['date'])
    return df.drop(columns=[c for c in V3_COLS_TO_DROP if c in df.columns])


def env_kwargs_for(variant):
    """Compose the env_kwargs for one experiment variant."""
    base = dict(
        initial_balance=1_000_000,
        cash_floors=PROD_CASH_FLOORS,
        risk_penalties=PROD_RISK_PENALTIES,
        include_regime=True, include_regime_probs=True, include_vol_features=True,
        use_markowitz_reward=False, use_hyperheuristic=False,
        use_concentration_bonus=False,
    )
    env_delta, _ppo_delta = variant_config(variant)
    return {**base, **env_delta}


def evaluate_one(model_path, env_kwargs):
    model = PPO.load(model_path, device='cpu')
    out = {}
    for p, csv in PERIODS.items():
        df = load_df(csv)
        env = PortfolioEnv(df=df, **env_kwargs)
        history = evaluate_agent(model, env)
        out[p] = calculate_metrics(history)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=['exp1','exp2','exp3','exp4'], required=True)
    args = parser.parse_args()

    ck_dir = f'models/saved/checkpoints/universal_v6_ndx_{args.variant}'
    out_dir = f'results/universal_v6_ndx_{args.variant}'
    os.makedirs(out_dir, exist_ok=True)
    checkpoints = sorted(glob.glob(os.path.join(ck_dir, 'step_*.zip')))
    env_kwargs = env_kwargs_for(args.variant)

    print(f"\n{'='*92}")
    print(f"   Universal v6 NDX {args.variant} — sweeping {len(checkpoints)} checkpoints")
    print(f"{'='*92}")
    print(f"  {'step':>9} | {'val':>7} {'bear':>7} {'2023':>7} {'bull':>7} | {'vSh':>5} {'bSh':>5} {'tSh':>5} {'uSh':>5}")
    print('  ' + '-'*70)
    sweep = []
    for ck in checkpoints:
        step = int(os.path.basename(ck).replace('step_','').replace('.zip',''))
        results = evaluate_one(ck, env_kwargs)
        v = results['val']['cumulative_return']
        b = results['test_bear']['cumulative_return']
        t = results['test_2023']['cumulative_return']
        u = results['test_bull']['cumulative_return']
        vs = results['val']['sharpe_ratio']
        bs = results['test_bear']['sharpe_ratio']
        ts = results['test_2023']['sharpe_ratio']
        us = results['test_bull']['sharpe_ratio']
        print(f"  {step:>9,} | {v:>+7.2f} {b:>+7.2f} {t:>+7.2f} {u:>+7.2f} | "
              f"{vs:>+5.2f} {bs:>+5.2f} {ts:>+5.2f} {us:>+5.2f}")
        sweep.append({
            'step': step, 'val_ret': v, 'test_bear_ret': b,
            'test_2023_ret': t, 'test_bull_ret': u,
            'val_sharpe': vs, 'test_bear_sharpe': bs,
            'test_2023_sharpe': ts, 'test_bull_sharpe': us,
            'val_mdd': results['val']['max_drawdown'],
            'test_bear_mdd': results['test_bear']['max_drawdown'],
            'test_2023_mdd': results['test_2023']['max_drawdown'],
            'test_bull_mdd': results['test_bull']['max_drawdown'],
        })

    with open(os.path.join(out_dir, 'checkpoint_sweep.json'), 'w') as f:
        json.dump(sweep, f, indent=2)
    print(f"\n  ✅ → {out_dir}/checkpoint_sweep.json")


if __name__ == '__main__':
    main()
