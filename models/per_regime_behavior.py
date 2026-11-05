"""
Capture per-day portfolio composition of the canonical agents on each
test period, alongside the day's HMM regime label.

For each (universe, period, day):
  - cash_weight (post-floor overlay, what the agent actually holds)
  - top_5_stock_weight (sum of the 5 largest stock weights)
  - sum_stocks_weight (= 1 - cash_weight)
  - regime_id, regime_name
  - date

Output:
  results/_canonical/per_regime_behavior.json
"""

import os
import sys
import json

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from portfolio_env import PortfolioEnv
from train_universal_v6_sp100 import V3_COLS_TO_DROP


CANONICAL = {
    'DJI': {
        'ckpt':  'models/saved/checkpoints/universal_v6_dji_mom_2m/step_0800000.zip',
        'floor': 0.7,
        'periods': {
            'val':       'val_data_ext.csv',
            'test_bear': 'test_bear_data_ext.csv',
            'test_2023': 'test_2023_data_ext.csv',
            'test_bull': 'test_bull_data_ext.csv',
        },
    },
    'SP100': {
        'ckpt':  'models/saved/checkpoints/universal_v6_sp100_mom_2m/step_1150000.zip',
        'floor': 0.3,
        'periods': {
            'val':       'val_data_sp100.csv',
            'test_bear': 'test_bear_data_sp100.csv',
            'test_2023': 'test_2023_data_sp100.csv',
            'test_bull': 'test_bull_data_sp100.csv',
        },
    },
    'NDX': {
        'ckpt':  'models/saved/checkpoints/universal_v6_ndx_2017_mom/step_0100000.zip',
        'floor': 0.0,
        'periods': {
            'val':       'val_data_ndx_2017.csv',
            'test_bear': 'test_bear_data_ndx_2017.csv',
            'test_2023': 'test_2023_data_ndx_2017.csv',
            'test_bull': 'test_bull_data_ndx_2017.csv',
        },
    },
}

OUT_PATH = 'results/_canonical/per_regime_behavior.json'


def load_df(name):
    df = pd.read_csv(f'data/{name}')
    df['date'] = pd.to_datetime(df['date'])
    return df.drop(columns=[c for c in V3_COLS_TO_DROP if c in df.columns])


def regime_name(rid):
    return {0: 'bull', 1: 'bear', 2: 'sideways'}.get(int(rid), 'unknown')


def run_canonical(ckpt, df, floor):
    env = PortfolioEnv(
        df=df,
        initial_balance=1_000_000,
        cash_floors={0: 0.0, 1: float(floor), 2: 0.0},
        risk_penalties={0: 0.1, 1: 0.5, 2: 0.3},
        include_regime=True,
        include_regime_probs=True,
        include_vol_features=True,
        use_markowitz_reward=False,
        use_hyperheuristic=False,
    )
    model = PPO.load(ckpt, device='cpu')

    obs, _ = env.reset()
    records = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(action)
        weights = env.portfolio_weights.copy()
        cash_w = float(weights[-1])
        stock_w = weights[:-1]
        top5_sum = float(np.sort(stock_w)[-5:].sum())
        sum_stocks = float(stock_w.sum())
        records.append({
            'date': info['date'],
            'regime_id': info['regime_id'],
            'regime_name': info['regime'],
            'cash_weight': cash_w,
            'top5_weight': top5_sum,
            'sum_stocks_weight': sum_stocks,
        })
        done = term or trunc
    return records


def main():
    out = {}
    for u, cfg in CANONICAL.items():
        out[u] = {}
        for p, csv in cfg['periods'].items():
            print(f"  {u:5}  {p:10}  running...", flush=True)
            df = load_df(csv)
            recs = run_canonical(cfg['ckpt'], df, cfg['floor'])
            out[u][p] = recs
            n_bull = sum(1 for r in recs if r['regime_name'] == 'bull')
            n_bear = sum(1 for r in recs if r['regime_name'] == 'bear')
            n_side = sum(1 for r in recs if r['regime_name'] == 'sideways')
            mean_cash = float(np.mean([r['cash_weight'] for r in recs]))
            print(f"    days={len(recs)} (bull={n_bull}, bear={n_bear}, sideways={n_side}) "
                  f"mean_cash={mean_cash*100:.1f}%")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_PATH}")


if __name__ == '__main__':
    main()
