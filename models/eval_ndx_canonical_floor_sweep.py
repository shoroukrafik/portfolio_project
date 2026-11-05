"""
NDX canonical floor sweep at the DEFENSIVE checkpoint (step 100k).

Background: the existing late_ckpt_x_cashfloor_search.json only covers
steps 375k-500k. We need a floor sweep at step 100k (the DEFENSIVE pick
under the 20% rule applied to a 500k training budget).

For each floor f in {0.0, 0.3, 0.5, 0.7}, evaluates the step-100k
checkpoint on val + 3 test periods with bear cash-floor = f, all other
floors = 0. Saves results matching the structure of
cashfloor_overlay_DEFENSIVE.json used by DJI and SP100.

Output: results/universal_v6_ndx_2017_mom/cashfloor_overlay_DEFENSIVE.json
"""

import os
import sys
import json

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from portfolio_env import PortfolioEnv
from train_universal_v6_sp100 import V3_COLS_TO_DROP


CHECKPOINT = 'models/saved/checkpoints/universal_v6_ndx_2017_mom/step_0100000.zip'
OUT_PATH   = 'results/universal_v6_ndx_2017_mom/cashfloor_overlay_DEFENSIVE.json'

PERIOD_CSVS = {
    'val':       'val_data_ndx_2017.csv',
    'test_bear': 'test_bear_data_ndx_2017.csv',
    'test_2023': 'test_2023_data_ndx_2017.csv',
    'test_bull': 'test_bull_data_ndx_2017.csv',
}

BENCH = {
    'val':       {'NDX BH': 26.63, 'EW NDX': 31.31},
    'test_bear': {'NDX BH': -33.71, 'EW NDX': -17.44},
    'test_2023': {'NDX BH': 54.90, 'EW NDX': 40.27},
    'test_bull': {'NDX BH': 28.13, 'EW NDX': 13.67},
}

FLOORS = [0.0, 0.3, 0.5, 0.7]


def load_df(name):
    df = pd.read_csv(f'data/{name}')
    df['date'] = pd.to_datetime(df['date'])
    return df.drop(columns=[c for c in V3_COLS_TO_DROP if c in df.columns])


def evaluate_with_floor(model, df, floor):
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
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(action)
        done = term or trunc
    h = env.get_portfolio_history()
    v = h['value'].values.astype(float)
    n = len(v)
    cum_ret = float((v[-1] / v[0] - 1) * 100)
    ann_ret = float(((v[-1] / v[0]) ** (252 / n) - 1) * 100)
    dr = np.diff(v) / v[:-1]
    ann_vol = float(np.std(dr) * np.sqrt(252) * 100)
    sharpe = float(ann_ret / ann_vol) if ann_vol > 0 else 0.0
    peak = np.maximum.accumulate(v)
    mdd = float(np.min((v - peak) / peak) * 100)
    return {
        'cumulative_return':    cum_ret,
        'annualized_return':    ann_ret,
        'annualized_volatility': ann_vol,
        'sharpe_ratio':         sharpe,
        'max_drawdown':         mdd,
        'final_value':          float(v[-1]),
    }


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
    print(f"Loading {CHECKPOINT}")
    model = PPO.load(CHECKPOINT, device='cpu')

    period_dfs = {p: load_df(csv) for p, csv in PERIOD_CSVS.items()}

    out = {
        'checkpoint': CHECKPOINT,
        'floor_only_applies_to': 'bear (regime_id=1)',
        'by_floor': {},
    }

    print(f"\n{'='*90}")
    print(f"   NDX-2017 step 100k floor sweep")
    print(f"{'='*90}")
    print(f"  {'floor':>6} | {'val':>7} {'bear':>7} {'2023':>7} {'bull':>7} | {'v.shp':>5} | wins")
    print(f"  {'-'*6} + {'-'*7} {'-'*7} {'-'*7} {'-'*7} + {'-'*5} + ----")

    for floor in FLOORS:
        per = {}
        for p, df in period_dfs.items():
            per[p] = evaluate_with_floor(model, df, floor)
        wins, total = count_wins(per, BENCH)
        v = per['val']['cumulative_return']
        b = per['test_bear']['cumulative_return']
        t = per['test_2023']['cumulative_return']
        u = per['test_bull']['cumulative_return']
        shp = per['val']['sharpe_ratio']
        print(f"  {floor:>6.2f} | {v:>+7.2f} {b:>+7.2f} {t:>+7.2f} {u:>+7.2f} | {shp:>5.3f} | {wins:>2}/{total:>2}")
        out['by_floor'][f"{floor}"] = per

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_PATH}")


if __name__ == '__main__':
    main()
