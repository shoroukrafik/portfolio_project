"""
Floor sweep on the existing SP100 no_regime model.

Tests the hypothesis: does the cash floor compensate for the bear
damage of removing the regime mechanism on S&P 100?

The no_regime model was trained with regime obs OFF, regime probs OFF,
vol features OFF, and uniform lambda. The cash floor uses the regime
label from the data file directly (not from the agent's observation),
so applying a floor at eval is mechanically valid.

For each floor f in {0.0, 0.3, 0.5, 0.7}, evaluates on val + 3 test
periods. Compares against the canonical S&P 100 reference numbers.

Output: results/_canonical/sp100_noregime_floor_sweep.json
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


# Find the noregime model checkpoint
# The training script saves to ppo_universal_v6_sp100_noregime.zip
CKPT_OPTIONS = [
    'models/saved/ppo_universal_v6_sp100_noregime.zip',
    'models/saved/ppo_universal_v6_sp100_noregime_final.zip',
]
CKPT = None
for c in CKPT_OPTIONS:
    if os.path.exists(c):
        CKPT = c
        break
if CKPT is None:
    raise FileNotFoundError(f"No noregime checkpoint found, tried: {CKPT_OPTIONS}")

PERIODS = {
    'val':       'val_data_sp100.csv',
    'test_bear': 'test_bear_data_sp100.csv',
    'test_2023': 'test_2023_data_sp100.csv',
    'test_bull': 'test_bull_data_sp100.csv',
}

FLOORS = [0.0, 0.3, 0.5, 0.7]
OUT_PATH = 'results/_canonical/sp100_noregime_floor_sweep.json'


def load_df(csv):
    df = pd.read_csv(f'data/{csv}')
    df['date'] = pd.to_datetime(df['date'])
    # Drop V3 lag/trend columns AND long-momentum columns, because the
    # noregime model was trained on the 20-feature observation set.
    drop_cols = [c for c in V3_COLS_TO_DROP if c in df.columns]
    drop_cols += [c for c in ('mom_120d_norm', 'mom_252d_norm') if c in df.columns]
    return df.drop(columns=drop_cols)


def eval_with_floor(model, df, floor):
    """Evaluate noregime model with the given cash floor.
    Critical: env_kwargs MUST match the model's training observation
    space (include_regime=False, etc.) or the obs shape won't fit."""
    env = PortfolioEnv(
        df=df,
        initial_balance=1_000_000,
        cash_floors={0: 0.0, 1: float(floor), 2: 0.0},
        risk_penalties={0: 0.3, 1: 0.3, 2: 0.3},  # uniform — matches noregime training
        include_regime=False,
        include_regime_probs=False,
        include_vol_features=False,
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
    }


def main():
    print(f"Loading {CKPT}")
    model = PPO.load(CKPT, device='cpu')

    out = {
        'checkpoint': CKPT,
        'description': 'SP100 no_regime model + cash floor sweep',
        'by_floor': {},
    }

    period_dfs = {p: load_df(csv) for p, csv in PERIODS.items()}
    print(f"\n{'floor':>5} | {'val':>8} {'bear':>8} {'2023':>8} {'bull':>8} | val_Sh    bear_Sh")
    print('-' * 80)
    for f in FLOORS:
        per = {}
        for p, df in period_dfs.items():
            per[p] = eval_with_floor(model, df, f)
        out['by_floor'][f"{f}"] = per
        print(f"{f:>5.2f} | "
              f"{per['val']['cumulative_return']:>+8.2f} "
              f"{per['test_bear']['cumulative_return']:>+8.2f} "
              f"{per['test_2023']['cumulative_return']:>+8.2f} "
              f"{per['test_bull']['cumulative_return']:>+8.2f} | "
              f"{per['val']['sharpe_ratio']:>+6.3f}  "
              f"{per['test_bear']['sharpe_ratio']:>+6.3f}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_PATH}")


if __name__ == '__main__':
    main()
