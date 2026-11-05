"""
Run each specialist ALONE on each test period.

Routes 100% to ONE specialist (defender / steady / hunter) for every day of
the period, through the SAME MACMetaEnvV3 master env so floor preconditioning
and execution are identical to MAC. Produces the 3×4 "personality on each
regime" grid used for the thesis analysis.

Output JSON layout:
{
  "defender": {"val": {...metrics...}, "test_bear": {...}, ...},
  "balanced": {...},
  "aggressive": {...},
}
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from mac_meta_env_v3 import (
    MACMetaEnvV3,
    SPECIALISTS_DJI_V3, SPECIALISTS_SP100_V3,
    V3_TRAINING_FLOORS,
)
from train_final_extended import calculate_metrics


V3_DROP = ['r_lag_1_norm','r_lag_5_norm','r_lag_10_norm',
           'r_lag_20_norm','r_lag_40_norm','r_lag_60_norm',
           'is_uptrend','is_sideways','is_downtrend']

# Routing logit index for each specialist
SPECIALIST_IDX = {'defensive': 0, 'balanced': 1, 'aggressive': 2}


def load(path):
    df = pd.read_csv(f'data/{path}')
    df['date'] = pd.to_datetime(df['date'])
    df = df.drop(columns=[c for c in V3_DROP if c in df.columns])
    return df


def fixed_logits(idx):
    """Return MAC action vector that softmaxes to 100% on specialist idx."""
    logits = np.full(3, -5.0, dtype=np.float32)
    logits[idx] = 5.0
    return logits


def evaluate(df, specialists, env_kwargs, spec_role):
    env = MACMetaEnvV3(df, specialists, env_kwargs, V3_TRAINING_FLOORS)
    obs, _ = env.reset()
    logits = fixed_logits(SPECIALIST_IDX[spec_role])
    records = []
    done = False
    while not done:
        obs, r, term, trunc, info = env.step(logits)
        records.append(env.portfolio_value)
        done = term or trunc
    return calculate_metrics(pd.DataFrame({'value': records}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--universe', choices=['dji','sp100'], required=True)
    args = parser.parse_args()

    if args.universe == 'dji':
        periods = {
            'val':       'val_data_ext.csv',
            'test_bear': 'test_bear_data_ext.csv',
            'test_2023': 'test_2023_data_ext.csv',
            'test_bull': 'test_bull_data_ext.csv',
        }
        specs = SPECIALISTS_DJI_V3
    else:
        periods = {
            'val':       'val_data_sp100.csv',
            'test_bear': 'test_bear_data_sp100.csv',
            'test_2023': 'test_2023_data_sp100.csv',
            'test_bull': 'test_bull_data_sp100.csv',
        }
        specs = SPECIALISTS_SP100_V3

    env_kwargs = dict(
        initial_balance=1_000_000,
        cash_floors={0:0.0, 1:0.0, 2:0.0},
        risk_penalties={0:0.1, 1:0.5, 2:0.3},
        include_regime=True,
        include_regime_probs=True,
        include_vol_features=True,
    )

    print("=" * 100)
    print(f"  SPECIALIST-ALONE GRID  —  {args.universe.upper()}")
    print(f"  Each cell: that specialist routed 100% for the WHOLE period")
    print("=" * 100)
    print()

    out = {}
    print(f"  {'specialist':>12} | {'val ret':>8} {'val mdd':>8} | {'bear ret':>9} {'bear mdd':>9} | "
          f"{'2023 ret':>9} {'2023 mdd':>9} | {'bull ret':>9} {'bull mdd':>9}")
    print(f"  {'-'*12} + {'-'*8} {'-'*8} + {'-'*9} {'-'*9} + {'-'*9} {'-'*9} + {'-'*9} {'-'*9}")
    for role in ['defensive', 'balanced', 'aggressive']:
        out[role] = {}
        cells = {}
        for p, csv in periods.items():
            df = load(csv)
            m = evaluate(df, specs, env_kwargs, role)
            out[role][p] = m
            cells[p] = (m['cumulative_return'], m['max_drawdown'])
        v, b, t, u = cells['val'], cells['test_bear'], cells['test_2023'], cells['test_bull']
        print(f"  {role:>12} | {v[0]:>+8.2f} {v[1]:>+8.2f} | {b[0]:>+9.2f} {b[1]:>+9.2f} | "
              f"{t[0]:>+9.2f} {t[1]:>+9.2f} | {u[0]:>+9.2f} {u[1]:>+9.2f}")

    out_path = f'results/mac_{args.universe}_v3/specialist_alone_grid.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n✅ → {out_path}")


if __name__ == '__main__':
    main()
