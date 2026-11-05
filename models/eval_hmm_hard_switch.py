"""
HMM-direct hard-switch baseline.

Routes 100% to the specialist whose role matches HMM's argmax regime that day:
  regime 0 (bull)     → 100% hunter
  regime 1 (bear)     → 100% defender
  regime 2 (sideways) → 100% steady

Runs through the same MACMetaEnvV3 master env so floor preconditioning + execution
are identical to the trained MAC. The only thing different is the routing decision.

This is the ablation baseline: "what if we just used HMM directly with no learned
controller?" Answer = this script's output.
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


def load(path):
    df = pd.read_csv(f'data/{path}')
    df['date'] = pd.to_datetime(df['date'])
    df = df.drop(columns=[c for c in V3_DROP if c in df.columns])
    return df


def regime_to_logits(regime_id):
    """One-hot routing logits.
       regime 0 (bull)   → aggressive (index 2)
       regime 1 (bear)   → defensive  (index 0)
       regime 2 (sideways) → balanced (index 1)
    """
    logits = np.array([-5.0, -5.0, -5.0], dtype=np.float32)
    if regime_id == 0:    logits[2] = 5.0
    elif regime_id == 1:  logits[0] = 5.0
    else:                 logits[1] = 5.0
    return logits


def evaluate(df, specialists, env_kwargs):
    env = MACMetaEnvV3(df, specialists, env_kwargs, V3_TRAINING_FLOORS)
    obs, _ = env.reset()
    records, routing_log = [], []
    done = False
    while not done:
        # Use the CURRENT day's regime_id to build one-hot logits
        step = env.master_env.current_step
        regime_id = int(env.master_env.regime_id_array[step])
        logits = regime_to_logits(regime_id)
        obs, r, term, trunc, info = env.step(logits)
        records.append(env.portfolio_value)
        routing_log.append(info['routing'])
        done = term or trunc
    m = calculate_metrics(pd.DataFrame({'value': records}))
    n = len(routing_log)
    avg_routing = {
        'def': sum(r['w_def'] for r in routing_log) / n,
        'bal': sum(r['w_bal'] for r in routing_log) / n,
        'agg': sum(r['w_agg'] for r in routing_log) / n,
    }
    return m, avg_routing


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
        cash_floors={0:0.0,1:0.0,2:0.0},
        risk_penalties={0:0.1,1:0.5,2:0.3},
        include_regime=True,
        include_regime_probs=True,
        include_vol_features=True,
    )

    print("=" * 92)
    print(f"   HMM-direct HARD-SWITCH baseline — {args.universe.upper()}")
    print(f"   One-hot routing on HMM argmax every day. No learned controller.")
    print("=" * 92)
    print()
    out = {}
    print(f"  {'period':>10} | {'cum_ret':>8} {'sharpe':>7} {'mdd':>7} | def/bal/agg avg")
    print(f"  {'-'*10} + {'-'*8} {'-'*7} {'-'*7} + {'-'*30}")
    for p, csv in periods.items():
        df = load(csv)
        m, r = evaluate(df, specs, env_kwargs)
        out[p] = {'metrics': m, 'routing': r}
        print(f"  {p:>10} | {m['cumulative_return']:>+8.2f} {m['sharpe_ratio']:>+7.3f} "
              f"{m['max_drawdown']:>+7.2f} | "
              f"def={r['def']:.2f} bal={r['bal']:.2f} agg={r['agg']:.2f}")

    out_path = f'results/mac_{args.universe}_v3/hmm_hard_switch.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n✅ → {out_path}")


if __name__ == '__main__':
    main()
