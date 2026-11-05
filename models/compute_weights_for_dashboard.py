"""
Compute per-day portfolio weights for any (variant, period) pair and save
in legacy-curves JSON format so the dashboard's concentration view can
display them.

Used to add concentration views for variants without legacy weight curves
(e.g. NDX production, NDX 2017+mom, SP100 +2015+mom).

Output format (matches dashboard_legacy/data/curves/{slug}__{period}.json):
{
  "label":    "<variant label>",
  "category": "main",
  "universe": "NDX 68 / SP100 90 / etc",
  "period":   "val|test_bear|test_2023|test_bull",
  "records":  [
    {"date": "...", "value": ..., "regime_name": "...",
     "weights": [...], "cash_weight": ..., "reward": ...},
    ...
  ]
}

Usage:
  python3 models/compute_weights_for_dashboard.py --variant ndx
  python3 models/compute_weights_for_dashboard.py --variant ndx_2017_mom
  python3 models/compute_weights_for_dashboard.py --variant sp100_2015_mom
"""

import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from portfolio_env import PortfolioEnv
from train_universal_v6_sp100 import V3_COLS_TO_DROP


# ─────────────────────────────────────────────────────────────────────────
# Variant config: model path + period CSVs + output slug
# ─────────────────────────────────────────────────────────────────────────
VARIANT_CONFIG = {
    'ndx': {
        'label':  'NDX v6',
        'slug':   'NDX_v6',
        'model':  'models/saved/ppo_universal_v6_ndx.zip',
        'csvs': {
            'val':       'val_data_ndx.csv',
            'test_bear': 'test_bear_data_ndx.csv',
            'test_2023': 'test_2023_data_ndx.csv',
            'test_bull': 'test_bull_data_ndx.csv',
        },
        'drop_long_mom': True,  # NDX prod was trained before long-mom existed
    },
    'ndx_2017': {
        'label':  'NDX v6 + 2017',
        'slug':   'NDX_2017',
        'model':  'models/saved/ppo_universal_v6_ndx_2017.zip',
        'csvs': {
            'val':       'val_data_ndx_2017.csv',
            'test_bear': 'test_bear_data_ndx_2017.csv',
            'test_2023': 'test_2023_data_ndx_2017.csv',
            'test_bull': 'test_bull_data_ndx_2017.csv',
        },
        'drop_long_mom': True,
    },
    'ndx_2017_mom': {
        'label':  'NDX v6 + 2017 + mom',
        'slug':   'NDX_2017_mom',
        'model':  'models/saved/ppo_universal_v6_ndx_2017_mom.zip',
        'csvs': {
            'val':       'val_data_ndx_2017.csv',
            'test_bear': 'test_bear_data_ndx_2017.csv',
            'test_2023': 'test_2023_data_ndx_2017.csv',
            'test_bull': 'test_bull_data_ndx_2017.csv',
        },
        'drop_long_mom': False,  # trained WITH long-mom
    },
    'sp100_2015_mom': {
        'label':  'SP100 v6 + 2015 + mom',
        'slug':   'SP100_2015_mom',
        'model':  'models/saved/ppo_universal_v6_sp100_2015_mom.zip',
        'csvs': {
            'val':       'val_data_sp100_2015.csv',
            'test_bear': 'test_bear_data_sp100_2015.csv',
            'test_2023': 'test_2023_data_sp100_2015.csv',
            'test_bull': 'test_bull_data_sp100_2015.csv',
        },
        'drop_long_mom': False,
    },
}


def env_kwargs():
    return dict(
        initial_balance=1_000_000,
        cash_floors={0: 0.0, 1: 0.0, 2: 0.0},
        risk_penalties={0: 0.1, 1: 0.5, 2: 0.3},
        include_regime=True, include_regime_probs=True, include_vol_features=True,
        use_markowitz_reward=False, use_hyperheuristic=False,
    )


def load_df(period_csv, drop_long_mom=False):
    df = pd.read_csv(f'data/{period_csv}')
    df['date'] = pd.to_datetime(df['date'])
    drop_cols = [c for c in V3_COLS_TO_DROP if c in df.columns]
    if drop_long_mom:
        drop_cols += [c for c in ('mom_120d_norm', 'mom_252d_norm') if c in df.columns]
    return df.drop(columns=drop_cols)


def run_period(model, df, period_name, label, universe_desc):
    env = PortfolioEnv(df=df, **env_kwargs())
    obs, info = env.reset()

    records = []
    done = False
    step_idx = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Current portfolio state
        # portfolio_weights is [stock_1, ..., stock_N, cash]
        stock_weights = env.portfolio_weights[:-1].astype(float).tolist()
        cash_w = float(env.portfolio_weights[-1])
        port_value = float(env.portfolio_value)

        # Regime at current step
        step_idx_safe = min(env.current_step, env.n_days - 1)
        regime_id = int(env.regime_id_array[step_idx_safe]) if hasattr(env, 'regime_id_array') else 0
        regime_name_map = {0: 'bull', 1: 'bear', 2: 'sideways'}
        regime_name = regime_name_map.get(regime_id, 'unknown')

        # Date for current step
        current_date = env.dates[step_idx_safe]
        date_str = pd.Timestamp(current_date).strftime('%Y-%m-%d')

        weights = stock_weights

        records.append({
            'date':         date_str,
            'value':        port_value,
            'regime_id':    regime_id,
            'regime_name':  regime_name,
            'weights':      weights,
            'cash_weight':  cash_w,
            'reward':       float(reward),
        })
        step_idx += 1

    return {
        'label':    label,
        'category': 'main',
        'universe': universe_desc,
        'period':   period_name,
        'records':  records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=list(VARIANT_CONFIG.keys()), required=True)
    parser.add_argument('--out_dir', default='dashboard_legacy/data/curves',
                        help='Output directory for the JSON curve files')
    args = parser.parse_args()

    cfg = VARIANT_CONFIG[args.variant]
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"=" * 80)
    print(f"   Computing per-day weights for {cfg['label']}")
    print(f"   Model: {cfg['model']}")
    print(f"=" * 80)

    if not os.path.exists(cfg['model']):
        print(f"ERROR: model not found at {cfg['model']}")
        return

    model = PPO.load(cfg['model'], device='cpu')

    for period_name, csv in cfg['csvs'].items():
        if not os.path.exists(f'data/{csv}'):
            print(f"  SKIP {period_name}: {csv} missing")
            continue
        print(f"  ▸ {period_name}: loading {csv} ...", end='', flush=True)
        df = load_df(csv, drop_long_mom=cfg['drop_long_mom'])
        try:
            result = run_period(model, df, period_name, cfg['label'],
                                 universe_desc=f"{args.variant} universe")
            out_path = os.path.join(args.out_dir, f"{cfg['slug']}__{period_name}.json")
            with open(out_path, 'w') as f:
                json.dump(result, f, separators=(',', ':'))
            print(f" {len(result['records'])} days  →  {out_path}")
        except Exception as e:
            print(f" ❌ {type(e).__name__}: {e}")

    print()
    print("✅ Done. The dashboard concentration view will now show this variant.")


if __name__ == '__main__':
    main()
