"""
Evaluate every MAC checkpoint on all periods + report benchmark wins.

Loads each step_*.zip checkpoint, runs it through val / test_bear / test_2023
/ test_bull, and reports head-to-head benchmark wins (matching v1's format).
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
from mac_meta_env import MACMetaEnv
from mac_meta_env_v1c import MACMetaEnvV1C
from mac_meta_env_v3 import (
    MACMetaEnvV3,
    SPECIALISTS_DJI_V3, SPECIALISTS_SP100_V3,
    V3_TRAINING_FLOORS,
)
from mac_meta_env_v5 import (
    MACMetaEnvV5,
    SPECIALISTS_DJI_V5, SPECIALISTS_SP100_V5,
    V5_TRAINING_FLOORS,
)
from mac_meta_env_v6 import (
    MACMetaEnvV6,
    SPECIALISTS_DJI_V6_ABL, SPECIALISTS_SP100_V6_ABL,
    V6_ABL_TRAINING_FLOORS,
)
from mac_meta_env_v7 import (
    MACMetaEnvV7,
    SPECIALISTS_DJI_V7_ABL, SPECIALISTS_SP100_V7_ABL,
    V7_ABL_TRAINING_FLOORS,
)
from mac_obs_wrappers import ShuffleRegimeMACObs, NoVolMACObs
from train_final_extended import calculate_metrics
from mars_pareto_controller import (
    SPECIALISTS_DJI, SPECIALISTS_SP100,
    SPECIALISTS_DJI_V1C, SPECIALISTS_SP100_V1C,
    V1C_ENV_CONFIG_DJI, V1C_ENV_CONFIG_SP100,
    V1C_TRAINING_FLOORS_DJI, V1C_TRAINING_FLOORS_SP100,
)


V3_DROP = ['r_lag_1_norm','r_lag_5_norm','r_lag_10_norm',
           'r_lag_20_norm','r_lag_40_norm','r_lag_60_norm',
           'is_uptrend','is_sideways','is_downtrend']


def load(path, drop_v3=True):
    df = pd.read_csv(f'data/{path}')
    df['date'] = pd.to_datetime(df['date'])
    if drop_v3:
        df = df.drop(columns=[c for c in V3_DROP if c in df.columns])
    return df


def evaluate_one(model_path, env_kwargs, specialists, period_csvs,
                  lineup='v6', env_cfg=None, floors=None,
                  shuffle_regime=False, no_vol=False, seed=42):
    """Run one MAC checkpoint over all periods. Return per-period metrics."""
    results = {}
    model = PPO.load(model_path, device='cpu')
    routing_summary = {}
    for period, csv in period_csvs.items():
        df = load(csv, drop_v3=(lineup != 'v1c'))
        if lineup == 'v1c':
            env = MACMetaEnvV1C(df, specialists, env_kwargs, env_cfg, floors)
        elif lineup == 'v3':
            env = MACMetaEnvV3(df, specialists, env_kwargs, V3_TRAINING_FLOORS)
        elif lineup == 'v5':
            env = MACMetaEnvV5(df, specialists, env_kwargs, V5_TRAINING_FLOORS)
        elif lineup == 'v6abl':
            env = MACMetaEnvV6(df, specialists, env_kwargs, V6_ABL_TRAINING_FLOORS)
        elif lineup == 'v7abl':
            env = MACMetaEnvV7(df, specialists, env_kwargs, V7_ABL_TRAINING_FLOORS)
        else:
            env = MACMetaEnv(df, specialists, env_kwargs)
        # Match training-time obs transformations so trained policy receives
        # the same obs shape/content distribution at eval.
        if shuffle_regime:
            env = ShuffleRegimeMACObs(env, seed=seed)
        if no_vol:
            env = NoVolMACObs(env)
        obs, _ = env.reset()
        done = False
        records = []
        w_log = []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            records.append(env.portfolio_value)
            w_log.append(info['routing'])
            done = term or trunc
        m = calculate_metrics(pd.DataFrame({'value': records}))
        results[period] = m
        # Track average routing this period
        n = len(w_log)
        routing_summary[period] = {
            'def': sum(r['w_def'] for r in w_log)/n,
            'bal': sum(r['w_bal'] for r in w_log)/n,
            'agg': sum(r['w_agg'] for r in w_log)/n,
        }
    return results, routing_summary


def count_wins(results, bench):
    wins = 0; total = 0
    for p, bs in bench.items():
        mars = results[p]['cumulative_return']
        for _, val in bs.items():
            total += 1
            if mars > val:
                wins += 1
    return wins, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--universe', choices=['dji','sp100'], required=True)
    parser.add_argument('--variant',  choices=['v2','v3','v4','v5','v6','v7'], default=None,
                        help='v2 = lineup v1c + ws. '
                             'v3 = lineup v3 + soft ws. '
                             'v4 = lineup v3 + one-hot ws. '
                             'v5 = drop balanced (ablation). '
                             'v6 = drop defender (ablation). '
                             'v7 = drop hunter   (ablation).')
    parser.add_argument('--lineup',   choices=['v6','v1c','v3','v5','v6abl','v7abl'], default='v6')
    parser.add_argument('--warm_start',  action='store_true')
    parser.add_argument('--no_warm_start', action='store_true')
    parser.add_argument('--warm_target', choices=['soft','one_hot'], default='soft')
    parser.add_argument('--shuffle_regime', action='store_true')
    parser.add_argument('--no_vol',         action='store_true')
    parser.add_argument('--ent_coef',       type=float, default=0.01)
    args = parser.parse_args()

    if args.variant == 'v2':
        args.lineup = 'v1c'
        args.warm_start = True
    elif args.variant == 'v3':
        args.lineup = 'v3'
        args.warm_start = True
        args.warm_target = 'soft'
    elif args.variant == 'v4':
        args.lineup = 'v3'
        args.warm_start = True
        args.warm_target = 'one_hot'
    elif args.variant == 'v5':
        args.lineup = 'v5'
        args.warm_start = True
        args.warm_target = 'soft'
    elif args.variant == 'v6':
        args.lineup = 'v6abl'
        args.warm_start = True
        args.warm_target = 'soft'
    elif args.variant == 'v7':
        args.lineup = 'v7abl'
        args.warm_start = True
        args.warm_target = 'soft'

    if args.no_warm_start:
        args.warm_start = False

    # Same suffix resolution as train_mac.py
    if args.lineup == 'v3' and args.warm_start and args.warm_target == 'one_hot':
        suffix = '_v4'
    elif args.lineup == 'v3' and args.warm_start:
        suffix = '_v3'
    elif args.lineup == 'v5' and args.warm_start:
        suffix = '_v5'
    elif args.lineup == 'v6abl' and args.warm_start:
        suffix = '_v6'
    elif args.lineup == 'v7abl' and args.warm_start:
        suffix = '_v7'
    elif args.lineup == 'v1c' and args.warm_start:
        suffix = '_v2'
    elif args.lineup == 'v3' and not args.warm_start:
        suffix = '_v3'
    else:
        suffix = f"_{args.lineup}" if args.lineup != 'v6' else ''
        if args.warm_start:
            suffix += '_ws'

    if args.no_warm_start:
        suffix += '_nowarm'
    if args.shuffle_regime:
        suffix += '_shuf'
    if args.no_vol:
        suffix += '_novol'
    if abs(args.ent_coef - 0.01) > 1e-6:
        suffix += f'_ent{int(round(args.ent_coef*100)):02d}'

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ck_dir = os.path.join(base, 'models', 'saved', 'checkpoints',
                           f'mac_{args.universe}{suffix}')
    out_dir = os.path.join(base, 'results', f'mac_{args.universe}{suffix}')
    os.makedirs(out_dir, exist_ok=True)

    if args.universe == 'dji':
        periods = {
            'val':       'val_data_ext.csv',
            'test_bear': 'test_bear_data_ext.csv',
            'test_2023': 'test_2023_data_ext.csv',
            'test_bull': 'test_bull_data_ext.csv',
        }
        bench = {
            'val':       {'DJI BH': 20.23, 'EW DJI': 25.09},
            'test_bear': {'DJI BH': -9.40, 'EW DJI': -10.25},
            'test_2023': {'DJI BH': 13.74},
            'test_bull': {'DJI BH': 12.88, 'EW DJI': 22.04},
        }
        specialists = SPECIALISTS_DJI
        specialists_v3 = SPECIALISTS_DJI_V3
    else:
        periods = {
            'val':       'val_data_sp100.csv',
            'test_bear': 'test_bear_data_sp100.csv',
            'test_2023': 'test_2023_data_sp100.csv',
            'test_bull': 'test_bull_data_sp100.csv',
        }
        bench = {
            'val':       {'SP500 BH': 28.79, 'OEX BH': 29.33, 'EW SP100': 33.11},
            'test_bear': {'SP500 BH': -19.95, 'OEX BH': -22.99, 'EW SP100': -9.48},
            'test_2023': {'SP500 BH': 24.73, 'OEX BH': 31.39, 'EW SP100': 21.17},
            'test_bull': {'SP500 BH': 24.54, 'OEX BH': 30.90, 'EW SP100': 18.16},
        }
        specialists = SPECIALISTS_SP100
        specialists_v3 = SPECIALISTS_SP100_V3

    # Swap specialists + env config for the chosen lineup
    if args.lineup == 'v1c':
        if args.universe == 'dji':
            specialists = SPECIALISTS_DJI_V1C
            env_cfg = V1C_ENV_CONFIG_DJI
            floors  = V1C_TRAINING_FLOORS_DJI
        else:
            specialists = SPECIALISTS_SP100_V1C
            env_cfg = V1C_ENV_CONFIG_SP100
            floors  = V1C_TRAINING_FLOORS_SP100
    elif args.lineup == 'v3':
        specialists = specialists_v3
        env_cfg = floors = None
    elif args.lineup == 'v5':
        specialists = SPECIALISTS_DJI_V5 if args.universe == 'dji' else SPECIALISTS_SP100_V5
        env_cfg = floors = None
    elif args.lineup == 'v6abl':
        specialists = SPECIALISTS_DJI_V6_ABL if args.universe == 'dji' else SPECIALISTS_SP100_V6_ABL
        env_cfg = floors = None
    elif args.lineup == 'v7abl':
        specialists = SPECIALISTS_DJI_V7_ABL if args.universe == 'dji' else SPECIALISTS_SP100_V7_ABL
        env_cfg = floors = None
    else:
        env_cfg = floors = None
    specialists = {k:v for k,v in specialists.items()
                   if k in ('defensive','balanced','aggressive')}

    if args.lineup == 'v1c':
        # v1c per-role envs each control their own flags — base kwargs
        # exclude include_regime_probs/include_vol_features
        env_kwargs = dict(
            initial_balance=1_000_000,
            cash_floors={0:0.0,1:0.0,2:0.0},
            risk_penalties={0:0.1,1:0.5,2:0.3},
            include_regime=True,
        )
    else:
        env_kwargs = dict(
            initial_balance=1_000_000,
            cash_floors={0:0.0,1:0.0,2:0.0},
            risk_penalties={0:0.1,1:0.5,2:0.3},
            include_regime=True,
            include_regime_probs=True,
            include_vol_features=True,
        )

    checkpoints = sorted(glob.glob(os.path.join(ck_dir, 'step_*.zip')))
    print(f"\n{'='*92}")
    print(f"   MAC {args.universe.upper()} — sweeping {len(checkpoints)} checkpoints")
    print(f"{'='*92}\n")
    print(f"  {'step':>8} | {'val':>7} {'bear':>7} {'2023':>7} {'bull':>7} | {'wins':>5} | routing avg (def/bal/agg)")
    print(f"  {'-'*8} + {'-'*7} {'-'*7} {'-'*7} {'-'*7} + {'-'*5} + {'-'*30}")

    sweep_rows = []
    for ck in checkpoints:
        step = int(os.path.basename(ck).replace('step_','').replace('.zip',''))
        print(f"  {step:>8,}", end='', flush=True)
        results, routing = evaluate_one(ck, env_kwargs, specialists, periods,
                                         lineup=args.lineup,
                                         env_cfg=env_cfg, floors=floors,
                                         shuffle_regime=args.shuffle_regime,
                                         no_vol=args.no_vol)
        wins, total = count_wins(results, bench)
        # Use bull-period routing to summarise behaviour
        r_bull = routing['test_bull']
        rstr = f"def={r_bull['def']:.2f} bal={r_bull['bal']:.2f} agg={r_bull['agg']:.2f}"
        v = results['val']['cumulative_return']
        b = results['test_bear']['cumulative_return']
        t = results['test_2023']['cumulative_return']
        u = results['test_bull']['cumulative_return']
        print(f" | {v:>+7.2f} {b:>+7.2f} {t:>+7.2f} {u:>+7.2f} | {wins:>2}/{total:>2} | {rstr}")
        sweep_rows.append({
            'step': step,
            'val_ret': v, 'test_bear_ret': b,
            'test_2023_ret': t, 'test_bull_ret': u,
            'val_sharpe': results['val']['sharpe_ratio'],
            'test_bear_sharpe': results['test_bear']['sharpe_ratio'],
            'test_2023_sharpe': results['test_2023']['sharpe_ratio'],
            'test_bull_sharpe': results['test_bull']['sharpe_ratio'],
            'wins': wins, 'total': total,
            'routing': routing,
        })

    # Best
    print(f"\n{'='*92}")
    print(f"  BEST checkpoints by criterion:")
    print(f"{'='*92}")
    best_by_wins   = max(sweep_rows, key=lambda r: r['wins'])
    best_by_sharpe = max(sweep_rows, key=lambda r: r['val_sharpe'])
    best_by_bull   = max(sweep_rows, key=lambda r: r['test_bull_ret'])
    best_by_bear   = max(sweep_rows, key=lambda r: r['test_bear_ret'])  # less-negative
    print(f"  by total wins:    step {best_by_wins['step']:,}   "
          f"{best_by_wins['wins']}/{best_by_wins['total']} wins   "
          f"(val {best_by_wins['val_ret']:+.2f}  bear {best_by_wins['test_bear_ret']:+.2f}  "
          f"2023 {best_by_wins['test_2023_ret']:+.2f}  bull {best_by_wins['test_bull_ret']:+.2f})")
    print(f"  by val Sharpe:    step {best_by_sharpe['step']:,}   "
          f"Sharpe {best_by_sharpe['val_sharpe']:.4f}   "
          f"{best_by_sharpe['wins']}/{best_by_sharpe['total']} wins")
    print(f"  by bull return:   step {best_by_bull['step']:,}   "
          f"{best_by_bull['test_bull_ret']:+.2f}%   "
          f"{best_by_bull['wins']}/{best_by_bull['total']} wins")
    print(f"  by bear return:   step {best_by_bear['step']:,}   "
          f"{best_by_bear['test_bear_ret']:+.2f}%   "
          f"{best_by_bear['wins']}/{best_by_bear['total']} wins")

    out_path = os.path.join(out_dir, 'checkpoint_sweep.json')
    with open(out_path, 'w') as f:
        json.dump(sweep_rows, f, indent=2)
    print(f"\n✅ → {out_path}")


if __name__ == '__main__':
    main()
