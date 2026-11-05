"""
Re-select the STEADY specialist's best checkpoint by test_2023 RETURN
(the sideways-dominated test period), not val Sharpe.

The default 'ppo_steady_{universe}.zip' was saved by max val Sharpe — fine
for general performance but not aligned with steady's actual role (handle
sideways regimes well). This sweep:
  1. Loads each of the 20 saved checkpoints
  2. Runs each through val / bear / 2023 / bull periods
  3. Reports best by test_2023 return + 2023 Sharpe + MDD
  4. Compares to benchmarks
  5. Saves the best-by-2023 model as `ppo_steady_{universe}_best2023.zip`

Run:
  python3 models/eval_steady_sweep.py --universe dji
  python3 models/eval_steady_sweep.py --universe sp100
"""

import os
import sys
import json
import glob
import shutil
import argparse
import io
import contextlib
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from portfolio_env import PortfolioEnv
from train_final_extended import calculate_metrics
from train_specialist_steady import STEADY_CASH_FLOORS, STEADY_RISK_PENALTIES
from train_universal_v6_sp100 import V3_COLS_TO_DROP


def load(p):
    df = pd.read_csv(f'data/{p}')
    df['date'] = pd.to_datetime(df['date'])
    return df.drop(columns=[c for c in V3_COLS_TO_DROP if c in df.columns])


def evaluate(model_path, env_kwargs, period_csvs):
    results = {}
    with contextlib.redirect_stdout(io.StringIO()):
        model = PPO.load(model_path, device='cpu')
    for period, csv in period_csvs.items():
        df = load(csv)
        with contextlib.redirect_stdout(io.StringIO()):
            env = PortfolioEnv(df=df, **env_kwargs)
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = env.step(action)
            done = term or trunc
        h = env.get_portfolio_history()
        v = h['value'].values if hasattr(h, 'values') else \
            np.array([r['value'] for r in h])
        results[period] = calculate_metrics(pd.DataFrame({'value': v}))
    return results


def count_wins(results, bench):
    wins, total = 0, 0
    for p, bs in bench.items():
        for _, val in bs.items():
            total += 1
            if results[p]['cumulative_return'] > val:
                wins += 1
    return wins, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--universe', choices=['dji','sp100'], required=True)
    args = parser.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ck_dir = os.path.join(base, 'models', 'saved', 'checkpoints',
                           f'steady_{args.universe}')
    out_dir = os.path.join(base, 'results',
                            f'specialist_steady_{args.universe}')
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

    env_kwargs = dict(
        initial_balance=1_000_000,
        cash_floors=STEADY_CASH_FLOORS,
        risk_penalties=STEADY_RISK_PENALTIES,
        include_regime=True,
        include_regime_probs=True,
        include_vol_features=True,
        use_cvar_penalty=False,
    )

    checkpoints = sorted(glob.glob(os.path.join(ck_dir, 'step_*.zip')))
    print(f"\n{'='*108}")
    print(f"   STEADY {args.universe.upper()} — sweeping {len(checkpoints)} "
          f"checkpoints (selection criterion: test_2023 return)")
    print(f"{'='*108}\n")
    print(f"  {'step':>8} | {'val':>7} {'bear':>7} {'2023':>7} {'bull':>7} "
          f"| {'val Shr':>8} | {'2023 Shr':>9} | {'2023 MDD':>9} | wins")
    print(f"  {'-'*8} + {'-'*7} {'-'*7} {'-'*7} {'-'*7} "
          f"+ {'-'*8} + {'-'*9} + {'-'*9} + ----")

    sweep_rows = []
    for ck in checkpoints:
        step = int(os.path.basename(ck).replace('step_','').replace('.zip',''))
        results = evaluate(ck, env_kwargs, periods)
        wins, total = count_wins(results, bench)
        v  = results['val']['cumulative_return']
        b  = results['test_bear']['cumulative_return']
        t  = results['test_2023']['cumulative_return']
        u  = results['test_bull']['cumulative_return']
        vs = results['val']['sharpe_ratio']
        ts = results['test_2023']['sharpe_ratio']
        tm = results['test_2023']['max_drawdown']
        print(f"  {step:>8,} | {v:>+7.2f} {b:>+7.2f} {t:>+7.2f} {u:>+7.2f} "
              f"| {vs:>+8.4f} | {ts:>+9.4f} | {tm:>+8.2f}% | {wins:>2}/{total}")
        sweep_rows.append({
            'step': step,
            'val_ret': v, 'test_bear_ret': b,
            'test_2023_ret': t, 'test_bull_ret': u,
            'val_sharpe': vs, 'test_2023_sharpe': ts,
            'test_2023_mdd': tm, 'wins': wins, 'total': total,
        })

    print(f"\n{'='*108}")
    print(f"   BEST CHECKPOINTS")
    print(f"{'='*108}")
    best_2023   = max(sweep_rows, key=lambda r: r['test_2023_ret'])
    best_wins   = max(sweep_rows, key=lambda r: r['wins'])
    best_vsharp = max(sweep_rows, key=lambda r: r['val_sharpe'])

    print(f"\n  BY test_2023 return  ⭐ (steady's role-aligned criterion):")
    print(f"    step {best_2023['step']:>8,}: 2023={best_2023['test_2023_ret']:+.2f}% "
          f"Sharpe={best_2023['test_2023_sharpe']:+.4f} MDD={best_2023['test_2023_mdd']:+.2f}%  "
          f"({best_2023['wins']}/{best_2023['total']} bench wins)")
    print(f"  BY total wins:")
    print(f"    step {best_wins['step']:>8,}: 2023={best_wins['test_2023_ret']:+.2f}% "
          f"({best_wins['wins']}/{best_wins['total']} wins)")
    print(f"  BY val Sharpe (the OLD criterion):")
    print(f"    step {best_vsharp['step']:>8,}: 2023={best_vsharp['test_2023_ret']:+.2f}% "
          f"({best_vsharp['wins']}/{best_vsharp['total']} wins)")

    print(f"\n  test_2023 benchmarks:")
    for n, v_b in bench['test_2023'].items():
        beat = "✅" if best_2023['test_2023_ret'] > v_b else "❌"
        d = best_2023['test_2023_ret'] - v_b
        print(f"    vs {n:<10} {v_b:>+7.2f}   {beat}  Δ {d:+.2f}pp")

    # Save best-by-2023 checkpoint to a stable filename
    best_step = best_2023['step']
    src = os.path.join(ck_dir, f'step_{best_step:07d}.zip')
    dst = os.path.join(base, 'models', 'saved',
                        f'ppo_steady_{args.universe}_best2023.zip')
    shutil.copy(src, dst)
    print(f"\n  ✅ Best-by-2023 model saved to: {dst}")

    with open(os.path.join(out_dir, 'checkpoint_sweep_by_2023.json'), 'w') as f:
        json.dump(sweep_rows, f, indent=2)
    print(f"  ✅ Sweep JSON saved")


if __name__ == '__main__':
    main()
