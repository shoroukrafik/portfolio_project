"""
Bootstrap 95% CIs for the canonical DEFENSIVE+floor pipeline.

For each canonical model:
  DJI 30        — step  800,000, bear cash floor = 0.7
  S&P 100       — step 1,150,000, bear cash floor = 0.3
  NASDAQ-100    — step    100,000, no cash floor (0.0)

Runs the trained checkpoint with its canonical floor on val + 3 test
periods, extracts daily portfolio values, then stationary-block
bootstraps (N=1000, block ≈ sqrt(T)) to produce 95% CIs for cumulative
return, annualised Sharpe, and max drawdown.

Output:
  results/_cross_analyses/bootstrap_ci/bootstrap_ci_canonical.json
  results/_cross_analyses/bootstrap_ci/bootstrap_ci_canonical.md
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


N_BOOT = 1000
SEED = 42

CANONICAL = {
    'dji': {
        'ckpt':  'models/saved/checkpoints/universal_v6_dji_mom_2m/step_0800000.zip',
        'floor': 0.7,
        'periods': {
            'val':       'val_data_ext.csv',
            'test_bear': 'test_bear_data_ext.csv',
            'test_2023': 'test_2023_data_ext.csv',
            'test_bull': 'test_bull_data_ext.csv',
        },
    },
    'sp100': {
        'ckpt':  'models/saved/checkpoints/universal_v6_sp100_mom_2m/step_1150000.zip',
        'floor': 0.3,
        'periods': {
            'val':       'val_data_sp100.csv',
            'test_bear': 'test_bear_data_sp100.csv',
            'test_2023': 'test_2023_data_sp100.csv',
            'test_bull': 'test_bull_data_sp100.csv',
        },
    },
    'ndx': {
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


def load_df(name):
    df = pd.read_csv(f'data/{name}')
    df['date'] = pd.to_datetime(df['date'])
    return df.drop(columns=[c for c in V3_COLS_TO_DROP if c in df.columns])


def daily_returns_from_values(values):
    v = np.asarray(values, dtype=float)
    return v[1:] / v[:-1] - 1.0


def sharpe(r, periods=252):
    if len(r) == 0 or r.std() == 0:
        return 0.0
    return (r.mean() / r.std()) * np.sqrt(periods)


def cumret(r):
    return (np.prod(1.0 + np.asarray(r)) - 1.0) * 100.0


def mdd(r):
    eq = np.cumprod(1.0 + np.asarray(r))
    peak = np.maximum.accumulate(eq)
    return (eq / peak - 1.0).min() * 100.0


def bootstrap_metric(daily_r, n_boot=N_BOOT, seed=SEED):
    """Stationary block bootstrap on daily returns → 95% CIs."""
    rng = np.random.default_rng(seed)
    n = len(daily_r)
    if n == 0:
        return {'cum_ret': (0.0, 0.0, 0.0),
                'sharpe':  (0.0, 0.0, 0.0),
                'mdd':     (0.0, 0.0, 0.0)}
    block = max(5, int(n ** 0.5))
    cum, shp, dd = [], [], []
    for _ in range(n_boot):
        idx = []
        while len(idx) < n:
            start = rng.integers(0, n)
            L = max(1, rng.geometric(1.0 / block))
            for k in range(L):
                idx.append((start + k) % n)
        idx = np.array(idx[:n])
        rb = daily_r[idx]
        cum.append(cumret(rb))
        shp.append(sharpe(rb))
        dd.append(mdd(rb))

    def ci(arr, point):
        arr = np.array(arr)
        lo, hi = np.percentile(arr, [2.5, 97.5])
        return (float(point), float(lo), float(hi))

    return {
        'cum_ret': ci(cum, cumret(daily_r)),
        'sharpe':  ci(shp, sharpe(daily_r)),
        'mdd':     ci(dd, mdd(daily_r)),
    }


def run_canonical(universe, period_csv, floor):
    df = load_df(period_csv)
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
    model = PPO.load(CANONICAL[universe]['ckpt'], device='cpu')
    obs, _ = env.reset()
    values = [env.portfolio_value]
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(action)
        values.append(env.portfolio_value)
        done = term or trunc
    return values


def main():
    results = {}
    for universe, cfg in CANONICAL.items():
        floor = cfg['floor']
        for pname, csv in cfg['periods'].items():
            key = f'canonical|{universe}|{pname}'
            print(f"  Running {key} (floor={floor}) ...", flush=True)
            values = run_canonical(universe, csv, floor)
            r = daily_returns_from_values(values)
            ci = bootstrap_metric(r)
            results[key] = ci
            print(f"    Sharpe = {ci['sharpe'][0]:+.3f}  CI [{ci['sharpe'][1]:+.3f}, {ci['sharpe'][2]:+.3f}]   "
                  f"cum_ret = {ci['cum_ret'][0]:+.2f}%  CI [{ci['cum_ret'][1]:+.2f}, {ci['cum_ret'][2]:+.2f}]   "
                  f"MDD = {ci['mdd'][0]:+.2f}%  CI [{ci['mdd'][1]:+.2f}, {ci['mdd'][2]:+.2f}]")

    out_dir = 'results/_cross_analyses/bootstrap_ci'
    os.makedirs(out_dir, exist_ok=True)
    with open(f'{out_dir}/bootstrap_ci_canonical.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ saved {out_dir}/bootstrap_ci_canonical.json")

    lines = ['# Bootstrap 95% CIs (n=1000 stationary block bootstrap) — CANONICAL pipeline\n']
    lines.append('| Universe | Period | Sharpe (95% CI) | Cum.Ret % (95% CI) | MDD % (95% CI) |')
    lines.append('|---|---|---|---|---|')
    for k, v in results.items():
        _, u, p = k.split('|')
        s = v['sharpe']; c = v['cum_ret']; m = v['mdd']
        lines.append(f"| {u.upper()} | {p} | "
                     f"{s[0]:+.3f} [{s[1]:+.3f}, {s[2]:+.3f}] | "
                     f"{c[0]:+.2f} [{c[1]:+.2f}, {c[2]:+.2f}] | "
                     f"{m[0]:+.2f} [{m[1]:+.2f}, {m[2]:+.2f}] |")
    with open(f'{out_dir}/bootstrap_ci_canonical.md', 'w') as f:
        f.write('\n'.join(lines))
    print(f"✅ saved {out_dir}/bootstrap_ci_canonical.md")


if __name__ == '__main__':
    main()
