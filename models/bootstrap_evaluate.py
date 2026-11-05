"""
Bootstrap-CI evaluation for thesis statistical rigor.

For each (system, universe, period):
  1. Run the production model through the period env
  2. Extract daily portfolio values → daily returns
  3. Bootstrap-resample N=1000 times to estimate 95% CIs on:
        - cumulative return
        - Sharpe ratio
        - max drawdown

Output:  results/bootstrap_ci.json   (used by ablation_summary upgrade)
         results/bootstrap_ci.md     (thesis table)
         results/bootstrap_ci.png    (visual)
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from stable_baselines3 import PPO
from portfolio_env import PortfolioEnv
from mac_meta_env_v3 import (
    MACMetaEnvV3, SPECIALISTS_DJI_V3, SPECIALISTS_SP100_V3,
    V3_TRAINING_FLOORS,
)
from train_universal_v6_sp100 import V3_COLS_TO_DROP, V6_FEATURE_COLUMNS


N_BOOT = 1000
SEED = 42


def load_df(name):
    df = pd.read_csv(f'data/{name}')
    df['date'] = pd.to_datetime(df['date'])
    return df.drop(columns=[c for c in V3_COLS_TO_DROP if c in df.columns])


def daily_returns_from_values(values):
    """portfolio values trajectory → daily returns array"""
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
    """Stationary block bootstrap on daily returns → CIs for cum_ret, Sharpe, MDD."""
    rng = np.random.default_rng(seed)
    n = len(daily_r)
    if n == 0:
        return {'cum_ret': (0, 0, 0), 'sharpe': (0, 0, 0), 'mdd': (0, 0, 0)}
    block = max(5, int(n ** 0.5))  # ~ sqrt-n block length, std for finance bootstrap
    cum, shp, dd = [], [], []
    for _ in range(n_boot):
        # circular stationary bootstrap with geometric block length (mean=block)
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


# --- System runners ---------------------------------------------------------
def run_mac_v3(universe, period_csv):
    df = load_df(period_csv)
    if universe == 'dji':
        specs = SPECIALISTS_DJI_V3
    else:
        specs = SPECIALISTS_SP100_V3
    env_kwargs = dict(
        initial_balance=1_000_000,
        cash_floors={0:0.0, 1:0.0, 2:0.0},
        risk_penalties={0:0.1, 1:0.5, 2:0.3},
        include_regime=True, include_regime_probs=True, include_vol_features=True,
    )
    env = MACMetaEnvV3(df, specs, env_kwargs, V3_TRAINING_FLOORS)
    model = PPO.load(f'models/saved/ppo_mac_{universe}_v3.zip', device='cpu')
    obs, _ = env.reset()
    values = [env.portfolio_value]
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(action)
        values.append(env.portfolio_value)
        done = term or trunc
    return values


def run_universal_v6(universe, period_csv):
    df = load_df(period_csv)
    env_kwargs = dict(
        initial_balance=1_000_000,
        cash_floors={0:0.0, 1:0.0, 2:0.0},
        risk_penalties={0:0.1, 1:0.5, 2:0.3},
        include_regime=True, include_regime_probs=True, include_vol_features=True,
        use_markowitz_reward=False, use_hyperheuristic=False,
    )
    env = PortfolioEnv(df=df, **env_kwargs)
    model = PPO.load(f'models/saved/ppo_universal_v6_{universe}.zip', device='cpu')
    obs, _ = env.reset()
    values = [env.portfolio_value]
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(action)
        values.append(env.portfolio_value)
        done = term or trunc
    return values


# --- Main -------------------------------------------------------------------
def main():
    PERIODS_DJI = {
        'val':       'val_data_ext.csv',
        'test_bear': 'test_bear_data_ext.csv',
        'test_2023': 'test_2023_data_ext.csv',
        'test_bull': 'test_bull_data_ext.csv',
    }
    PERIODS_SP = {
        'val':       'val_data_sp100.csv',
        'test_bear': 'test_bear_data_sp100.csv',
        'test_2023': 'test_2023_data_sp100.csv',
        'test_bull': 'test_bull_data_sp100.csv',
    }
    results = {}
    for system, runner in [('MAC_v3', run_mac_v3), ('universal_v6', run_universal_v6)]:
        for universe, periods in [('dji', PERIODS_DJI), ('sp100', PERIODS_SP)]:
            for pname, csv in periods.items():
                key = f'{system}|{universe}|{pname}'
                print(f"  Running {key} ...", flush=True)
                values = runner(universe, csv)
                r = daily_returns_from_values(values)
                ci = bootstrap_metric(r)
                results[key] = ci
                print(f"    Sharpe = {ci['sharpe'][0]:+.3f}  CI [{ci['sharpe'][1]:+.3f}, {ci['sharpe'][2]:+.3f}]   "
                      f"cum_ret = {ci['cum_ret'][0]:+.2f}%  CI [{ci['cum_ret'][1]:+.2f}, {ci['cum_ret'][2]:+.2f}]")

    os.makedirs('results', exist_ok=True)
    with open('results/bootstrap_ci.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ saved results/bootstrap_ci.json")

    # Markdown table
    lines = ['# Bootstrap 95% CIs (n=1000 stationary block bootstrap)\n']
    lines.append('| System | Universe | Period | Sharpe (95% CI) | Cum.Ret % (95% CI) | MDD % (95% CI) |')
    lines.append('|---|---|---|---|---|---|')
    for k, v in results.items():
        sys_, u, p = k.split('|')
        s = v['sharpe']; c = v['cum_ret']; m = v['mdd']
        lines.append(f"| {sys_} | {u.upper()} | {p} | "
                     f"{s[0]:+.3f} [{s[1]:+.3f}, {s[2]:+.3f}] | "
                     f"{c[0]:+.2f} [{c[1]:+.2f}, {c[2]:+.2f}] | "
                     f"{m[0]:+.2f} [{m[1]:+.2f}, {m[2]:+.2f}] |")
    with open('results/bootstrap_ci.md', 'w') as f:
        f.write('\n'.join(lines))
    print("✅ saved results/bootstrap_ci.md")

    # PNG — Sharpe with CIs per system per universe per period
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    period_order = ['val', 'test_bear', 'test_2023', 'test_bull']
    for ax_idx, u in enumerate(['dji', 'sp100']):
        ax = axes[ax_idx]
        x = np.arange(len(period_order))
        w = 0.35
        for off, (sys_, color) in enumerate([('MAC_v3', '#1f77b4'), ('universal_v6', '#d62728')]):
            point = [results[f'{sys_}|{u}|{p}']['sharpe'][0] for p in period_order]
            lo    = [results[f'{sys_}|{u}|{p}']['sharpe'][1] for p in period_order]
            hi    = [results[f'{sys_}|{u}|{p}']['sharpe'][2] for p in period_order]
            err = [[p - l for p, l in zip(point, lo)], [h - p for p, h in zip(point, hi)]]
            ax.bar(x + (off - 0.5)*w, point, w, yerr=err, label=sys_.replace('_', ' '),
                   color=color, capsize=4, alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(period_order)
        ax.set_ylabel('Sharpe ratio')
        ax.set_title(f'{u.upper()} — Sharpe with 95% bootstrap CI')
        ax.axhline(0, color='gray', lw=0.5)
        ax.grid(axis='y', alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.savefig('results/bootstrap_ci.png', dpi=130, bbox_inches='tight')
    print("✅ saved results/bootstrap_ci.png")
    plt.close()


if __name__ == '__main__':
    main()
