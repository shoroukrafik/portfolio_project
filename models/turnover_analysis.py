"""
Turnover analysis — measures how aggressively each system rebalances.

For each day t, turnover_t = 0.5 * sum(|w_t - w_{t-1}|)
  = fraction of portfolio reshuffled that day.

Reported per system per period:
  mean turnover (% per day)
  total turnover (% over full period — sum of daily turnovers)

Output:
  results/turnover.json
  results/turnover.md
  results/turnover.png
"""

import os
import sys
import json
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
from train_universal_v6_sp100 import V3_COLS_TO_DROP


def load_df(name):
    df = pd.read_csv(f'data/{name}')
    df['date'] = pd.to_datetime(df['date'])
    return df.drop(columns=[c for c in V3_COLS_TO_DROP if c in df.columns])


def daily_turnover(weights):
    """weights is list of np arrays — daily weights including cash."""
    w = np.stack(weights)
    diffs = np.abs(w[1:] - w[:-1])
    return 0.5 * diffs.sum(axis=1)  # daily turnover series


def env_kwargs_for_mac():
    return dict(
        initial_balance=1_000_000,
        cash_floors={0:0.0, 1:0.0, 2:0.0},
        risk_penalties={0:0.1, 1:0.5, 2:0.3},
        include_regime=True, include_regime_probs=True, include_vol_features=True,
    )


def env_kwargs_for_v6():
    return dict(
        initial_balance=1_000_000,
        cash_floors={0:0.0, 1:0.0, 2:0.0},
        risk_penalties={0:0.1, 1:0.5, 2:0.3},
        include_regime=True, include_regime_probs=True, include_vol_features=True,
        use_markowitz_reward=False, use_hyperheuristic=False,
    )


def run_mac_v3(universe, period_csv):
    df = load_df(period_csv)
    specs = SPECIALISTS_DJI_V3 if universe == 'dji' else SPECIALISTS_SP100_V3
    env = MACMetaEnvV3(df, specs, env_kwargs_for_mac(), V3_TRAINING_FLOORS)
    model = PPO.load(f'models/saved/ppo_mac_{universe}_v3.zip', device='cpu')
    obs, _ = env.reset()
    weights = [env.master_env.portfolio_weights.copy()]
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(action)
        weights.append(env.master_env.portfolio_weights.copy())  # MAC's master env

        done = term or trunc
    return weights


def run_universal_v6(universe, period_csv):
    df = load_df(period_csv)
    env = PortfolioEnv(df=df, **env_kwargs_for_v6())
    model = PPO.load(f'models/saved/ppo_universal_v6_{universe}.zip', device='cpu')
    obs, _ = env.reset()
    weights = [env.portfolio_weights.copy()]
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(action)
        weights.append(env.portfolio_weights.copy())
        done = term or trunc
    return weights


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
    out = {}
    for system, runner in [('MAC_v3', run_mac_v3), ('universal_v6', run_universal_v6)]:
        for u, P in [('dji', PERIODS_DJI), ('sp100', PERIODS_SP)]:
            for p, csv in P.items():
                key = f'{system}|{u}|{p}'
                print(f"  Running {key} ...", flush=True)
                w = runner(u, csv)
                t = daily_turnover(w)
                out[key] = {
                    'mean_daily_turnover_pct':  float(t.mean() * 100),
                    'median_daily_turnover_pct': float(np.median(t) * 100),
                    'total_period_turnover_pct': float(t.sum() * 100),
                    'n_days': int(len(t)),
                }
                print(f"    mean = {out[key]['mean_daily_turnover_pct']:.2f}%/day  "
                      f"total = {out[key]['total_period_turnover_pct']:.1f}%")

    os.makedirs('results', exist_ok=True)
    with open('results/turnover.json', 'w') as f:
        json.dump(out, f, indent=2)
    print("\n✅ saved results/turnover.json")

    # Markdown
    lines = ['# Portfolio turnover analysis\n']
    lines.append('Daily turnover = 0.5 × sum(|w_t - w_{t-1}|). 100% = full portfolio rebalanced.\n')
    lines.append('| System | Universe | Period | Mean daily turn (%) | Median (%) | Total (%) | Days |')
    lines.append('|---|---|---|---|---|---|---|')
    for k, v in out.items():
        sys_, u, p = k.split('|')
        lines.append(f"| {sys_} | {u.upper()} | {p} | "
                     f"{v['mean_daily_turnover_pct']:.2f} | "
                     f"{v['median_daily_turnover_pct']:.2f} | "
                     f"{v['total_period_turnover_pct']:.1f} | "
                     f"{v['n_days']} |")
    with open('results/turnover.md', 'w') as f:
        f.write('\n'.join(lines))
    print("✅ saved results/turnover.md")

    # PNG
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    periods = ['val', 'test_bear', 'test_2023', 'test_bull']
    for ax_idx, u in enumerate(['dji', 'sp100']):
        ax = axes[ax_idx]
        x = np.arange(len(periods))
        w = 0.35
        for off, (sys_, color) in enumerate([('MAC_v3', '#1f77b4'), ('universal_v6', '#d62728')]):
            vals = [out[f'{sys_}|{u}|{p}']['mean_daily_turnover_pct'] for p in periods]
            ax.bar(x + (off - 0.5)*w, vals, w, label=sys_.replace('_',' '), color=color, alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(periods)
        ax.set_ylabel('Mean daily turnover (%)')
        ax.set_title(f'{u.upper()} — mean daily turnover by system')
        ax.grid(axis='y', alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.savefig('results/turnover.png', dpi=130, bbox_inches='tight')
    print("✅ saved results/turnover.png")
    plt.close()


if __name__ == '__main__':
    main()
