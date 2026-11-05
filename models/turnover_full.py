"""
Deep-dive A — Turnover per specialist + MAC v3 + v6 universal.

Extends results/turnover.json (which only had MAC + v6) to include
per-specialist turnover via the master env. Same metric:
   daily turnover_t = 0.5 * sum(|w_t - w_{t-1}|)
   100% = full portfolio reshuffled.

Outputs:
  results/turnover_full.json
  results/turnover_full.md
  results/turnover_full.png
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


PERIODS = {
    'dji': {
        'val':       'val_data_ext.csv',
        'test_bear': 'test_bear_data_ext.csv',
        'test_2023': 'test_2023_data_ext.csv',
        'test_bull': 'test_bull_data_ext.csv',
    },
    'sp100': {
        'val':       'val_data_sp100.csv',
        'test_bear': 'test_bear_data_sp100.csv',
        'test_2023': 'test_2023_data_sp100.csv',
        'test_bull': 'test_bull_data_sp100.csv',
    },
}
SPEC_IDX = {'defensive': 0, 'balanced': 1, 'aggressive': 2}


def load_df(name):
    df = pd.read_csv(f'data/{name}')
    df['date'] = pd.to_datetime(df['date'])
    return df.drop(columns=[c for c in V3_COLS_TO_DROP if c in df.columns])


def env_kwargs():
    return dict(
        initial_balance=1_000_000,
        cash_floors={0:0.0, 1:0.0, 2:0.0},
        risk_penalties={0:0.1, 1:0.5, 2:0.3},
        include_regime=True, include_regime_probs=True, include_vol_features=True,
    )


def daily_turnover(w):
    diffs = np.abs(w[1:] - w[:-1])
    return 0.5 * diffs.sum(axis=1)


def specialist_alone(universe, period_csv, role):
    df = load_df(period_csv)
    specs = SPECIALISTS_DJI_V3 if universe == 'dji' else SPECIALISTS_SP100_V3
    env = MACMetaEnvV3(df, specs, env_kwargs(), V3_TRAINING_FLOORS)
    logits = np.full(3, -5.0, dtype=np.float32)
    logits[SPEC_IDX[role]] = 5.0
    obs, _ = env.reset()
    weights = [env.master_env.portfolio_weights.copy()]
    done = False
    while not done:
        obs, _, term, trunc, _ = env.step(logits)
        weights.append(env.master_env.portfolio_weights.copy())
        done = term or trunc
    return np.stack(weights)


def mac_alone(universe, period_csv):
    df = load_df(period_csv)
    specs = SPECIALISTS_DJI_V3 if universe == 'dji' else SPECIALISTS_SP100_V3
    env = MACMetaEnvV3(df, specs, env_kwargs(), V3_TRAINING_FLOORS)
    model = PPO.load(f'models/saved/ppo_mac_{universe}_v3.zip', device='cpu')
    obs, _ = env.reset()
    weights = [env.master_env.portfolio_weights.copy()]
    done = False
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(a)
        weights.append(env.master_env.portfolio_weights.copy())
        done = term or trunc
    return np.stack(weights)


def v6_alone(universe, period_csv):
    df = load_df(period_csv)
    env = PortfolioEnv(df=df, **env_kwargs(),
                        use_markowitz_reward=False, use_hyperheuristic=False)
    model = PPO.load(f'models/saved/ppo_universal_v6_{universe}.zip', device='cpu')
    obs, _ = env.reset()
    weights = [env.portfolio_weights.copy()]
    done = False
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(a)
        weights.append(env.portfolio_weights.copy())
        done = term or trunc
    return np.stack(weights)


def main():
    SYSTEMS = [
        ('defender', specialist_alone, {'role': 'defensive'}),
        ('steady',   specialist_alone, {'role': 'balanced'}),
        ('hunter',   specialist_alone, {'role': 'aggressive'}),
        ('MAC v3',   mac_alone, {}),
        ('v6 univ',  v6_alone,  {}),
    ]
    results = {}
    for u in ['dji','sp100']:
        results[u] = {}
        for p in PERIODS[u]:
            print(f"  Computing turnover for {u} | {p} ...")
            results[u][p] = {}
            for label, fn, extra in SYSTEMS:
                w = fn(u, PERIODS[u][p], **extra)
                t = daily_turnover(w)
                results[u][p][label] = {
                    'mean_daily_turn_pct':   float(t.mean() * 100),
                    'median_daily_turn_pct': float(np.median(t) * 100),
                    'total_period_turn_pct': float(t.sum() * 100),
                    'n_days': int(len(t)),
                }

    os.makedirs('results', exist_ok=True)
    with open('results/turnover_full.json','w') as f:
        json.dump(results, f, indent=2, default=float)

    # Markdown
    md = ['# Turnover per system (Deep-dive A)\n',
          'Daily turnover = 0.5 × Σ|w_t − w_{t-1}|. 100% = full portfolio rebalanced.\n',
          'Specialists evaluated alone through the MAC master env (hard one-hot routing).\n',
          '## DJI mean daily turnover (%)\n',
          '| Period | defender | steady | hunter | MAC v3 | v6 univ |',
          '|---|---|---|---|---|---|']
    for p in ['val','test_bear','test_2023','test_bull']:
        r = results['dji'][p]
        md.append(f"| {p} | {r['defender']['mean_daily_turn_pct']:.2f} | "
                  f"{r['steady']['mean_daily_turn_pct']:.2f} | "
                  f"{r['hunter']['mean_daily_turn_pct']:.2f} | "
                  f"{r['MAC v3']['mean_daily_turn_pct']:.2f} | "
                  f"{r['v6 univ']['mean_daily_turn_pct']:.2f} |")
    md.append('\n## SP100 mean daily turnover (%)\n')
    md.append('| Period | defender | steady | hunter | MAC v3 | v6 univ |')
    md.append('|---|---|---|---|---|---|')
    for p in ['val','test_bear','test_2023','test_bull']:
        r = results['sp100'][p]
        md.append(f"| {p} | {r['defender']['mean_daily_turn_pct']:.2f} | "
                  f"{r['steady']['mean_daily_turn_pct']:.2f} | "
                  f"{r['hunter']['mean_daily_turn_pct']:.2f} | "
                  f"{r['MAC v3']['mean_daily_turn_pct']:.2f} | "
                  f"{r['v6 univ']['mean_daily_turn_pct']:.2f} |")

    with open('results/turnover_full.md','w') as f:
        f.write('\n'.join(md))

    # PNG
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    periods = ['val','test_bear','test_2023','test_bull']
    x = np.arange(len(periods))
    w = 0.16
    SYS_NAMES = ['defender','steady','hunter','MAC v3','v6 univ']
    COLORS    = ['#1f77b4','#2ca02c','#ff7f0e','#9467bd','#d62728']
    for ax_idx, u in enumerate(['dji','sp100']):
        ax = axes[ax_idx]
        for k, (sys_name, color) in enumerate(zip(SYS_NAMES, COLORS)):
            vals = [results[u][p][sys_name]['mean_daily_turn_pct'] for p in periods]
            ax.bar(x + (k - 2) * w, vals, w, label=sys_name, color=color, alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(periods)
        ax.set_ylabel('Mean daily turnover (%)')
        ax.set_title(f'{u.upper()} — daily turnover per system')
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=8)
    plt.suptitle('Daily turnover comparison: defender / steady / hunter vs MAC v3 vs v6 universal',
                  fontsize=12, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('results/turnover_full.png', dpi=130, bbox_inches='tight')
    plt.close()

    print('\n=== DJI mean daily turnover (%) ===')
    print(f"  {'Period':<11}   {'def':>6} {'steady':>7} {'hunter':>7} {'MAC v3':>7} {'v6 univ':>8}")
    for p in periods:
        r = results['dji'][p]
        print(f"  {p:<11}   "
              f"{r['defender']['mean_daily_turn_pct']:>5.2f}% "
              f"{r['steady']['mean_daily_turn_pct']:>6.2f}% "
              f"{r['hunter']['mean_daily_turn_pct']:>6.2f}% "
              f"{r['MAC v3']['mean_daily_turn_pct']:>6.2f}% "
              f"{r['v6 univ']['mean_daily_turn_pct']:>7.2f}%")
    print('\n=== SP100 mean daily turnover (%) ===')
    for p in periods:
        r = results['sp100'][p]
        print(f"  {p:<11}   "
              f"{r['defender']['mean_daily_turn_pct']:>5.2f}% "
              f"{r['steady']['mean_daily_turn_pct']:>6.2f}% "
              f"{r['hunter']['mean_daily_turn_pct']:>6.2f}% "
              f"{r['MAC v3']['mean_daily_turn_pct']:>6.2f}% "
              f"{r['v6 univ']['mean_daily_turn_pct']:>7.2f}%")
    print('\n✅ saved results/turnover_full.json')
    print('✅ saved results/turnover_full.md')
    print('✅ saved results/turnover_full.png')


if __name__ == '__main__':
    main()
