"""
Deep-dive B — Drawdown (underwater equity) profiles per system.

For each test period × universe, plot the rolling drawdown curve:
   drawdown_t = (value_t - running_max_t) / running_max_t * 100

Systems shown:
  - defender alone (single specialist through master env)
  - steady alone
  - hunter alone
  - MAC v3 (full lineup)
  - v6 universal single agent

Reveals defensive behavior visually: defender's flat curve vs hunter's
catastrophic dive during bear periods.

Outputs:
  results/drawdown_profiles.png  (4 periods × 2 universes = 8 panels)
  results/drawdown_data.json     (raw underwater curves)
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


def v6_env_kwargs():
    return dict(**env_kwargs(),
                 use_markowitz_reward=False, use_hyperheuristic=False)


def specialist_alone(universe, period_csv, role):
    """Run ONE specialist alone via MAC master env with hard one-hot routing."""
    df = load_df(period_csv)
    specs = SPECIALISTS_DJI_V3 if universe == 'dji' else SPECIALISTS_SP100_V3
    env = MACMetaEnvV3(df, specs, env_kwargs(), V3_TRAINING_FLOORS)
    logits = np.full(3, -5.0, dtype=np.float32)
    logits[SPEC_IDX[role]] = 5.0
    obs, _ = env.reset()
    values = [env.portfolio_value]
    dates  = [env.master_env.dates[0]]
    done = False
    while not done:
        obs, _, term, trunc, _ = env.step(logits)
        values.append(env.portfolio_value)
        step = min(env.master_env.current_step, env.master_env.n_days - 1)
        dates.append(env.master_env.dates[step])
        done = term or trunc
    return dates, values


def mac_alone(universe, period_csv):
    df = load_df(period_csv)
    specs = SPECIALISTS_DJI_V3 if universe == 'dji' else SPECIALISTS_SP100_V3
    env = MACMetaEnvV3(df, specs, env_kwargs(), V3_TRAINING_FLOORS)
    model = PPO.load(f'models/saved/ppo_mac_{universe}_v3.zip', device='cpu')
    obs, _ = env.reset()
    values = [env.portfolio_value]
    dates  = [env.master_env.dates[0]]
    done = False
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(a)
        values.append(env.portfolio_value)
        step = min(env.master_env.current_step, env.master_env.n_days - 1)
        dates.append(env.master_env.dates[step])
        done = term or trunc
    return dates, values


def v6_alone(universe, period_csv):
    df = load_df(period_csv)
    env = PortfolioEnv(df=df, **v6_env_kwargs())
    model = PPO.load(f'models/saved/ppo_universal_v6_{universe}.zip', device='cpu')
    obs, _ = env.reset()
    values = [env.portfolio_value]
    dates  = [env.dates[0]]
    done = False
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(a)
        values.append(env.portfolio_value)
        step = min(env.current_step, env.n_days - 1)
        dates.append(env.dates[step])
        done = term or trunc
    return dates, values


def underwater(values):
    v = np.array(values, dtype=float)
    peak = np.maximum.accumulate(v)
    return (v / peak - 1.0) * 100


def main():
    SYSTEMS = [
        ('defender', '#1f77b4', '--', specialist_alone, {'role': 'defensive'}),
        ('steady',   '#2ca02c', '--', specialist_alone, {'role': 'balanced'}),
        ('hunter',   '#ff7f0e', '--', specialist_alone, {'role': 'aggressive'}),
        ('MAC v3',   '#9467bd', '-',  mac_alone,         {}),
        ('v6 univ',  '#d62728', '-',  v6_alone,          {}),
    ]
    results = {}
    fig, axes = plt.subplots(4, 2, figsize=(16, 13), sharey=True)
    period_order = ['val','test_bear','test_2023','test_bull']

    for j, u in enumerate(['dji', 'sp100']):
        results[u] = {}
        for i, p in enumerate(period_order):
            print(f"  Computing drawdowns for {u} | {p} ...")
            ax = axes[i, j]
            results[u][p] = {}
            for label, color, ls, fn, extra in SYSTEMS:
                dates, values = fn(u, PERIODS[u][p], **extra)
                dd = underwater(values)
                results[u][p][label] = {
                    'min_dd': float(dd.min()),
                    'final_value_idx': float(values[-1] / values[0] - 1) * 100,
                }
                ax.plot(dates, dd, label=label, color=color, linestyle=ls, lw=1.4, alpha=0.9)
            ax.set_title(f"{u.upper()} | {p}", fontsize=10)
            ax.set_ylabel('Drawdown (%)')
            ax.axhline(0, color='gray', lw=0.5)
            ax.grid(alpha=0.3)
            ax.set_ylim(bottom=ax.get_ylim()[0])
            if i == 0 and j == 0:
                ax.legend(loc='lower left', fontsize=8, ncol=2)

    plt.suptitle('Drawdown profiles — defender / steady / hunter (specialist-alone) '
                  'vs MAC v3 vs v6 universal',
                  fontsize=13, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig('results/drawdown_profiles.png', dpi=130, bbox_inches='tight')
    plt.close()

    with open('results/drawdown_data.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)

    print('\n=== MIN DRAWDOWNS (% — closer to 0 is better) ===')
    print(f"  {'universe':<8} {'period':<10}   {'def':>7} {'steady':>7} {'hunter':>7} {'MAC v3':>7} {'v6 univ':>7}")
    for u in ['dji','sp100']:
        for p in period_order:
            row = results[u][p]
            print(f"  {u.upper():<8} {p:<10}   "
                  f"{row['defender']['min_dd']:>+7.2f} {row['steady']['min_dd']:>+7.2f} "
                  f"{row['hunter']['min_dd']:>+7.2f} {row['MAC v3']['min_dd']:>+7.2f} "
                  f"{row['v6 univ']['min_dd']:>+7.2f}")
    print('\n✅ saved results/drawdown_profiles.png')
    print('✅ saved results/drawdown_data.json')


if __name__ == '__main__':
    main()
