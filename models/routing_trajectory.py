"""
Deep-dive C — MAC v3 routing trajectory by day, with HMM regime overlay.

Records (w_def, w_bal, w_agg) and HMM regime per day, plots stacked area
chart with regime background bands. Visual proof that the MAC controller
responds to regime signals.

Outputs:
  results/routing_data.json
  results/routing_trajectory.png  (one panel per period × universe = 8 panels)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from stable_baselines3 import PPO
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

REGIME_COLOR = {0: '#c8e6c9', 1: '#ffcdd2', 2: '#fff9c4'}
REGIME_NAME  = {0: 'bull', 1: 'bear', 2: 'sideways'}


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


def trace_mac(universe, period_csv):
    df = load_df(period_csv)
    specs = SPECIALISTS_DJI_V3 if universe == 'dji' else SPECIALISTS_SP100_V3
    env = MACMetaEnvV3(df, specs, env_kwargs(), V3_TRAINING_FLOORS)
    model = PPO.load(f'models/saved/ppo_mac_{universe}_v3.zip', device='cpu')

    obs, _ = env.reset()
    dates = []
    weights = []  # list of (def, bal, agg) softmax weights
    regimes = []
    done = False
    while not done:
        # Record the regime BEFORE step (the regime that drove this decision)
        step = env.master_env.current_step
        regime_id = int(env.master_env.regime_id_array[step])
        date = env.master_env.dates[step]
        action, _ = model.predict(obs, deterministic=True)
        # softmax of action → routing weights
        a = np.array(action, dtype=np.float32)
        a = a - a.max()
        w = np.exp(a)
        w = w / w.sum()
        dates.append(pd.to_datetime(date))
        weights.append(w.tolist())
        regimes.append(regime_id)
        obs, _, term, trunc, _ = env.step(action)
        done = term or trunc
    return dates, np.array(weights), np.array(regimes)


def plot_trajectory(ax, dates, weights, regimes, title):
    """Stacked area of routing weights with regime background."""
    # Regime background bands
    rid_curr = regimes[0]
    start_i = 0
    for i in range(1, len(regimes) + 1):
        if i == len(regimes) or regimes[i] != rid_curr:
            ax.axvspan(dates[start_i], dates[i-1],
                       color=REGIME_COLOR[rid_curr], alpha=0.6, zorder=0)
            if i < len(regimes):
                rid_curr = regimes[i]
                start_i = i
    # Stacked area: def (bottom blue), bal (middle green), agg (top orange)
    ax.fill_between(dates, 0, weights[:, 0], color='#1f77b4', alpha=0.85, label='defender')
    ax.fill_between(dates, weights[:, 0], weights[:, 0]+weights[:, 1],
                     color='#2ca02c', alpha=0.85, label='steady')
    ax.fill_between(dates, weights[:, 0]+weights[:, 1], 1.0,
                     color='#ff7f0e', alpha=0.85, label='hunter')
    ax.set_ylim(0, 1)
    ax.set_ylabel('routing weight')
    ax.set_title(title, fontsize=10)
    ax.grid(axis='y', alpha=0.3)


def main():
    results = {}
    fig, axes = plt.subplots(4, 2, figsize=(16, 12), sharey=True)
    period_order = ['val','test_bear','test_2023','test_bull']
    for j, u in enumerate(['dji', 'sp100']):
        results[u] = {}
        for i, p in enumerate(period_order):
            print(f"  Tracing MAC v3 {u}|{p}...")
            dates, weights, regimes = trace_mac(u, PERIODS[u][p])
            results[u][p] = {
                'dates':   [d.strftime('%Y-%m-%d') for d in dates],
                'weights': weights.tolist(),
                'regimes': regimes.tolist(),
                'regime_breakdown': {
                    REGIME_NAME[k]: int((regimes == k).sum())
                    for k in [0, 1, 2]
                },
            }
            ax = axes[i, j]
            title = f"{u.upper()} | {p}  ({len(dates)} days)"
            plot_trajectory(ax, dates, weights, regimes, title)
            # Annotate routing averages on top of plot
            avg_w = weights.mean(axis=0)
            ax.text(0.02, 0.97,
                     f"avg: def={avg_w[0]:.2f} bal={avg_w[1]:.2f} agg={avg_w[2]:.2f}",
                     transform=ax.transAxes, fontsize=8, va='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Legend (specialists + regime bands)
    handles = [
        Patch(facecolor='#1f77b4', alpha=0.85, label='defender weight'),
        Patch(facecolor='#2ca02c', alpha=0.85, label='steady weight'),
        Patch(facecolor='#ff7f0e', alpha=0.85, label='hunter weight'),
        Patch(facecolor=REGIME_COLOR[0], alpha=0.6, label='HMM: bull regime'),
        Patch(facecolor=REGIME_COLOR[1], alpha=0.6, label='HMM: bear regime'),
        Patch(facecolor=REGIME_COLOR[2], alpha=0.6, label='HMM: sideways regime'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.02))
    plt.suptitle('MAC v3 routing trajectory — daily weights with HMM regime overlay',
                  fontsize=13, weight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig('results/routing_trajectory.png', dpi=130, bbox_inches='tight')
    plt.close()

    with open('results/routing_data.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)

    print("\n✅ saved results/routing_trajectory.png")
    print("✅ saved results/routing_data.json")


if __name__ == '__main__':
    main()
