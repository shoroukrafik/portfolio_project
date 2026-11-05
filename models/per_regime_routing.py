"""
Deep-dive — Per-regime routing analysis for MAC v3.

Existing 'routing_data.json' has per-PERIOD averages. This file extends
that to per-REGIME averages WITHIN each period — revealing how MAC's
controller actually responds to regime transitions.

For each (universe, period, regime), compute the mean routing weights
(def/bal/agg) on days where HMM identified that regime.

Output:
  results/per_regime_routing.json
  results/per_regime_routing.png
  results/per_regime_routing.md
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
REGIME_NAMES = {0: 'bull', 1: 'bear', 2: 'sideways'}


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
    weights = []
    regimes = []
    done = False
    while not done:
        step = env.master_env.current_step
        regime_id = int(env.master_env.regime_id_array[step])
        a, _ = model.predict(obs, deterministic=True)
        a_arr = np.array(a, dtype=np.float32)
        a_arr = a_arr - a_arr.max()
        w = np.exp(a_arr); w = w / w.sum()
        weights.append(w.tolist())
        regimes.append(regime_id)
        obs, _, term, trunc, _ = env.step(a)
        done = term or trunc
    return np.array(weights), np.array(regimes)


def main():
    results = {}
    print(f"\n  Computing per-regime routing for MAC v3 ...\n")
    for u in ['dji', 'sp100']:
        results[u] = {}
        for p, csv in PERIODS[u].items():
            print(f"  {u} | {p}")
            W, R = trace_mac(u, csv)
            results[u][p] = {'overall_avg': W.mean(axis=0).tolist(),
                              'n_days': int(len(R)),
                              'per_regime': {}}
            for rid in [0, 1, 2]:
                mask = R == rid
                n = int(mask.sum())
                if n == 0:
                    continue
                W_r = W[mask]
                avg = W_r.mean(axis=0)
                results[u][p]['per_regime'][REGIME_NAMES[rid]] = {
                    'avg_def': float(avg[0]),
                    'avg_bal': float(avg[1]),
                    'avg_agg': float(avg[2]),
                    'n_days': n,
                }

    with open('results/per_regime_routing.json','w') as f:
        json.dump(results, f, indent=2, default=float)

    # Markdown table
    md = ['# Per-regime MAC v3 routing (Deep-dive)\n',
          'Average routing weights on days HMM identified each regime.\n']
    for u in ['dji','sp100']:
        md.append(f'\n## {u.upper()}\n')
        md.append('| Period | Regime | days | def | bal | agg |')
        md.append('|---|---|---|---|---|---|')
        for p in ['val','test_bear','test_2023','test_bull']:
            pr = results[u][p]['per_regime']
            for regime in ['bull','bear','sideways']:
                if regime in pr:
                    r = pr[regime]
                    md.append(f"| {p} | {regime} | {r['n_days']} | "
                              f"{r['avg_def']:.2f} | {r['avg_bal']:.2f} | "
                              f"{r['avg_agg']:.2f} |")
    with open('results/per_regime_routing.md','w') as f:
        f.write('\n'.join(md))

    # PNG — grouped bars per universe per regime
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    regimes = ['bull','bear','sideways']
    REGIME_COLOR = {'bull':'#2ca02c', 'bear':'#d62728', 'sideways':'#ff7f0e'}
    for ux, u in enumerate(['dji','sp100']):
        ax = axes[ux]
        rows = []
        for p in ['val','test_bear','test_2023','test_bull']:
            for regime in regimes:
                pr = results[u][p]['per_regime'].get(regime)
                if pr is None:
                    continue
                rows.append({
                    'period': p, 'regime': regime,
                    'def': pr['avg_def'], 'bal': pr['avg_bal'], 'agg': pr['avg_agg'],
                    'n_days': pr['n_days'],
                })
        df = pd.DataFrame(rows)
        x_labels = [f"{r['period']}\n{r['regime']}\n(n={r['n_days']})" for _, r in df.iterrows()]
        x = np.arange(len(df))
        ax.bar(x, df['def'], color='#1f77b4', label='defender', alpha=0.85)
        ax.bar(x, df['bal'], bottom=df['def'], color='#2ca02c', label='steady', alpha=0.85)
        ax.bar(x, df['agg'], bottom=df['def']+df['bal'], color='#ff7f0e', label='hunter', alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(x_labels, fontsize=7)
        ax.set_ylabel('routing weight')
        ax.set_title(f'{u.upper()} — MAC v3 routing per (period, regime)', fontsize=11, weight='bold')
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
    plt.suptitle('MAC v3 routing decisions PER HMM REGIME within each test period',
                  fontsize=13, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('results/per_regime_routing.png', dpi=130, bbox_inches='tight')
    plt.close()

    # Console summary
    print('\n=== MAC v3 routing per regime (defender / steady / hunter weights) ===')
    for u in ['dji','sp100']:
        print(f"\n  {u.upper()}")
        for p in ['val','test_bear','test_2023','test_bull']:
            pr = results[u][p]['per_regime']
            print(f"  {p}:")
            for regime in ['bull','bear','sideways']:
                if regime in pr:
                    r = pr[regime]
                    print(f"    {regime:<9} (n={r['n_days']:>3} days)  "
                          f"def={r['avg_def']:.2f}  bal={r['avg_bal']:.2f}  agg={r['avg_agg']:.2f}")
    print('\n✅ saved results/per_regime_routing.json/md/png')


if __name__ == '__main__':
    main()
