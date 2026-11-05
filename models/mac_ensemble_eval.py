"""
MAC ensemble inference — averages the routing actions across 3 MAC v3 seeds at
each timestep. Tests whether ensemble produces tighter / better performance
than any single seed.

Method:
  At each timestep, each of the 3 MAC seeds predicts an action (3 routing logits).
  We average the logits across seeds, then proceed as normal (softmax → blend
  specialists → execute in master env).

Compared against:
  - Each individual MAC seed (42, 41, 43)
  - Their mean ± SD (from multi_seed_summary)

Output:
  results/mac_ensemble.json
  results/mac_ensemble.md
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from stable_baselines3 import PPO
from mac_meta_env_v3 import (
    MACMetaEnvV3, SPECIALISTS_DJI_V3, SPECIALISTS_SP100_V3,
    V3_TRAINING_FLOORS,
)
from train_final_extended import calculate_metrics
from train_universal_v6_sp100 import V3_COLS_TO_DROP


SEEDS = [42, 41, 43]
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


def ensemble_eval(universe, period_csv):
    """Run the 3-seed ensemble. Each seed's MAC predicts action;
       average those actions; step the master env with averaged action."""
    df = load_df(period_csv)
    specs = SPECIALISTS_DJI_V3 if universe == 'dji' else SPECIALISTS_SP100_V3

    # Load all 3 MAC v3 seed models
    models = []
    for s in SEEDS:
        suffix = '' if s == 42 else f'_s{s}'
        path = f'models/saved/ppo_mac_{universe}_v3{suffix}.zip'
        models.append(PPO.load(path, device='cpu'))

    env = MACMetaEnvV3(df, specs, env_kwargs(), V3_TRAINING_FLOORS)
    obs, _ = env.reset()
    records = []
    done = False
    while not done:
        # Each seed predicts a 3-vector of routing logits
        actions = []
        for m in models:
            a, _ = m.predict(obs, deterministic=True)
            actions.append(np.array(a, dtype=np.float32))
        # Average logits across seeds → ensemble action
        avg_action = np.mean(np.stack(actions), axis=0)
        obs, _, term, trunc, _ = env.step(avg_action)
        records.append(env.portfolio_value)
        done = term or trunc
    return calculate_metrics(pd.DataFrame({'value': records}))


def main():
    print(f"\n{'='*100}")
    print(f"  MAC v3 ENSEMBLE INFERENCE — averaging actions across {len(SEEDS)} seeds")
    print(f"{'='*100}\n")

    # Load existing multi-seed summary for comparison
    with open('results/multi_seed_summary.json') as f:
        ms = json.load(f)

    results = {}
    for u in ['dji','sp100']:
        results[u] = {}
        print(f"  {u.upper()} ensemble:")
        for p, csv in PERIODS[u].items():
            m = ensemble_eval(u, csv)
            results[u][p] = {
                'cum_ret': m['cumulative_return'],
                'sharpe':  m['sharpe_ratio'],
                'mdd':     m['max_drawdown'],
            }
            print(f"    {p:<11} cum={m['cumulative_return']:>+7.2f}%  "
                  f"sharpe={m['sharpe_ratio']:>+.3f}  mdd={m['max_drawdown']:>+.2f}%")
        print()

    # Comparison table
    print(f"{'='*100}")
    print(f"  ENSEMBLE vs PER-SEED MAC v3 vs MULTI-SEED MEAN")
    print(f"{'='*100}")
    md_lines = [
        '# MAC v3 ensemble inference results\n',
        'Ensemble averages routing logits across 3 MAC seeds (42, 41, 43) before softmax + blend.\n',
        '| Universe | Period | Ensemble | Mean ± SD (3 seeds) | Best single seed | Δ Ensemble−Mean |',
        '|---|---|---|---|---|---|',
    ]
    for u in ['dji','sp100']:
        print(f"\n  {u.upper()}:")
        print(f"  {'Period':<11} {'Ensemble':>11} {'Mean (3s)':>12} {'SD':>7} {'Best single':>13} {'Δ Ens-Mean':>12}")
        for p in ['val','test_bear','test_2023','test_bull']:
            ens_ret = results[u][p]['cum_ret']
            per_seed = [ms[f'MAC_v3|{u}'][str(s)][p]['cum_ret'] for s in SEEDS]
            mean = np.mean(per_seed); sd = np.std(per_seed, ddof=1)
            best_single = max(per_seed)
            delta = ens_ret - mean
            print(f"  {p:<11} {ens_ret:>+10.2f}% {mean:>+11.2f}% {sd:>6.2f} "
                  f"{best_single:>+12.2f}% {delta:>+11.2f}")
            md_lines.append(f"| {u.upper()} | {p} | {ens_ret:+.2f} | "
                            f"{mean:+.2f} ± {sd:.2f} | {best_single:+.2f} | {delta:+.2f} |")

    os.makedirs('results', exist_ok=True)
    with open('results/mac_ensemble.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)
    with open('results/mac_ensemble.md', 'w') as f:
        f.write('\n'.join(md_lines))
    print(f"\n  ✅ → results/mac_ensemble.json")
    print(f"  ✅ → results/mac_ensemble.md")


if __name__ == '__main__':
    main()
