"""
Multi-seed analysis — aggregates val-Sharpe-best checkpoint metrics across
seeds 42 (production), 41, 43 for both MAC v3 and universal v6 baselines on
DJI and SP100.

Loads each seed's best-by-val-Sharpe checkpoint, evaluates on all 4 periods,
reports mean ± SD across the 3 seeds.

Output:
  results/multi_seed_summary.json
  results/multi_seed_summary.md
  results/multi_seed_summary.png
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
from train_final_extended import calculate_metrics, evaluate_agent
from train_universal_v6_sp100 import V3_COLS_TO_DROP


SEEDS = [42, 41, 43]
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


def load_df(name):
    df = pd.read_csv(f'data/{name}')
    df['date'] = pd.to_datetime(df['date'])
    return df.drop(columns=[c for c in V3_COLS_TO_DROP if c in df.columns])


def mac_env_kwargs():
    return dict(
        initial_balance=1_000_000,
        cash_floors={0:0.0, 1:0.0, 2:0.0},
        risk_penalties={0:0.1, 1:0.5, 2:0.3},
        include_regime=True, include_regime_probs=True, include_vol_features=True,
    )


def v6_env_kwargs():
    return dict(
        initial_balance=1_000_000,
        cash_floors={0:0.0, 1:0.0, 2:0.0},
        risk_penalties={0:0.1, 1:0.5, 2:0.3},
        include_regime=True, include_regime_probs=True, include_vol_features=True,
        use_markowitz_reward=False, use_hyperheuristic=False,
    )


def eval_mac_v3(seed, universe):
    suffix = '' if seed == 42 else f'_s{seed}'
    model_path = f'models/saved/ppo_mac_{universe}_v3{suffix}.zip'
    specs = SPECIALISTS_DJI_V3 if universe == 'dji' else SPECIALISTS_SP100_V3
    periods = PERIODS_DJI if universe == 'dji' else PERIODS_SP
    model = PPO.load(model_path, device='cpu')
    out = {}
    for p, csv in periods.items():
        df = load_df(csv)
        env = MACMetaEnvV3(df, specs, mac_env_kwargs(), V3_TRAINING_FLOORS)
        obs, _ = env.reset()
        records = []
        done = False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = env.step(a)
            records.append(env.portfolio_value)
            done = term or trunc
        out[p] = calculate_metrics(pd.DataFrame({'value': records}))
    return out


def eval_v6(seed, universe):
    suffix = '' if seed == 42 else f'_s{seed}'
    model_path = f'models/saved/ppo_universal_v6_{universe}{suffix}.zip'
    periods = PERIODS_DJI if universe == 'dji' else PERIODS_SP
    model = PPO.load(model_path, device='cpu')
    out = {}
    for p, csv in periods.items():
        df = load_df(csv)
        env = PortfolioEnv(df=df, **v6_env_kwargs())
        history = evaluate_agent(model, env)
        out[p] = calculate_metrics(history)
    return out


def main():
    print(f"\n{'='*100}")
    print(f"   MULTI-SEED ANALYSIS — seeds {SEEDS}, val-Sharpe-best checkpoint per seed")
    print(f"{'='*100}\n")

    summary = {}
    for system, evaluator in [('MAC_v3', eval_mac_v3), ('universal_v6', eval_v6)]:
        for u in ['dji', 'sp100']:
            key = f'{system}|{u}'
            print(f"  Evaluating {key} ...")
            seed_results = {}
            for s in SEEDS:
                m = evaluator(s, u)
                seed_results[s] = {p: {'cum_ret': m[p]['cumulative_return'],
                                       'sharpe':  m[p]['sharpe_ratio'],
                                       'mdd':     m[p]['max_drawdown']}
                                   for p in m}
            summary[key] = seed_results

    os.makedirs('results', exist_ok=True)
    with open('results/multi_seed_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=float)

    # Aggregate mean ± SD per metric per period per system
    print(f"\n{'='*100}")
    print(f"  AGGREGATED RESULTS — mean ± SD across {len(SEEDS)} seeds")
    print(f"{'='*100}\n")
    rows = []
    for key, seed_results in summary.items():
        system, u = key.split('|')
        for p in ['val','test_bear','test_2023','test_bull']:
            cum  = [seed_results[s][p]['cum_ret'] for s in SEEDS]
            shp  = [seed_results[s][p]['sharpe']  for s in SEEDS]
            mdd  = [seed_results[s][p]['mdd']     for s in SEEDS]
            rows.append({
                'system': system, 'universe': u.upper(), 'period': p,
                'cum_mean': float(np.mean(cum)), 'cum_sd': float(np.std(cum, ddof=1)),
                'shp_mean': float(np.mean(shp)), 'shp_sd': float(np.std(shp, ddof=1)),
                'mdd_mean': float(np.mean(mdd)), 'mdd_sd': float(np.std(mdd, ddof=1)),
                'cum_per_seed': cum, 'shp_per_seed': shp,
            })

    # Print headline table
    print(f"  {'System':<14} {'Univ':<5} {'Period':<10}  {'Return %':<22}  {'Sharpe':<22}")
    print(f"  {'-'*14} {'-'*5} {'-'*10}  {'-'*22}  {'-'*22}")
    for r in rows:
        cum_str = f"{r['cum_mean']:>+7.2f} ± {r['cum_sd']:.2f}  [n={len(SEEDS)}]"
        shp_str = f"{r['shp_mean']:>+7.3f} ± {r['shp_sd']:.3f}"
        print(f"  {r['system']:<14} {r['universe']:<5} {r['period']:<10}  {cum_str:<22}  {shp_str:<22}")
        # Show per-seed
        seeds_str = '  '.join(f's{s}={c:+.2f}' for s, c in zip(SEEDS, r['cum_per_seed']))
        print(f"  {'':>30}  per-seed: {seeds_str}")

    # Markdown
    md_lines = ['# Multi-seed summary (3 seeds: 42, 41, 43)\n',
                'Each row: cumulative return mean ± SD (3 seeds) | Sharpe mean ± SD\n',
                '| System | Universe | Period | Return (mean ± SD) | Sharpe (mean ± SD) |',
                '|---|---|---|---|---|']
    for r in rows:
        md_lines.append(f"| {r['system']} | {r['universe']} | {r['period']} | "
                        f"{r['cum_mean']:+.2f} ± {r['cum_sd']:.2f} | "
                        f"{r['shp_mean']:+.3f} ± {r['shp_sd']:.3f} |")
    with open('results/multi_seed_summary.md', 'w') as f:
        f.write('\n'.join(md_lines))
    print(f"\n  ✅ → results/multi_seed_summary.json")
    print(f"  ✅ → results/multi_seed_summary.md")

    # PNG — Sharpe bars with seed-SD error bars, MAC vs v6, per universe
    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    periods = ['val','test_bear','test_2023','test_bull']
    for ax_idx, u in enumerate(['DJI','SP100']):
        ax = axes[ax_idx]
        x = np.arange(len(periods))
        w = 0.35
        for off, (system, color) in enumerate([('MAC_v3', '#1f77b4'),
                                                ('universal_v6', '#d62728')]):
            means = []
            sds   = []
            for p in periods:
                r = next(r for r in rows if r['system']==system and r['universe']==u and r['period']==p)
                means.append(r['shp_mean'])
                sds.append(r['shp_sd'])
            ax.bar(x + (off-0.5)*w, means, w, yerr=sds, label=system.replace('_',' '),
                   color=color, capsize=5, alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(periods)
        ax.set_ylabel('Sharpe ratio')
        ax.set_title(f'{u} — Sharpe across {len(SEEDS)} seeds (mean ± SD)')
        ax.axhline(0, color='gray', lw=0.5)
        ax.grid(axis='y', alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.savefig('results/multi_seed_summary.png', dpi=130, bbox_inches='tight')
    print(f"  ✅ → results/multi_seed_summary.png")
    plt.close()


if __name__ == '__main__':
    main()
