"""
Deep-dive — Sortino + Calmar ratios per system.

Sortino  = mean_return / downside_vol  (asymmetric, aligns with our DSR+downside reward)
Calmar   = annualized_return / |max_drawdown|  (tail-risk-aware, matches thesis criterion)

Computed for:
  defender / steady / hunter (specialist-alone, hard-routed through master env)
  MAC v3 (full production)
  v6 universal (single agent)

On all 4 periods × 2 universes.

Outputs:
  results/sortino_calmar.json
  results/sortino_calmar.md
  results/sortino_calmar.png
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
ANN = 252  # trading days


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
    return dict(**env_kwargs(), use_markowitz_reward=False, use_hyperheuristic=False)


def specialist_alone(universe, period_csv, role):
    df = load_df(period_csv)
    specs = SPECIALISTS_DJI_V3 if universe == 'dji' else SPECIALISTS_SP100_V3
    env = MACMetaEnvV3(df, specs, env_kwargs(), V3_TRAINING_FLOORS)
    logits = np.full(3, -5.0, dtype=np.float32)
    logits[SPEC_IDX[role]] = 5.0
    obs, _ = env.reset()
    values = [env.portfolio_value]
    done = False
    while not done:
        obs, _, term, trunc, _ = env.step(logits)
        values.append(env.portfolio_value)
        done = term or trunc
    return values


def mac_alone(universe, period_csv):
    df = load_df(period_csv)
    specs = SPECIALISTS_DJI_V3 if universe == 'dji' else SPECIALISTS_SP100_V3
    env = MACMetaEnvV3(df, specs, env_kwargs(), V3_TRAINING_FLOORS)
    model = PPO.load(f'models/saved/ppo_mac_{universe}_v3.zip', device='cpu')
    obs, _ = env.reset()
    values = [env.portfolio_value]
    done = False
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(a)
        values.append(env.portfolio_value)
        done = term or trunc
    return values


def v6_alone(universe, period_csv):
    df = load_df(period_csv)
    env = PortfolioEnv(df=df, **v6_env_kwargs())
    model = PPO.load(f'models/saved/ppo_universal_v6_{universe}.zip', device='cpu')
    obs, _ = env.reset()
    values = [env.portfolio_value]
    done = False
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(a)
        values.append(env.portfolio_value)
        done = term or trunc
    return values


def metrics(values):
    v = np.asarray(values, dtype=float)
    r = v[1:] / v[:-1] - 1.0
    cum_ret = (v[-1] / v[0] - 1.0) * 100.0
    # Sharpe (for reference)
    sharpe = (r.mean() / r.std()) * np.sqrt(ANN) if r.std() > 0 else 0.0
    # Sortino: downside vol only (negative returns)
    neg = r[r < 0]
    downside_vol = neg.std() if len(neg) > 1 else r.std()
    sortino = (r.mean() / downside_vol) * np.sqrt(ANN) if downside_vol > 0 else 0.0
    # Calmar: annualized return / |MDD|
    peak = np.maximum.accumulate(v)
    dd = v / peak - 1.0
    mdd = float(dd.min()) * 100  # percent
    n_days = len(r)
    ann_ret = ((v[-1] / v[0]) ** (ANN / n_days) - 1) * 100 if n_days > 0 else 0
    calmar = ann_ret / abs(mdd) if abs(mdd) > 0.01 else 0.0
    return {
        'cum_ret': cum_ret, 'sharpe': sharpe, 'sortino': sortino,
        'calmar': calmar, 'mdd': mdd, 'ann_ret': ann_ret,
    }


def main():
    SYSTEMS = [
        ('defender', specialist_alone, {'role': 'defensive'}),
        ('steady',   specialist_alone, {'role': 'balanced'}),
        ('hunter',   specialist_alone, {'role': 'aggressive'}),
        ('MAC v3',   mac_alone, {}),
        ('v6 univ',  v6_alone,  {}),
    ]
    results = {}
    print(f"\n  Computing Sortino + Calmar across {sum(len(p) for p in PERIODS.values())} period-cells × {len(SYSTEMS)} systems...\n")
    for u in ['dji','sp100']:
        results[u] = {}
        for p, csv in PERIODS[u].items():
            results[u][p] = {}
            for label, fn, extra in SYSTEMS:
                v = fn(u, csv, **extra)
                results[u][p][label] = metrics(v)
    os.makedirs('results', exist_ok=True)
    with open('results/sortino_calmar.json','w') as f:
        json.dump(results, f, indent=2, default=float)

    # Markdown table
    md = ['# Sortino + Calmar deep-dive\n',
          'Sortino = mean_return / downside_vol × √252  (asymmetric: penalizes only downside)',
          'Calmar  = annualized_return / |MDD|  (tail-risk-aware)\n']
    for u in ['dji','sp100']:
        md.append(f'\n## {u.upper()}\n')
        md.append('| Period | System | Cum.Ret% | Sharpe | Sortino | Calmar | MDD% |')
        md.append('|---|---|---|---|---|---|---|')
        for p in ['val','test_bear','test_2023','test_bull']:
            for sys_name in ['defender','steady','hunter','MAC v3','v6 univ']:
                m = results[u][p][sys_name]
                md.append(f"| {p} | {sys_name} | {m['cum_ret']:+.2f} | "
                          f"{m['sharpe']:+.3f} | {m['sortino']:+.3f} | "
                          f"{m['calmar']:+.2f} | {m['mdd']:+.2f} |")
    with open('results/sortino_calmar.md','w') as f:
        f.write('\n'.join(md))

    # PNG — 3-panel (Sharpe, Sortino, Calmar) per universe per period
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    periods = ['val','test_bear','test_2023','test_bull']
    sys_names = ['defender','steady','hunter','MAC v3','v6 univ']
    colors = ['#1f77b4','#2ca02c','#ff7f0e','#9467bd','#d62728']
    for ux, u in enumerate(['dji','sp100']):
        for mx, (mkey, mtitle) in enumerate([('sharpe','Sharpe'), ('sortino','Sortino'), ('calmar','Calmar')]):
            ax = axes[ux, mx]
            x = np.arange(len(periods))
            w = 0.16
            for k, (sys_name, color) in enumerate(zip(sys_names, colors)):
                vals = [results[u][p][sys_name][mkey] for p in periods]
                ax.bar(x + (k - 2) * w, vals, w, label=sys_name, color=color, alpha=0.85)
            ax.set_xticks(x); ax.set_xticklabels(periods, fontsize=9)
            ax.set_ylabel(mtitle)
            ax.set_title(f'{u.upper()} — {mtitle}', fontsize=11, weight='bold')
            ax.axhline(0, color='gray', lw=0.5)
            ax.grid(axis='y', alpha=0.3)
            if ux == 0 and mx == 0:
                ax.legend(fontsize=8, loc='upper left')
    plt.suptitle('Risk-adjusted return metrics — Sharpe / Sortino / Calmar across systems and periods',
                  fontsize=13, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('results/sortino_calmar.png', dpi=130, bbox_inches='tight')
    plt.close()

    # Console summary — focus on key bear period
    print('\n=== BEAR PERIOD risk-adjusted metrics (most relevant for thesis) ===')
    for u in ['dji','sp100']:
        print(f"\n  {u.upper()}")
        print(f"  {'System':<10}{'Sharpe':>9}{'Sortino':>10}{'Calmar':>9}{'MDD%':>9}")
        for sys_name in sys_names:
            m = results[u]['test_bear'][sys_name]
            print(f"  {sys_name:<10}{m['sharpe']:>+9.3f}{m['sortino']:>+10.3f}{m['calmar']:>+9.2f}{m['mdd']:>+9.2f}")
    print('\n✅ saved results/sortino_calmar.json/md/png')


if __name__ == '__main__':
    main()
