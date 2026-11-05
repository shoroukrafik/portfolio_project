"""
Benchmark comparison: agents vs DJI index vs equal-weight portfolio.

For each period (val 2021, bear 2022, bull 2024) computes:
  - DJI buy-and-hold ($1M into the Dow Jones Industrial Average on day 1)
  - Equal-weight daily-rebalanced (1/30 in each stock, rebal every day)

Produces:
  results/ablation_ext_summary/benchmark_metrics.json
  results/ablation_ext_summary/benchmark_curves.png         — FULL agent vs DJI vs EW (3 equity-curve panels + metrics table)
  results/ablation_ext_summary/all_systems_vs_benchmarks.png — all 7 systems + DJI + EW (bar grid)
  results/ablation_ext_summary/benchmark_table.md           — markdown table for thesis
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from portfolio_env import PortfolioEnv


INITIAL_BALANCE = 1_000_000.0

PERIODS = [
    # (key, period CSV, DJI index CSV, label)
    ('val',       'val_data_ext.csv',       'val_index_ext.csv',       'Validation (2021)'),
    ('test_bear', 'test_bear_data_ext.csv', 'test_bear_index_ext.csv', 'Test — Bear (2022)'),
    ('test_2023', 'test_2023_data_ext.csv', 'test_2023_index_ext.csv', 'Test — Recovery (2023)'),
    ('test_bull', 'test_bull_data_ext.csv', 'test_bull_index_ext.csv', 'Test — Bull (2024)'),
]


def metrics_from_values(values, initial_balance=INITIAL_BALANCE):
    values = np.asarray(values, dtype=float)
    final = values[-1]
    cum_ret = (final - initial_balance) / initial_balance * 100
    daily_ret = np.diff(values) / values[:-1]
    n = len(values)
    ann_ret = ((final / initial_balance) ** (252 / n) - 1) * 100
    ann_vol = float(np.std(daily_ret) * np.sqrt(252) * 100)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    peak = np.maximum.accumulate(values)
    dd = (values - peak) / peak
    mdd = float(np.min(dd) * 100)
    return {
        'cumulative_return':    round(cum_ret, 2),
        'sharpe_ratio':         round(sharpe, 4),
        'max_drawdown':         round(mdd, 2),
        'annualized_volatility': round(ann_vol, 2),
    }


def dji_buy_and_hold(index_df, initial_balance=INITIAL_BALANCE):
    """$1M into DJI on day 1, follow daily returns from then on."""
    rets = index_df['daily_return'].fillna(0.0).values
    # Day 1 return = 0 (we just bought); compound from day 2
    values = np.empty(len(rets))
    values[0] = initial_balance
    for i in range(1, len(rets)):
        values[i] = values[i - 1] * (1.0 + rets[i])
    return values


def equal_weight_daily_rebal(period_df, initial_balance=INITIAL_BALANCE):
    """1/30 in each stock, rebalanced every day. Mean of stock returns."""
    # Average daily return across the 30 tickers, per date
    daily = (period_df.groupby('date')['daily_return']
                       .mean()
                       .sort_index())
    rets = daily.values
    values = np.empty(len(rets))
    values[0] = initial_balance
    for i in range(1, len(rets)):
        values[i] = values[i - 1] * (1.0 + rets[i])
    return values, daily.index.values


def evaluate_full_agent(model, env):
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(action)
        done = term or trunc
    h = env.get_portfolio_history()
    return h['value'].values, pd.to_datetime(h['date']).values


def load_existing_ablation_table(project_dir):
    """Reuse the table already produced by build_ablation_summary.py."""
    path = os.path.join(project_dir, 'results', 'ablation_ext_summary',
                        'ablation_table.csv')
    return pd.read_csv(path)


def main():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir    = os.path.join(project_dir, 'data')
    saved_dir   = os.path.join(project_dir, 'models', 'saved')
    out_dir     = os.path.join(project_dir, 'results', 'ablation_ext_summary')
    os.makedirs(out_dir, exist_ok=True)

    # ---- Load FULL agent ----
    print("\nLoading FULL baseline agent (ppo_best_extended.zip)...")
    full_model = PPO.load(os.path.join(saved_dir, 'ppo_best_extended'),
                          device='cpu')

    # ---- Per-period: compute benchmarks + agent equity curves ----
    curves = {}        # {period_key: {'dates','dji','ew','full'}}
    bench_metrics = {} # {period_key: {'DJI': {...}, 'EW': {...}}}
    for key, period_csv, dji_csv, label in PERIODS:
        print(f"\n[{label}]")
        # Period data
        period_df = pd.read_csv(os.path.join(data_dir, period_csv))
        period_df['date'] = pd.to_datetime(period_df['date'])
        # DJI series
        dji_df = pd.read_csv(os.path.join(data_dir, dji_csv))
        dji_df['date'] = pd.to_datetime(dji_df['date'])
        dji_df = dji_df.sort_values('date').reset_index(drop=True)
        # Compute benchmarks
        dji_vals = dji_buy_and_hold(dji_df)
        ew_vals, ew_dates = equal_weight_daily_rebal(period_df)
        # Compute FULL agent curve
        env = PortfolioEnv(df=period_df, initial_balance=INITIAL_BALANCE)
        full_vals, full_dates = evaluate_full_agent(full_model, env)

        # Align lengths if there's a 1-day offset (window_size warmup, etc.)
        # The agent env may skip the first window_size days. Truncate
        # benchmarks to match the agent curve length so they're comparable.
        L = min(len(full_vals), len(dji_vals), len(ew_vals))
        dji_vals_aligned  = dji_vals[-L:]  / dji_vals[-L] * INITIAL_BALANCE
        ew_vals_aligned   = ew_vals[-L:]   / ew_vals[-L]  * INITIAL_BALANCE
        full_vals_aligned = full_vals[-L:]

        curves[key] = {
            'dates': full_dates[-L:],
            'full':  full_vals_aligned,
            'dji':   dji_vals_aligned,
            'ew':    ew_vals_aligned,
        }

        m_dji = metrics_from_values(dji_vals_aligned)
        m_ew  = metrics_from_values(ew_vals_aligned)
        bench_metrics[key] = {'DJI': m_dji, 'EW': m_ew}
        print(f"  DJI buy&hold:        Ret={m_dji['cumulative_return']:>7.2f}%  "
              f"Sharpe={m_dji['sharpe_ratio']:>7.4f}  "
              f"MDD={m_dji['max_drawdown']:>7.2f}%")
        print(f"  Equal-weight (rebal): Ret={m_ew['cumulative_return']:>7.2f}%  "
              f"Sharpe={m_ew['sharpe_ratio']:>7.4f}  "
              f"MDD={m_ew['max_drawdown']:>7.2f}%")

    # ---- Save benchmark JSON ----
    json_path = os.path.join(out_dir, 'benchmark_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(bench_metrics, f, indent=2)
    print(f"\n  Wrote {json_path}")

    # ============================================================
    # PNG 1: FULL agent vs DJI vs EW  (equity curves + metric bars)
    # ============================================================
    n_periods = len(PERIODS)
    fig = plt.figure(figsize=(5.5 * n_periods, 10))
    gs = fig.add_gridspec(2, n_periods, height_ratios=[2.2, 1.0],
                          hspace=0.45, wspace=0.28)

    # Row 1: equity curves
    for col_i, (key, _, _, label) in enumerate(PERIODS):
        ax = fig.add_subplot(gs[0, col_i])
        c = curves[key]
        ax.plot(c['dates'], c['full'], label='FULL agent',  color='#2ca02c', linewidth=2.2)
        ax.plot(c['dates'], c['dji'],  label='DJI (buy & hold)', color='#1f77b4',
                linewidth=1.6, linestyle='-')
        ax.plot(c['dates'], c['ew'],   label='Equal-weight (rebal)', color='#d62728',
                linewidth=1.6, linestyle='--')
        ax.axhline(INITIAL_BALANCE, color='black', linewidth=0.6, alpha=0.5)
        ax.set_title(label, fontsize=11)
        ax.set_ylabel('Portfolio value ($)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
        ax.tick_params(axis='x', rotation=30)

    # Row 2: bar chart of cumulative return per period, 3 systems side-by-side
    ax_bar = fig.add_subplot(gs[1, :])
    labels = [p[3] for p in PERIODS]
    full_rets = [( curves[k]['full'][-1] / INITIAL_BALANCE - 1) * 100 for k,_,_,_ in PERIODS]
    dji_rets  = [bench_metrics[k]['DJI']['cumulative_return'] for k,_,_,_ in PERIODS]
    ew_rets   = [bench_metrics[k]['EW']['cumulative_return']  for k,_,_,_ in PERIODS]
    x = np.arange(len(labels))
    w = 0.27
    ax_bar.bar(x - w, full_rets, w, label='FULL agent',      color='#2ca02c',
               edgecolor='black', linewidth=0.4)
    ax_bar.bar(x,     dji_rets,  w, label='DJI (buy & hold)', color='#1f77b4',
               edgecolor='black', linewidth=0.4)
    ax_bar.bar(x + w, ew_rets,   w, label='Equal-weight',    color='#d62728',
               edgecolor='black', linewidth=0.4)
    for i in range(len(labels)):
        for off, v in zip([-w, 0, w], [full_rets[i], dji_rets[i], ew_rets[i]]):
            ax_bar.text(i + off, v + (0.3 if v >= 0 else -0.6),
                        f'{v:+.1f}%', ha='center',
                        va='bottom' if v >= 0 else 'top', fontsize=9)
    ax_bar.axhline(0, color='black', linewidth=0.6)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels)
    ax_bar.set_ylabel('Cumulative return (%)')
    ax_bar.set_title('Cumulative return — FULL agent vs benchmarks')
    ax_bar.grid(True, axis='y', alpha=0.3)
    ax_bar.legend(loc='best', fontsize=10)

    plt.suptitle('FULL extended baseline vs DJI buy-and-hold vs equal-weight portfolio',
                 fontsize=13, y=0.995)
    curves_path = os.path.join(out_dir, 'benchmark_curves.png')
    plt.savefig(curves_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Wrote {curves_path}")

    # ============================================================
    # PNG 2: ALL systems + DJI + EW (bar grid)
    # ============================================================
    ablations_df = load_existing_ablation_table(project_dir)
    # Append DJI and EW rows
    bench_rows = []
    for key, _, _, _ in PERIODS:
        for name in ['DJI', 'EW']:
            m = bench_metrics[key][name]
            bench_rows.append({
                'system': f'BENCHMARK: {name}',
                'period': key,
                'cumulative_return': m['cumulative_return'],
                'sharpe_ratio':      m['sharpe_ratio'],
                'max_drawdown':      m['max_drawdown'],
            })
    full_df = pd.concat([ablations_df, pd.DataFrame(bench_rows)], ignore_index=True)

    # Preserve a deterministic system order
    SYSTEM_ORDER = [
        'FULL (baseline)',
        'VANILLA',
        'no_floor (eval)',
        'no_dsr',
        'no_attention',
        'no_regime_λ=0.1',
        'no_regime_λ=0.3',
        'no_regime_λ=0.5',
        'BENCHMARK: DJI',
        'BENCHMARK: EW',
    ]
    full_df = full_df[full_df['system'].isin(SYSTEM_ORDER)]
    full_df['__order'] = full_df['system'].map({s: i for i, s in enumerate(SYSTEM_ORDER)})
    full_df = full_df.sort_values(['period', '__order']).drop(columns='__order')

    def color_for(label):
        if label == 'FULL (baseline)':       return '#2ca02c'
        if label == 'VANILLA':               return '#ff7f0e'
        if label == 'no_floor (eval)':       return '#d62728'
        if label.startswith('no_regime_λ'):  return '#1f77b4'
        if label.startswith('BENCHMARK'):    return '#9467bd'
        return '#7f7f7f'

    METRICS = [
        ('cumulative_return', 'Cumulative Return (%)'),
        ('sharpe_ratio',      'Sharpe Ratio'),
        ('max_drawdown',      'Max Drawdown (%)'),
    ]
    PERIOD_TITLES = {
        'val':       'Validation (2021)',
        'test_bear': 'Test — Bear (2022)',
        'test_2023': 'Test — Recovery (2023)',
        'test_bull': 'Test — Bull (2024)',
    }
    fig, axes = plt.subplots(4, 3, figsize=(20, 17))
    labels_present = [s for s in SYSTEM_ORDER if s in full_df['system'].values]
    x = np.arange(len(labels_present))
    for row_i, (key, _, _, _) in enumerate(PERIODS):
        for col_i, (metric, mlabel) in enumerate(METRICS):
            ax = axes[row_i, col_i]
            vals = []
            for lab in labels_present:
                row = full_df[(full_df['system'] == lab) &
                              (full_df['period'] == key)]
                vals.append(row.iloc[0][metric])
            colors = [color_for(lab) for lab in labels_present]
            bars = ax.bar(x, vals, color=colors, edgecolor='black', linewidth=0.5)
            # FULL reference line
            full_val = full_df[(full_df['system'] == 'FULL (baseline)') &
                               (full_df['period'] == key)].iloc[0][metric]
            ax.axhline(full_val, color='#2ca02c', linestyle='--',
                       linewidth=1, alpha=0.6, label='FULL reference')
            # DJI reference line (purple)
            dji_val = full_df[(full_df['system'] == 'BENCHMARK: DJI') &
                              (full_df['period'] == key)].iloc[0][metric]
            ax.axhline(dji_val, color='#9467bd', linestyle=':',
                       linewidth=1, alpha=0.7, label='DJI reference')
            for bar, v in zip(bars, vals):
                h = bar.get_height()
                span = max(vals) - min(vals) if max(vals) != min(vals) else 1.0
                ax.text(bar.get_x() + bar.get_width()/2,
                        h + (0.015 * span if h >= 0 else -0.04 * span),
                        f'{v:.2f}' if abs(v) >= 1 else f'{v:.3f}',
                        ha='center', va='bottom' if h >= 0 else 'top', fontsize=7)
            ax.set_xticks(x)
            ax.set_xticklabels(labels_present, rotation=40, ha='right', fontsize=7)
            ax.set_title(f'{PERIOD_TITLES[key]} — {mlabel}', fontsize=10)
            ax.axhline(0, color='black', linewidth=0.6)
            ax.grid(True, axis='y', alpha=0.3)
            if row_i == 0 and col_i == 0:
                ax.legend(loc='best', fontsize=7)
    plt.suptitle('All extended systems vs DJI buy-and-hold and equal-weight benchmark',
                 fontsize=13, y=1.00)
    plt.tight_layout()
    all_path = os.path.join(out_dir, 'all_systems_vs_benchmarks.png')
    plt.savefig(all_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Wrote {all_path}")

    # ============================================================
    # Markdown table including benchmarks
    # ============================================================
    md = ['# Extended systems vs benchmarks (DJI, equal-weight)\n']
    for key, _, _, label in PERIODS:
        md.append(f'\n## {label}\n')
        md.append('| System | Cumulative Return (%) | Sharpe | Max Drawdown (%) |')
        md.append('|---|---:|---:|---:|')
        for lab in labels_present:
            row = full_df[(full_df['system'] == lab) &
                          (full_df['period'] == key)].iloc[0]
            md.append(f"| {lab} | {row['cumulative_return']:.2f} | "
                      f"{row['sharpe_ratio']:.4f} | {row['max_drawdown']:.2f} |")
    md_path = os.path.join(out_dir, 'benchmark_table.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(md) + '\n')
    print(f"  Wrote {md_path}")

    # ============================================================
    # Console print
    # ============================================================
    print('\n' + '=' * 96)
    print(' SYSTEMS vs BENCHMARKS')
    print('=' * 96)
    for key, _, _, label in PERIODS:
        print(f"\n  {label}")
        print(f"  {'System':<24} {'Return %':>10} {'Sharpe':>10} {'MDD %':>10}")
        print(f"  {'-' * 58}")
        for lab in labels_present:
            row = full_df[(full_df['system'] == lab) &
                          (full_df['period'] == key)].iloc[0]
            print(f"  {lab:<24} {row['cumulative_return']:>10.2f} "
                  f"{row['sharpe_ratio']:>10.4f} {row['max_drawdown']:>10.2f}")
    print('\n' + '=' * 96 + '\n')


if __name__ == '__main__':
    main()
