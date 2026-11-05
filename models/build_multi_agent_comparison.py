"""
Multi-agent architectures vs FULL — a separate comparison.

Keeps the multi-agent variants segregated from the component-ablation
summary so the two stories are visually distinct in the thesis:

  Component-ablation summary:  FULL + vanilla + ablations + DJI + EW
                               (build_ablation_summary.py / benchmark_comparison.py)

  Multi-agent comparison:      FULL + MTL v1 + MTL v2 + Multi-agent
                               specialists (soft) + Risk-personality (soft)
                               (this script)

Each variant runs across 4 periods: val 2021, bear 2022, recovery 2023,
bull 2024.  2023 metrics are loaded from results/test_2023_summary.json.

Outputs:
  results/multi_agent_comparison/comparison_table.csv
  results/multi_agent_comparison/comparison_table.md
  results/multi_agent_comparison/grid.png      — 4×3 bar grid (periods×metrics)
  results/multi_agent_comparison/deltas.png    — Δ vs FULL per metric
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# (label, primary-file relative path, shape, key-in-test_2023_summary)
# NOTE: MTL v1 is intentionally excluded from this main comparison —
# its saved checkpoint was a known val-spike artifact (val Sharpe 3.47
# while training reward was crashing, classic overfit signature).
# v1 is discussed in the methodology section of the writeup as an
# iteration step, not reported as an architectural result. The fair
# MTL result is v2 (deep heads + diversity loss, no aux).
SYSTEMS = [
    ('FULL (baseline)',          'results/extended/final_results.json',                       'agent',   'FULL (baseline)'),
    ('MTL v2',                   'results/mtl_v2_extended/final_results.json',                'agent',   'MTL v2'),
    ('Multi-agent (soft)',       'results/multi_agent_ext/final_results_soft.json',           'periods_metrics', 'Multi-agent (soft)'),
    ('Risk-personality (soft)',  'results/risk_personality_ext/final_results_soft.json',      'periods_metrics', 'Risk-personality (soft)'),
]

PERIODS = ['val', 'test_bear', 'test_2023', 'test_bull']
PERIOD_TITLES = {
    'val':       'Validation (2021)',
    'test_bear': 'Test — Bear (2022)',
    'test_2023': 'Test — Recovery (2023)',
    'test_bull': 'Test — Bull (2024)',
}
METRICS = [
    ('cumulative_return', 'Cumulative Return (%)'),
    ('sharpe_ratio',      'Sharpe Ratio'),
    ('max_drawdown',      'Max Drawdown (%)'),
]


def load_period(path, shape, period, label, t2023_lookup, t2023_key):
    # 2023 lives in a consolidated file keyed by system label.
    if period == 'test_2023':
        if t2023_key is None or t2023_key not in t2023_lookup:
            return None
        return t2023_lookup[t2023_key]
    with open(path) as f:
        d = json.load(f)
    if shape == 'agent':
        return d.get(f'agent_{period}')
    if shape == 'periods_metrics':
        # multi_agent / risk_personality save under 'periods.<period>.metrics'
        per = d.get('periods', {}).get(period, {})
        return per.get('metrics', per)
    if shape == 'periods':
        return d.get('periods', {}).get(period)
    if shape == 'flat':
        return d.get(period)
    return None


def color_for(label):
    if label == 'FULL (baseline)':            return '#2ca02c'    # green
    if label == 'MTL v2':                     return '#d62728'    # red
    if label == 'Multi-agent (soft)':         return '#1f77b4'    # blue
    if label == 'Risk-personality (soft)':    return '#9467bd'    # purple
    return '#7f7f7f'


def main():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(project_dir, 'results', 'multi_agent_comparison')
    os.makedirs(out_dir, exist_ok=True)

    # ---- Load consolidated 2023 metrics ----
    t2023_lookup = {}
    t2023_path = os.path.join(project_dir, 'results', 'test_2023_summary.json')
    if os.path.exists(t2023_path):
        t2023_lookup = json.load(open(t2023_path)).get('results', {})

    # ---- Flatten into a long dataframe ----
    rows = []
    for label, rel_path, shape, t2023_key in SYSTEMS:
        full_path = os.path.join(project_dir, rel_path)
        if not os.path.exists(full_path):
            print(f"  MISSING: {rel_path}")
            continue
        for period in PERIODS:
            m = load_period(full_path, shape, period, label,
                            t2023_lookup, t2023_key)
            if m is None:
                rows.append({'system': label, 'period': period,
                             'cumulative_return': np.nan,
                             'sharpe_ratio': np.nan,
                             'max_drawdown': np.nan})
                continue
            rows.append({
                'system': label,
                'period': period,
                'cumulative_return': m['cumulative_return'],
                'sharpe_ratio':      m['sharpe_ratio'],
                'max_drawdown':      m['max_drawdown'],
            })

    df = pd.DataFrame(rows)

    # ---- CSV ----
    csv_path = os.path.join(out_dir, 'comparison_table.csv')
    df.to_csv(csv_path, index=False)
    print(f"  Wrote {csv_path}")

    # ---- Markdown table ----
    md = ['# Multi-agent architectures vs FULL\n',
          'Separate from the component-ablation comparison. Each row is a '
          'multi-agent design tested against FULL on four out-of-sample '
          'periods.\n']
    for period in PERIODS:
        md.append(f'\n## {PERIOD_TITLES[period]}\n')
        md.append('| System | Return (%) | Sharpe | MDD (%) |')
        md.append('|---|---:|---:|---:|')
        sub = df[df['period'] == period]
        for _, r in sub.iterrows():
            if pd.isna(r['cumulative_return']):
                md.append(f"| {r['system']} | n/a | n/a | n/a |")
            else:
                md.append(f"| {r['system']} | {r['cumulative_return']:.2f} | "
                          f"{r['sharpe_ratio']:.4f} | {r['max_drawdown']:.2f} |")

    # Δ vs FULL block
    md.append('\n## Δ vs FULL (system − FULL)\n')
    md.append('Positive Δ-return / Δ-Sharpe = system better. '
              'Positive Δ-MDD = system has less-negative drawdown (better).\n')
    md.append('| System | Period | ΔReturn (pp) | ΔSharpe | ΔMDD (pp) |')
    md.append('|---|---|---:|---:|---:|')
    full = df[df['system'] == 'FULL (baseline)'].set_index('period')
    for label, _, _, _ in SYSTEMS:
        if label == 'FULL (baseline)':
            continue
        for period in PERIODS:
            row = df[(df['system'] == label) &
                     (df['period'] == period)]
            if row.empty or pd.isna(row.iloc[0]['cumulative_return']):
                md.append(f"| {label} | {period} | n/a | n/a | n/a |")
                continue
            r = row.iloc[0]
            if period not in full.index or pd.isna(full.loc[period,
                                                            'cumulative_return']):
                md.append(f"| {label} | {period} | n/a | n/a | n/a |")
                continue
            d_ret = r['cumulative_return'] - full.loc[period, 'cumulative_return']
            d_shp = r['sharpe_ratio']      - full.loc[period, 'sharpe_ratio']
            d_mdd = r['max_drawdown']      - full.loc[period, 'max_drawdown']
            md.append(f"| {label} | {period} | {d_ret:+.2f} | "
                      f"{d_shp:+.4f} | {d_mdd:+.2f} |")

    md_path = os.path.join(out_dir, 'comparison_table.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(md) + '\n')
    print(f"  Wrote {md_path}")

    # ---- 4×3 grid of bar charts ----
    labels_present = [s[0] for s in SYSTEMS
                      if s[0] in df['system'].values
                      and not df[df['system'] == s[0]]['cumulative_return'].isna().all()]
    fig, axes = plt.subplots(4, 3, figsize=(16, 16))
    x = np.arange(len(labels_present))
    for row_i, period in enumerate(PERIODS):
        for col_i, (metric, metric_label) in enumerate(METRICS):
            ax = axes[row_i, col_i]
            vals = []
            for lab in labels_present:
                v = df[(df['system'] == lab) & (df['period'] == period)][metric].iloc[0]
                vals.append(v if not pd.isna(v) else 0.0)
            colors = [color_for(lab) for lab in labels_present]
            bars = ax.bar(x, vals, color=colors, edgecolor='black', linewidth=0.5)
            # FULL reference line
            full_val = df[(df['system'] == 'FULL (baseline)') &
                          (df['period'] == period)][metric]
            if not full_val.empty and not pd.isna(full_val.iloc[0]):
                ax.axhline(full_val.iloc[0], color='#2ca02c', linestyle='--',
                           linewidth=1, alpha=0.6)
            for bar, v in zip(bars, vals):
                h = bar.get_height()
                span = (max(vals) - min(vals)) if max(vals) != min(vals) else 1.0
                ax.text(bar.get_x() + bar.get_width()/2,
                        h + (0.02 * span if h >= 0 else -0.04 * span),
                        f'{v:.2f}' if abs(v) >= 1 else f'{v:.3f}',
                        ha='center', va='bottom' if h >= 0 else 'top', fontsize=8)
            ax.set_xticks(x)
            ax.set_xticklabels(labels_present, rotation=35, ha='right', fontsize=8)
            ax.set_title(f'{PERIOD_TITLES[period]} — {metric_label}', fontsize=10)
            ax.grid(True, axis='y', alpha=0.3)
            ax.axhline(0, color='black', linewidth=0.6)
    plt.suptitle('Multi-agent architectures vs FULL (green dashed = FULL reference)',
                 fontsize=13, y=1.00)
    plt.tight_layout()
    grid_path = os.path.join(out_dir, 'grid.png')
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Wrote {grid_path}")

    # ---- Δ vs FULL plot (one panel per metric, periods grouped) ----
    multi_agents = [lab for lab in labels_present if lab != 'FULL (baseline)']
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    period_colors = {'val': '#4c72b0', 'test_bear': '#dd8452',
                     'test_2023': '#937860', 'test_bull': '#55a467'}
    x = np.arange(len(multi_agents))
    width = 0.20
    n_periods = len(PERIODS)
    offsets = (np.arange(n_periods) - (n_periods - 1) / 2) * width
    for col_i, (metric, metric_label) in enumerate(METRICS):
        ax = axes[col_i]
        for i, period in enumerate(PERIODS):
            deltas = []
            for lab in multi_agents:
                row = df[(df['system'] == lab) & (df['period'] == period)]
                base_row = df[(df['system'] == 'FULL (baseline)') &
                              (df['period'] == period)]
                if (row.empty or base_row.empty
                    or pd.isna(row.iloc[0][metric])
                    or pd.isna(base_row.iloc[0][metric])):
                    deltas.append(0.0)
                else:
                    deltas.append(row.iloc[0][metric] - base_row.iloc[0][metric])
            ax.bar(x + offsets[i], deltas, width,
                   label=PERIOD_TITLES[period], color=period_colors[period],
                   edgecolor='black', linewidth=0.4)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(multi_agents, rotation=25, ha='right', fontsize=9)
        ax.set_title(f'Δ {metric_label}  (system − FULL)')
        ax.grid(True, axis='y', alpha=0.3)
        if col_i == 0:
            ax.set_ylabel('Δ vs FULL  (positive = system better)')
        ax.legend(fontsize=8, loc='best')
    plt.suptitle('Multi-agent variants vs FULL — positive bars mean the '
                 'variant beats FULL on that metric+period',
                 fontsize=12)
    plt.tight_layout()
    delta_path = os.path.join(out_dir, 'deltas.png')
    plt.savefig(delta_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Wrote {delta_path}")

    # ---- Console print ----
    print('\n' + '=' * 90)
    print(' MULTI-AGENT COMPARISON — all systems × all periods')
    print('=' * 90)
    for period in PERIODS:
        print(f"\n  {PERIOD_TITLES[period]}")
        print(f"  {'System':<28} {'Return %':>10} {'Sharpe':>10} {'MDD %':>10}")
        print(f"  {'-' * 64}")
        for _, r in df[df['period'] == period].iterrows():
            if pd.isna(r['cumulative_return']):
                print(f"  {r['system']:<28} {'n/a':>10} {'n/a':>10} {'n/a':>10}")
            else:
                print(f"  {r['system']:<28} {r['cumulative_return']:>10.2f} "
                      f"{r['sharpe_ratio']:>10.4f} {r['max_drawdown']:>10.2f}")
    print('\n' + '=' * 90)


if __name__ == '__main__':
    main()
