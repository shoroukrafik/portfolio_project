"""
SP30 external-validity comparison chart.

Compares three PPO systems on the same 4 test periods, each against
its universe's natural buy-and-hold benchmark:

  1. DJI 30 FULL              (PPO, DJI universe, DJI-calibrated env)
  2. SP30 v1                  (PPO, SP30 universe, DJI-calibrated env)
  3. SP30 v2                  (PPO, SP30 universe, SP30-calibrated env)
  4. DJI buy-and-hold         (passive benchmark for DJI universe)
  5. S&P 500 buy-and-hold     (passive benchmark for SP30 universe)

Periods: val 2021, bear 2022, recovery 2023, bull 2024.

Story this chart tells:
  - SP30 v1: same PPO config moved to different stocks → underperforms
  - SP30 v2: SP30-calibrated env → tradeoff (better bull, worse bear)
  - Neither SP30 variant fully closes the gap vs DJI 30 FULL
  - But both still produce reasonable agents — design IDEAS transfer,
    universe-specific calibration is required for full performance

Outputs:
  results/sp30_comparison/comparison_table.csv
  results/sp30_comparison/comparison_table.md
  results/sp30_comparison/grid.png       — 4×3 bar grid (periods×metrics)
  results/sp30_comparison/deltas.png     — Δ vs DJI 30 per metric
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PERIODS = ['val', 'test_bear', 'test_2023', 'test_bull']
PERIOD_TITLES = {
    'val':       'Validation (2021)',
    'test_bear': 'Test — Bear (2022)',
    'test_2023': 'Test — Recovery (2023)',
    'test_bull': 'Test — Bull (2024)',
}
PERIOD_RANGES = {
    'val':       ('2021-01-01', '2021-12-31'),
    'test_bear': ('2022-01-01', '2022-12-31'),
    'test_2023': ('2023-01-01', '2023-12-31'),
    'test_bull': ('2024-01-01', '2024-12-31'),
}
METRICS = [
    ('cumulative_return', 'Cumulative Return (%)'),
    ('sharpe_ratio',      'Sharpe Ratio'),
    ('max_drawdown',      'Max Drawdown (%)'),
]


def metrics_from_index_slice(idx_df, start, end, initial_balance=1_000_000):
    """Buy-and-hold metrics for an index series between dates."""
    sub = idx_df[(idx_df['date'] >= start) & (idx_df['date'] <= end)].copy()
    sub = sub.sort_values('date').reset_index(drop=True)
    if len(sub) < 2:
        return None
    start_close = sub['close'].iloc[0]
    values = (sub['close'].values / start_close) * initial_balance
    final = values[-1]
    cum_ret = (final - initial_balance) / initial_balance * 100
    dr = np.diff(values) / values[:-1]
    n = len(values)
    ann_ret = ((final / initial_balance) ** (252 / n) - 1) * 100
    ann_vol = float(np.std(dr) * np.sqrt(252) * 100)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    peak = np.maximum.accumulate(values)
    mdd = float(np.min((values - peak) / peak) * 100)
    return {'cumulative_return': round(cum_ret, 2),
            'sharpe_ratio': round(sharpe, 4),
            'max_drawdown': round(mdd, 2)}


def color_for(label):
    if label.startswith('DJI 30 FULL'):  return '#2ca02c'    # green
    if label == 'SP30 v1':                return '#ff7f0e'    # orange
    if label == 'SP30 v2':                return '#d62728'    # red
    if label == 'DJI buy-and-hold':       return '#1f77b4'    # blue
    if label == 'S&P 500 buy-and-hold':   return '#9467bd'    # purple
    return '#7f7f7f'


def main():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_dir, 'data')
    out_dir  = os.path.join(project_dir, 'results', 'sp30_comparison')
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 78)
    print("   SP30 external-validity comparison")
    print("=" * 78)

    # ---- Load PPO results ----
    dji_results = json.load(open(os.path.join(project_dir, 'results',
                                              'extended', 'final_results.json')))
    sp30_v1_results = json.load(open(os.path.join(project_dir, 'results',
                                                   'extended_sp30', 'final_results.json')))
    sp30_v2_results = json.load(open(os.path.join(project_dir, 'results',
                                                   'extended_sp30_v2', 'final_results.json')))
    t2023_path = os.path.join(project_dir, 'results', 'test_2023_summary.json')
    dji_2023 = {}
    if os.path.exists(t2023_path):
        dji_2023 = json.load(open(t2023_path)).get('results', {}).get(
            'FULL (baseline)', {})

    # ---- Load index data for benchmarks ----
    dji_idx = pd.read_csv(os.path.join(data_dir, 'dji_index_ext.csv'))
    dji_idx['date'] = pd.to_datetime(dji_idx['date'])
    sp500_idx = pd.read_csv(os.path.join(data_dir, 'sp500_index_ext.csv'))
    sp500_idx['date'] = pd.to_datetime(sp500_idx['date'])

    # ---- Helpers to pull metrics ----
    def dji_metrics(period):
        if period == 'test_2023':
            return dji_2023 or None
        key = f'agent_{period}'
        return dji_results.get(key)

    def sp30_metrics(results, period):
        key = f'agent_{period}'
        return results.get(key)

    # ---- Build rows ----
    rows = []
    for period in PERIODS:
        start, end = PERIOD_RANGES[period]

        # PPO systems
        for label, m in [('DJI 30 FULL', dji_metrics(period)),
                         ('SP30 v1',     sp30_metrics(sp30_v1_results, period)),
                         ('SP30 v2',     sp30_metrics(sp30_v2_results, period))]:
            if m is None:
                continue
            rows.append({
                'system': label,
                'period': period,
                'cumulative_return': m['cumulative_return'],
                'sharpe_ratio':      m['sharpe_ratio'],
                'max_drawdown':      m['max_drawdown'],
            })

        # Benchmarks (buy-and-hold for each universe's index)
        for label, idx_df in [('DJI buy-and-hold',     dji_idx),
                              ('S&P 500 buy-and-hold', sp500_idx)]:
            bm = metrics_from_index_slice(idx_df, start, end)
            if bm:
                rows.append({
                    'system': label,
                    'period': period,
                    **bm,
                })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, 'comparison_table.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n  Wrote {csv_path}")

    # ---- Markdown table ----
    md = ['# SP30 external-validity comparison\n',
          '5 systems × 4 test periods. PPO systems compared against each '
          'universe\'s natural buy-and-hold benchmark.\n']
    for period in PERIODS:
        md.append(f'\n## {PERIOD_TITLES[period]}\n')
        md.append('| System | Return (%) | Sharpe | MDD (%) |')
        md.append('|---|---:|---:|---:|')
        sub = df[df['period'] == period]
        order = ['DJI 30 FULL', 'SP30 v1', 'SP30 v2',
                 'DJI buy-and-hold', 'S&P 500 buy-and-hold']
        for label in order:
            r = sub[sub['system'] == label]
            if r.empty:
                continue
            r = r.iloc[0]
            md.append(f"| {label} | {r['cumulative_return']:.2f} | "
                      f"{r['sharpe_ratio']:.4f} | {r['max_drawdown']:.2f} |")

    # Delta vs DJI 30 FULL (only PPO systems)
    md.append('\n## Δ vs DJI 30 FULL\n')
    md.append('| System | Period | ΔReturn (pp) | ΔSharpe | ΔMDD (pp) |')
    md.append('|---|---|---:|---:|---:|')
    base = df[df['system'] == 'DJI 30 FULL'].set_index('period')
    for label in ['SP30 v1', 'SP30 v2']:
        for period in PERIODS:
            r = df[(df['system'] == label) & (df['period'] == period)]
            if r.empty or period not in base.index:
                continue
            r = r.iloc[0]
            b = base.loc[period]
            d_ret = r['cumulative_return'] - b['cumulative_return']
            d_shp = r['sharpe_ratio']      - b['sharpe_ratio']
            d_mdd = r['max_drawdown']      - b['max_drawdown']
            md.append(f"| {label} | {period} | {d_ret:+.2f} | "
                      f"{d_shp:+.4f} | {d_mdd:+.2f} |")

    md_path = os.path.join(out_dir, 'comparison_table.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(md) + '\n')
    print(f"  Wrote {md_path}")

    # ---- 4×3 grid bar chart ----
    labels_present = ['DJI 30 FULL', 'SP30 v1', 'SP30 v2',
                      'DJI buy-and-hold', 'S&P 500 buy-and-hold']
    labels_present = [l for l in labels_present if l in df['system'].values]

    fig, axes = plt.subplots(4, 3, figsize=(18, 17))
    x = np.arange(len(labels_present))
    for row_i, period in enumerate(PERIODS):
        for col_i, (metric, mlabel) in enumerate(METRICS):
            ax = axes[row_i, col_i]
            vals = []
            for lab in labels_present:
                r = df[(df['system'] == lab) & (df['period'] == period)]
                vals.append(r.iloc[0][metric] if not r.empty else 0.0)
            colors = [color_for(l) for l in labels_present]
            bars = ax.bar(x, vals, color=colors, edgecolor='black', linewidth=0.5)
            # DJI 30 FULL reference line
            ref_row = df[(df['system'] == 'DJI 30 FULL') &
                         (df['period'] == period)]
            if not ref_row.empty:
                ax.axhline(ref_row.iloc[0][metric], color='#2ca02c',
                           linestyle='--', linewidth=1, alpha=0.6)
            for bar, v in zip(bars, vals):
                h = bar.get_height()
                span = (max(vals) - min(vals)) if max(vals) != min(vals) else 1.0
                ax.text(bar.get_x() + bar.get_width()/2,
                        h + (0.02 * span if h >= 0 else -0.04 * span),
                        f'{v:.2f}' if abs(v) >= 1 else f'{v:.3f}',
                        ha='center', va='bottom' if h >= 0 else 'top',
                        fontsize=8)
            ax.set_xticks(x)
            ax.set_xticklabels(labels_present, rotation=30, ha='right',
                               fontsize=8)
            ax.set_title(f'{PERIOD_TITLES[period]} — {mlabel}', fontsize=10)
            ax.axhline(0, color='black', linewidth=0.6)
            ax.grid(True, axis='y', alpha=0.3)
    plt.suptitle('SP30 external validity: PPO systems + buy-and-hold benchmarks '
                 '(green dashed = DJI 30 FULL reference)',
                 fontsize=13, y=1.00)
    plt.tight_layout()
    grid_path = os.path.join(out_dir, 'grid.png')
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Wrote {grid_path}")

    # ---- Δ vs DJI 30 plot (per metric, periods grouped) ----
    others = ['SP30 v1', 'SP30 v2']
    period_colors = {'val': '#4c72b0', 'test_bear': '#dd8452',
                     'test_2023': '#937860', 'test_bull': '#55a467'}
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(len(others))
    width = 0.20
    offsets = (np.arange(len(PERIODS)) - (len(PERIODS) - 1) / 2) * width
    for col_i, (metric, mlabel) in enumerate(METRICS):
        ax = axes[col_i]
        for i, period in enumerate(PERIODS):
            deltas = []
            for lab in others:
                r = df[(df['system'] == lab) & (df['period'] == period)]
                b = df[(df['system'] == 'DJI 30 FULL') &
                       (df['period'] == period)]
                if r.empty or b.empty:
                    deltas.append(0.0)
                else:
                    deltas.append(r.iloc[0][metric] - b.iloc[0][metric])
            ax.bar(x + offsets[i], deltas, width,
                   label=PERIOD_TITLES[period], color=period_colors[period],
                   edgecolor='black', linewidth=0.4)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(others, fontsize=10)
        ax.set_title(f'Δ {mlabel}  (system − DJI 30 FULL)')
        ax.grid(True, axis='y', alpha=0.3)
        if col_i == 0:
            ax.set_ylabel('Δ vs DJI 30 FULL  (positive = SP30 better)')
        ax.legend(fontsize=8, loc='best')
    plt.suptitle('SP30 v1 / v2 deltas vs DJI 30 FULL — positive bars = SP30 outperforms',
                 fontsize=12)
    plt.tight_layout()
    delta_path = os.path.join(out_dir, 'deltas.png')
    plt.savefig(delta_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Wrote {delta_path}")

    # ---- Console summary ----
    print('\n' + '=' * 90)
    print(' SP30 COMPARISON — all 5 systems × 4 periods')
    print('=' * 90)
    for period in PERIODS:
        print(f"\n  {PERIOD_TITLES[period]}")
        print(f"  {'System':<26} {'Return %':>10} {'Sharpe':>10} {'MDD %':>10}")
        print(f"  {'-' * 60}")
        for lab in labels_present:
            r = df[(df['system'] == lab) & (df['period'] == period)]
            if r.empty:
                continue
            r = r.iloc[0]
            print(f"  {lab:<26} {r['cumulative_return']:>10.2f} "
                  f"{r['sharpe_ratio']:>10.4f} {r['max_drawdown']:>10.2f}")
    print('\n' + '=' * 90 + '\n')


if __name__ == '__main__':
    main()
