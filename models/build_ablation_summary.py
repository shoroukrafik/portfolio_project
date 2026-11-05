"""
Consolidated extended-baseline ablation summary.

Reads the seven ablation result JSONs and produces:
  results/ablation_ext_summary/ablation_table.csv      — full comparison table
  results/ablation_ext_summary/ablation_table.md       — markdown table for thesis
  results/ablation_ext_summary/ablation_summary.json   — machine-readable bundle
  results/ablation_ext_summary/ablation_grid.png       — 3 (periods) x 3 (metrics) bar grid
  results/ablation_ext_summary/ablation_deltas.png     — Δ vs FULL baseline, one panel per metric

FULL baseline is highlighted; ablations are gray. A dashed line in each
panel marks the FULL baseline value to make the comparison instant.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---- Config ---------------------------------------------------------------

# (label-for-table, json-path-relative-to-project-root, source-shape)
#   source-shape values:
#     'agent'   → keys like agent_val / agent_test_bear / agent_test_bull
#     'flat'    → keys val / test_bear / test_bull directly
#     'periods' → keys under top-level 'periods' dict
SYSTEMS = [
    ('FULL (baseline)',     'results/extended/final_results.json',                          'agent'),
    ('VANILLA',             'results/vanilla_extended/final_results.json',                  'periods'),
    ('no_floor (eval)',     'results/extended/final_results_no_floor.json',                 'flat'),
    ('no_dsr',              'results/ablation_ext_no_dsr/final_results.json',               'periods'),
    ('no_attention',        'results/ablation_ext_no_attention/final_results.json',         'periods'),
    ('no_regime_λ=0.1',     'results/ablation_ext_no_regime_lambda_0p1/final_results.json', 'periods'),
    ('no_regime_λ=0.3',     'results/ablation_ext_no_regime_lambda_0p3/final_results.json', 'periods'),
    ('no_regime_λ=0.5',     'results/ablation_ext_no_regime_lambda_0p5/final_results.json', 'periods'),
]

PERIODS = ['val', 'test_bear', 'test_2023', 'test_bull']
PERIOD_TITLES = {
    'val':       'Validation (2021)',
    'test_bear': 'Test — Bear (2022)',
    'test_2023': 'Test — Recovery (2023)',
    'test_bull': 'Test — Bull (2024)',
}
METRICS = [
    ('cumulative_return', 'Cumulative Return (%)', True),    # higher is better
    ('sharpe_ratio',      'Sharpe Ratio',          True),
    ('max_drawdown',      'Max Drawdown (%)',      False),   # less negative is better
]


def load_period_metrics(path, shape, period, label=None, t2023_lookup=None):
    # 2023 was added later (after the per-system JSONs were written), so it
    # lives in a consolidated file keyed by system label. Always read 2023
    # from there.
    if period == 'test_2023':
        if t2023_lookup is None or label not in t2023_lookup:
            raise KeyError(f"No 2023 data for label '{label}'")
        return t2023_lookup[label]
    with open(path) as f:
        d = json.load(f)
    if shape == 'agent':
        return d[f'agent_{period}']
    if shape == 'flat':
        return d[period]
    if shape == 'periods':
        return d['periods'][period]
    raise ValueError(shape)


def main():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(project_dir, 'results', 'ablation_ext_summary')
    os.makedirs(out_dir, exist_ok=True)

    # ---- Load the consolidated 2023 lookup (one file keyed by system) ----
    t2023_path = os.path.join(project_dir, 'results', 'test_2023_summary.json')
    t2023_lookup = {}
    if os.path.exists(t2023_path):
        bundle = json.load(open(t2023_path))
        t2023_lookup = bundle.get('results', {})

    # ---- Load all systems into a flat dataframe ----
    rows = []
    raw_bundle = {}
    for label, rel_path, shape in SYSTEMS:
        full_path = os.path.join(project_dir, rel_path)
        if not os.path.exists(full_path):
            print(f"  MISSING: {rel_path}")
            continue
        sys_dict = {}
        for period in PERIODS:
            try:
                m = load_period_metrics(full_path, shape, period,
                                        label=label, t2023_lookup=t2023_lookup)
            except KeyError as e:
                print(f"  SKIP {label} / {period}: {e}")
                continue
            sys_dict[period] = m
            rows.append({
                'system': label,
                'period': period,
                'cumulative_return': m['cumulative_return'],
                'sharpe_ratio':      m['sharpe_ratio'],
                'max_drawdown':      m['max_drawdown'],
            })
        raw_bundle[label] = sys_dict

    df = pd.DataFrame(rows)

    # ---- CSV ----
    csv_path = os.path.join(out_dir, 'ablation_table.csv')
    df.to_csv(csv_path, index=False)
    print(f"  Wrote {csv_path}")

    # ---- Markdown table (one block per period, easy to paste into thesis) ----
    md_lines = ['# Extended-baseline ablation summary\n']
    for period in PERIODS:
        md_lines.append(f'\n## {PERIOD_TITLES[period]}\n')
        md_lines.append('| System | Cumulative Return (%) | Sharpe | Max Drawdown (%) |')
        md_lines.append('|---|---:|---:|---:|')
        sub = df[df['period'] == period]
        for _, r in sub.iterrows():
            md_lines.append(
                f"| {r['system']} | {r['cumulative_return']:.2f} | "
                f"{r['sharpe_ratio']:.4f} | {r['max_drawdown']:.2f} |"
            )

    # Append a Δ-vs-FULL block for each ablation, since this is what the
    # thesis discussion actually cares about.
    md_lines.append('\n## Δ vs FULL baseline (ablation − FULL)\n')
    md_lines.append('Positive Δ-return is better; positive Δ-Sharpe is better; '
                    'positive Δ-MDD is better (closer to 0).\n')
    md_lines.append('| System | Period | ΔReturn (pp) | ΔSharpe | ΔMDD (pp) |')
    md_lines.append('|---|---|---:|---:|---:|')
    full = df[df['system'] == 'FULL (baseline)'].set_index('period')
    for label, _, _ in SYSTEMS:
        if label == 'FULL (baseline)':
            continue
        for period in PERIODS:
            row = df[(df['system'] == label) & (df['period'] == period)].iloc[0]
            d_ret = row['cumulative_return'] - full.loc[period, 'cumulative_return']
            d_shp = row['sharpe_ratio']      - full.loc[period, 'sharpe_ratio']
            d_mdd = row['max_drawdown']      - full.loc[period, 'max_drawdown']
            md_lines.append(
                f"| {label} | {period} | {d_ret:+.2f} | {d_shp:+.4f} | {d_mdd:+.2f} |"
            )

    md_path = os.path.join(out_dir, 'ablation_table.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(md_lines) + '\n')
    print(f"  Wrote {md_path}")

    # ---- JSON bundle ----
    json_path = os.path.join(out_dir, 'ablation_summary.json')
    with open(json_path, 'w') as f:
        json.dump(raw_bundle, f, indent=2)
    print(f"  Wrote {json_path}")

    # ---- 3x3 grid of bar charts ----
    labels = [s[0] for s in SYSTEMS if s[0] in df['system'].values]
    # Color: FULL highlighted, ablations muted; lambda sweep in a distinct hue
    def color_for(label):
        if label == 'FULL (baseline)':
            return '#2ca02c'    # green
        if label == 'VANILLA':
            return '#ff7f0e'    # orange — the naive baseline
        if label == 'no_floor (eval)':
            return '#d62728'    # red — strongest counter-example
        if label.startswith('no_regime_λ'):
            return '#1f77b4'    # blue family
        return '#7f7f7f'        # gray — no_dsr, no_attention

    fig, axes = plt.subplots(4, 3, figsize=(16, 16))
    x = np.arange(len(labels))
    for row_i, period in enumerate(PERIODS):
        for col_i, (metric, metric_label, higher_is_better) in enumerate(METRICS):
            ax = axes[row_i, col_i]
            vals = [df[(df['system'] == lab) & (df['period'] == period)][metric].iloc[0]
                    for lab in labels]
            colors = [color_for(lab) for lab in labels]
            bars = ax.bar(x, vals, color=colors, edgecolor='black', linewidth=0.5)
            # baseline reference line
            full_val = df[(df['system'] == 'FULL (baseline)') &
                          (df['period'] == period)][metric].iloc[0]
            ax.axhline(full_val, color='#2ca02c', linestyle='--',
                       linewidth=1, alpha=0.6)
            # value labels on bars
            for bar, v in zip(bars, vals):
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2,
                        h + (0.02 * (max(vals) - min(vals)) if max(vals) != min(vals) else 0.1),
                        f'{v:.2f}' if abs(v) >= 1 else f'{v:.3f}',
                        ha='center', va='bottom' if h >= 0 else 'top', fontsize=8)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=8)
            ax.set_title(f'{PERIOD_TITLES[period]} — {metric_label}', fontsize=10)
            ax.grid(True, axis='y', alpha=0.3)
            ax.axhline(0, color='black', linewidth=0.6)
            if metric == 'max_drawdown':
                ax.set_ylabel('MDD % (less negative = better)', fontsize=8)
    plt.suptitle('Extended-baseline ablation grid — FULL (green dashed = reference)',
                 fontsize=13, y=1.00)
    plt.tight_layout()
    grid_path = os.path.join(out_dir, 'ablation_grid.png')
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Wrote {grid_path}")

    # ---- Δ vs FULL chart (one panel per metric, periods grouped) ----
    ablations = [lab for lab in labels if lab != 'FULL (baseline)']
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    x = np.arange(len(ablations))
    # 4 periods grouped per ablation → narrower bars centered on x
    width = 0.20
    n_periods = len(PERIODS)
    offsets = (np.arange(n_periods) - (n_periods - 1) / 2) * width
    period_colors = {'val': '#4c72b0', 'test_bear': '#dd8452',
                     'test_2023': '#937860', 'test_bull': '#55a467'}
    for col_i, (metric, metric_label, higher_is_better) in enumerate(METRICS):
        ax = axes[col_i]
        for i, period in enumerate(PERIODS):
            deltas = []
            for lab in ablations:
                v = df[(df['system'] == lab) & (df['period'] == period)][metric].iloc[0]
                base = df[(df['system'] == 'FULL (baseline)') &
                          (df['period'] == period)][metric].iloc[0]
                deltas.append(v - base)
            ax.bar(x + offsets[i], deltas, width,
                   label=PERIOD_TITLES[period], color=period_colors[period],
                   edgecolor='black', linewidth=0.4)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(ablations, rotation=30, ha='right', fontsize=9)
        ax.set_title(f'Δ {metric_label}  (ablation − FULL)')
        ax.grid(True, axis='y', alpha=0.3)
        if col_i == 0:
            ax.set_ylabel('Δ vs FULL  (positive = ablation better)')
        ax.legend(fontsize=8, loc='best')
    plt.suptitle('How each ablation moves vs FULL baseline '
                 '(positive bars = ablation outperforms baseline on that metric+period)',
                 fontsize=12)
    plt.tight_layout()
    delta_path = os.path.join(out_dir, 'ablation_deltas.png')
    plt.savefig(delta_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Wrote {delta_path}")

    # ---- Print the table to console ----
    print('\n' + '=' * 96)
    print(' EXTENDED ABLATION SUMMARY — all systems × all periods')
    print('=' * 96)
    for period in PERIODS:
        print(f"\n  {PERIOD_TITLES[period]}")
        print(f"  {'System':<22} {'Return %':>10} {'Sharpe':>10} {'MDD %':>10}")
        print(f"  {'-' * 56}")
        sub = df[df['period'] == period]
        for _, r in sub.iterrows():
            print(f"  {r['system']:<22} {r['cumulative_return']:>10.2f} "
                  f"{r['sharpe_ratio']:>10.4f} {r['max_drawdown']:>10.2f}")
    print('\n' + '=' * 96)
    print(f' Files written to: {out_dir}')
    print('=' * 96 + '\n')


if __name__ == '__main__':
    main()
