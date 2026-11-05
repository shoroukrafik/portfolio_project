"""
Full comparison chart for all 6 NDX experimental variants.

Plots, per period (bear / 2023 / bull):
  - Cumulative return over training steps  (top row, 3 panels)
  - Sharpe ratio over training steps        (bottom row, 3 panels)

Plus the val Sharpe trajectory as a 7th panel below.
Includes ^NDX benchmark reference lines.

Output:
  results/ndx_experiments_full_chart.png
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt


VARIANTS = [
    # (display_name, results_subdir, color, linestyle, marker)
    ('production v6',     'universal_v6_ndx',       '#1f77b4', '-',  'o'),
    ('exp1 lower bull λ', 'universal_v6_ndx_exp1',  '#ff7f0e', '--', 's'),
    ('exp2 concentration','universal_v6_ndx_exp2',  '#2ca02c', '--', '^'),
    ('exp3 log-return',   'universal_v6_ndx_exp3',  '#d62728', '--', 'D'),
    ('exp4 low ent_coef', 'universal_v6_ndx_exp4',  '#9467bd', '--', 'v'),
    ('2017-variant',      'universal_v6_ndx_2017',  '#8c564b', '-',  '*'),
]
NDX_BENCH = {
    'val':       +26.63,
    'test_bear': -33.71,
    'test_2023': +54.90,
    'test_bull': +28.13,
}


def load_sweep(subdir):
    path = f'results/{subdir}/checkpoint_sweep.json'
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main():
    # Pre-load every variant
    sweeps = {}
    for label, sub, color, ls, marker in VARIANTS:
        s = load_sweep(sub)
        if s is None:
            print(f"  ⚠  missing {sub}/checkpoint_sweep.json")
            continue
        sweeps[label] = {'rows': s, 'color': color, 'ls': ls, 'marker': marker}

    fig = plt.figure(figsize=(18, 12))
    # 3x2 grid for bear/2023/bull (rows) × return/Sharpe (cols)
    # Layout: top row = bear (left=return, right=Sharpe), middle = 2023, bottom = bull
    period_labels = ['test_bear', 'test_2023', 'test_bull']
    period_titles = ['BEAR (2022)', '2023 Recovery', 'BULL (2024)']
    metric_keys = ['_ret', '_sharpe']
    metric_titles = ['Cumulative return (%)', 'Sharpe ratio']

    for row, (p, ptitle) in enumerate(zip(period_labels, period_titles)):
        for col, (mk, mtitle) in enumerate(zip(metric_keys, metric_titles)):
            ax = plt.subplot(3, 2, row*2 + col + 1)
            metric_field = f'{p}{mk}'
            for label, data in sweeps.items():
                steps = [r['step'] for r in data['rows']]
                vals  = [r[metric_field] for r in data['rows']]
                ax.plot(steps, vals, color=data['color'], linestyle=data['ls'],
                        marker=data['marker'], markersize=4, lw=1.5,
                        label=label, alpha=0.9)
            ax.set_xlabel('PPO training step')
            ax.set_ylabel(mtitle)
            ax.set_title(f'{ptitle} — {mtitle}', fontsize=11, weight='bold')
            ax.grid(alpha=0.3)
            ax.axhline(0, color='gray', lw=0.5)
            # Benchmark reference line for returns
            if mk == '_ret':
                bench = NDX_BENCH[p]
                ax.axhline(bench, color='black', linestyle=':', lw=1.2, alpha=0.6,
                           label=f'^NDX {p.replace("test_","")} ({bench:+.1f}%)')
            if row == 0 and col == 0:
                ax.legend(loc='best', fontsize=8, ncol=2)

    plt.suptitle('NDX experimental variants — per-period returns & Sharpe across training\n'
                  '(production = 1M steps, all others = 500k)',
                  fontsize=13, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('results/ndx_experiments_full_chart.png', dpi=130, bbox_inches='tight')
    plt.close()
    print('✅ saved results/ndx_experiments_full_chart.png')


if __name__ == '__main__':
    main()
