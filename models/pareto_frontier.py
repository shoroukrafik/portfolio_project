"""
Deep-dive D — Pareto-frontier scatter showing the bull/bear trade-off
across training checkpoints, for every universe + every variant.

Scatter plot:  x-axis = test_bear return,  y-axis = test_bull return
                Each point is one checkpoint, colored by training step.
                Each variant has its own panel.

Reveals visually:
  - Production v6 systems cluster tightly (low variance across training)
  - NDX 2017 traces a clear Pareto curve (bear ↓ as bull ↑)
  - The val-Sharpe-best point per variant overlaid as a star

Outputs:
  results/pareto_frontier.png
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap


# All variants with full per-checkpoint sweeps
VARIANTS = [
    # (label, results_subdir, panel_position (row, col))
    ('DJI v6 universal',     'universal_v6_dji',        (0, 0)),
    ('SP100 v6 universal',   'universal_v6_sp100',      (0, 1)),
    ('NDX v6 production',    'universal_v6_ndx',        (0, 2)),
    ('NDX exp1 lower bull λ','universal_v6_ndx_exp1',   (1, 0)),
    ('NDX exp2 concentration','universal_v6_ndx_exp2',  (1, 1)),
    ('NDX exp3 log-return',  'universal_v6_ndx_exp3',   (1, 2)),
    ('NDX exp4 low ent_coef','universal_v6_ndx_exp4',   (2, 0)),
    ('NDX 2017-variant',     'universal_v6_ndx_2017',   (2, 1)),
]


def load_sweep(subdir):
    path = f'results/{subdir}/checkpoint_sweep.json'
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main():
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    cmap = get_cmap('viridis')

    for label, sub, (row, col) in VARIANTS:
        ax = axes[row, col]
        rows = load_sweep(sub)
        if rows is None:
            ax.set_title(f'{label}\n(no sweep data)')
            ax.axis('off')
            continue
        bears = np.array([r['test_bear_ret'] for r in rows])
        bulls = np.array([r['test_bull_ret'] for r in rows])
        steps = np.array([r['step']           for r in rows])
        val_sharpes = np.array([r['val_sharpe'] for r in rows])

        # Color by training step
        sc = ax.scatter(bears, bulls, c=steps / 1000, cmap=cmap,
                        s=50, edgecolor='black', lw=0.5, alpha=0.85)
        ax.plot(bears, bulls, color='gray', lw=0.5, alpha=0.5)  # connect path

        # Highlight val-Sharpe-best
        best_idx = int(np.argmax(val_sharpes))
        ax.scatter([bears[best_idx]], [bulls[best_idx]], marker='*',
                    s=400, color='red', edgecolor='black', lw=1.2, zorder=5,
                    label=f'val-best (step {steps[best_idx]/1000:.0f}k)')

        # Annotate the best checkpoint
        ax.annotate(f"  step {steps[best_idx]/1000:.0f}k\n  bull={bulls[best_idx]:+.1f}%\n  bear={bears[best_idx]:+.1f}%",
                    xy=(bears[best_idx], bulls[best_idx]),
                    xytext=(8, 8), textcoords='offset points',
                    fontsize=7, color='darkred',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

        # Add benchmark lines if this is an NDX panel
        if 'NDX' in label:
            ax.axhline(28.13, color='green', linestyle=':', lw=1, alpha=0.6, label='^NDX bull')
            ax.axvline(-33.71, color='red',  linestyle=':', lw=1, alpha=0.6, label='^NDX bear')

        ax.set_xlabel('test_bear return (%)')
        ax.set_ylabel('test_bull return (%)')
        ax.set_title(label, fontsize=11, weight='bold')
        ax.grid(alpha=0.3)
        ax.axhline(0, color='gray', lw=0.5)
        ax.axvline(0, color='gray', lw=0.5)
        ax.legend(loc='lower right', fontsize=7)
        plt.colorbar(sc, ax=ax, label='step (×1000)', shrink=0.7)

    # Hide the unused subplot (row 2, col 2)
    axes[2, 2].axis('off')

    plt.suptitle('Pareto frontier across training — bull vs bear trade-off per variant\n'
                  '(★ = val-Sharpe-best checkpoint; color = training step)',
                  fontsize=13, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('results/pareto_frontier.png', dpi=130, bbox_inches='tight')
    plt.close()
    print('✅ saved results/pareto_frontier.png')


if __name__ == '__main__':
    main()
