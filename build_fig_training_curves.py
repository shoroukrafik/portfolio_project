"""
Training-curve figure for Section 3.5.1 (Two-step selection rule).

Plots validation Sharpe ratio vs PPO training step for the three
canonical training runs (DJI 2M, S&P 100 2M, NDX 2017 500k). The
DEFENSIVE rule's chosen checkpoint is marked on each curve with a
larger filled marker. The 20% training-budget threshold (below
which DEFENSIVE excludes checkpoints) is drawn as a vertical
dashed line per universe.

Reads from results/*/checkpoint_sweep.json.
Writes to images/fig_training_curves.{pdf,png}.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np


CANONICAL = {
    'DJI 30 (2M budget)': {
        'path': 'results/universal_v6_dji_mom_2m/checkpoint_sweep.json',
        'defensive_step': 800_000,
        'budget': 2_000_000,
        'color': '#1f77b4',  # blue
    },
    'S&P 100 (2M budget)': {
        'path': 'results/universal_v6_sp100_mom_2m/checkpoint_sweep.json',
        'defensive_step': 1_150_000,
        'budget': 2_000_000,
        'color': '#d62728',  # red
    },
    'NDX 2017 (500k budget)': {
        'path': 'results/universal_v6_ndx_2017_mom/checkpoint_sweep.json',
        'defensive_step': 100_000,
        'budget': 500_000,
        'color': '#2ca02c',  # green
    },
}


def main():
    fig, ax = plt.subplots(figsize=(9.0, 5.0))

    for label, cfg in CANONICAL.items():
        with open(cfg['path']) as f:
            sweep = json.load(f)
        steps = np.array([r['step'] for r in sweep])
        sharpes = np.array([r['val_sharpe'] for r in sweep])
        # Express step as fraction of training budget so the three runs
        # share a common x-axis without rescaling losses
        x = steps / cfg['budget']

        # Main curve
        ax.plot(x, sharpes, color=cfg['color'], linewidth=1.6,
                label=label, alpha=0.85)

        # DEFENSIVE pick marker
        pick_idx = int(np.argmin(np.abs(steps - cfg['defensive_step'])))
        ax.scatter([x[pick_idx]], [sharpes[pick_idx]],
                   color=cfg['color'], s=130, marker='*',
                   edgecolor='black', linewidth=0.8, zorder=5)

    # 20% defensive threshold
    ax.axvline(0.20, color='gray', linestyle='--', linewidth=0.9, alpha=0.8)
    ax.text(0.205, ax.get_ylim()[0] + 0.05, 'DEFENSIVE threshold (20% of budget)',
            fontsize=9, color='gray', va='bottom')

    ax.set_xlabel('Training progress (fraction of total step budget)')
    ax.set_ylabel('Validation Sharpe ratio')
    ax.set_title('Validation Sharpe over training for the three canonical models',
                 fontsize=12)
    ax.grid(True, axis='y', linestyle=':', linewidth=0.4, alpha=0.5)
    ax.set_xlim(0, 1.0)
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)

    # Star legend entry
    ax.scatter([], [], color='black', s=130, marker='*',
               edgecolor='black', linewidth=0.8,
               label='DEFENSIVE pick (canonical checkpoint)')
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)

    plt.tight_layout()
    os.makedirs('images', exist_ok=True)
    fig.savefig('images/fig_training_curves.pdf', bbox_inches='tight')
    fig.savefig('images/fig_training_curves.png', dpi=180, bbox_inches='tight')
    print('Wrote images/fig_training_curves.pdf')
    print('Wrote images/fig_training_curves.png')
    plt.close(fig)


if __name__ == '__main__':
    main()
