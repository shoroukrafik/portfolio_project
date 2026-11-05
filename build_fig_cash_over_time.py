"""
Figure 4.2 — Cash weight over time with regime shading.

For each universe × each test period, plot the canonical agent's cash
weight per day with the background shaded by HMM regime label (bull,
sideways, bear).

Reads results/_canonical/per_regime_behavior.json.

Output: images/fig_cash_over_time.pdf + images/fig_cash_over_time.png
"""

import json
import os

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BEHAVIOR_PATH = 'results/_canonical/per_regime_behavior.json'
OUT_DIR = 'images'

REGIME_COLORS = {
    'bull':     '#cde7c8',
    'sideways': '#e6e6e6',
    'bear':     '#f4c7c3',
}

UNIVERSE_TITLES = {
    'DJI':   'DJI 30 (70% bear floor)',
    'SP100': 'S&P 100 (30% bear floor)',
    'NDX':   'NASDAQ-100 (no floor)',
}

PERIOD_TITLES = {
    'val':       'Validation (2021)',
    'test_bear': 'Bear (2022)',
    'test_2023': 'Recovery (2023)',
    'test_bull': 'Bull (2024)',
}


def main():
    with open(BEHAVIOR_PATH) as f:
        data = json.load(f)

    fig, axes = plt.subplots(3, 4, figsize=(15, 8.5))

    for row_idx, u in enumerate(['DJI', 'SP100', 'NDX']):
        for col_idx, p in enumerate(['val', 'test_bear', 'test_2023', 'test_bull']):
            ax = axes[row_idx, col_idx]
            records = data[u][p]
            df = pd.DataFrame(records)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)

            # Shade regime blocks
            df['block'] = (df['regime_name'] != df['regime_name'].shift()).cumsum()
            for _, block in df.groupby('block'):
                regime = block['regime_name'].iloc[0]
                start = block['date'].iloc[0]
                end = block['date'].iloc[-1]
                ax.axvspan(start, end, color=REGIME_COLORS.get(regime, '#ffffff'),
                           alpha=0.9, zorder=0)

            # Plot cash %
            ax.plot(df['date'], df['cash_weight'] * 100,
                    color='#1f3a5f', linewidth=1.2, zorder=2)
            ax.set_ylim(-2, 100)
            ax.axhline(0, color='#bbbbbb', linewidth=0.4)

            ax.tick_params(axis='both', labelsize=7)
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

            if row_idx == 0:
                ax.set_title(PERIOD_TITLES[p], fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(f"{UNIVERSE_TITLES[u]}\nCash %", fontsize=9, fontweight='bold')

    # Legend at the top
    legend_handles = [
        mpatches.Patch(color=REGIME_COLORS['bull'],     label='Bull regime'),
        mpatches.Patch(color=REGIME_COLORS['sideways'], label='Sideways regime'),
        mpatches.Patch(color=REGIME_COLORS['bear'],     label='Bear regime'),
    ]
    fig.legend(handles=legend_handles, loc='upper center', ncol=3,
               bbox_to_anchor=(0.5, 0.99), fontsize=9, framealpha=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(OUT_DIR, exist_ok=True)
    pdf = os.path.join(OUT_DIR, 'fig_cash_over_time.pdf')
    png = os.path.join(OUT_DIR, 'fig_cash_over_time.png')
    fig.savefig(pdf, bbox_inches='tight')
    fig.savefig(png, dpi=180, bbox_inches='tight')
    print(f"Wrote {pdf}")
    print(f"Wrote {png}")


if __name__ == '__main__':
    main()
