"""
Figure for Section 4.5 (Attention diagnostics).

Two figures:
  fig_attention_top_stocks.{pdf,png}
    Per universe, two bar panels: top 10 attended stocks in bull-regime
    days vs in bear-regime days, pooled across all four test periods.
    Shows the agent's attention shift across regimes.

  fig_attention_diff.{pdf,png}
    Per universe, ranked horizontal bar of the stocks with the largest
    bear-minus-bull attention change. Positive bars = more attention
    on bear days; negative bars = more attention on bull days.

Both figures use the data extracted by models/extract_attention_canonical.py.
"""

import json
import os

import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = 'results/_canonical/attention'
OUT_DIR  = 'images'
UNIVERSES = ['DJI', 'SP100']
PERIODS   = ['val', 'test_bear', 'test_2023', 'test_bull']


def load_pooled_regime_attention(universe):
    """For each universe, average attention per stock across all four
    test periods, separately for bull-regime and bear-regime days.
    Days are weighted by the number of regime-days in each period.

    Returns: dict mapping ticker -> (bull_score, bear_score).
    """
    bull_sum = None
    bear_sum = None
    bull_days = 0
    bear_days = 0
    tickers = None

    for period in PERIODS:
        with open(os.path.join(DATA_DIR, f'{universe}_{period}_tickers.json')) as f:
            tickers = json.load(f)
        with open(os.path.join(DATA_DIR, 'all_summaries.json')) as f:
            summaries = json.load(f)
        regime_days = summaries[universe][period]['regime_day_counts']

        # Stacked matrix (3, N, N): bull/bear/sideways
        stacked = np.load(os.path.join(DATA_DIR, f'{universe}_{period}_attention_by_regime.npy'))
        # Per-stock attention received (mean over the column dim = how much each stock is attended to on average)
        bull_per_stock = stacked[0].mean(axis=0)  # (N,)
        bear_per_stock = stacked[1].mean(axis=0)  # (N,)

        n_bull = regime_days['bull']
        n_bear = regime_days['bear']

        if bull_sum is None:
            bull_sum = np.zeros_like(bull_per_stock)
            bear_sum = np.zeros_like(bear_per_stock)

        if n_bull > 0:
            bull_sum += bull_per_stock * n_bull
            bull_days += n_bull
        if n_bear > 0:
            bear_sum += bear_per_stock * n_bear
            bear_days += n_bear

    bull_avg = bull_sum / max(1, bull_days)
    bear_avg = bear_sum / max(1, bear_days)

    return {tickers[i]: (float(bull_avg[i]), float(bear_avg[i])) for i in range(len(tickers))}


def fig_top_stocks_bar():
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5))

    for row, u in enumerate(UNIVERSES):
        scores = load_pooled_regime_attention(u)
        items = list(scores.items())
        # Top 10 by bull regime score
        items.sort(key=lambda kv: kv[1][0], reverse=True)
        top_bull = items[:10]
        # Top 10 by bear regime score
        items.sort(key=lambda kv: kv[1][1], reverse=True)
        top_bear = items[:10]

        # Bull panel
        ax_b = axes[row, 0]
        names = [t for t, _ in top_bull][::-1]
        vals  = [s[0] for _, s in top_bull][::-1]
        ax_b.barh(range(len(names)), vals, color='#79b864', edgecolor='#3b8439', height=0.7)
        ax_b.set_yticks(range(len(names)))
        ax_b.set_yticklabels(names, fontsize=9)
        ax_b.set_title(f'{u} — top 10 attended stocks on bull-regime days', fontsize=10)
        ax_b.tick_params(axis='x', labelsize=7)
        ax_b.grid(True, axis='x', linestyle=':', linewidth=0.4, alpha=0.5)

        # Bear panel
        ax_r = axes[row, 1]
        names = [t for t, _ in top_bear][::-1]
        vals  = [s[1] for _, s in top_bear][::-1]
        ax_r.barh(range(len(names)), vals, color='#d27e7e', edgecolor='#a44141', height=0.7)
        ax_r.set_yticks(range(len(names)))
        ax_r.set_yticklabels(names, fontsize=9)
        ax_r.set_title(f'{u} — top 10 attended stocks on bear-regime days', fontsize=10)
        ax_r.tick_params(axis='x', labelsize=7)
        ax_r.grid(True, axis='x', linestyle=':', linewidth=0.4, alpha=0.5)

    plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    pdf = os.path.join(OUT_DIR, 'fig_attention_top_stocks.pdf')
    png = os.path.join(OUT_DIR, 'fig_attention_top_stocks.png')
    fig.savefig(pdf, bbox_inches='tight')
    fig.savefig(png, dpi=180, bbox_inches='tight')
    print(f'Wrote {pdf}')
    print(f'Wrote {png}')
    plt.close(fig)


def fig_attention_diff():
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    for col, u in enumerate(UNIVERSES):
        scores = load_pooled_regime_attention(u)
        # Compute bear - bull
        deltas = [(t, s[1] - s[0]) for t, s in scores.items()]
        deltas.sort(key=lambda kv: kv[1], reverse=True)
        # Top 7 most-into-bear and top 7 most-into-bull
        most_bear = deltas[:7]
        most_bull = deltas[-7:][::-1]

        combined = most_bear + most_bull
        names = [t for t, _ in combined]
        vals  = [d for _, d in combined]
        colors = ['#d27e7e'] * 7 + ['#79b864'] * 7

        ax = axes[col]
        positions = np.arange(len(names))
        ax.barh(positions, vals, color=colors, edgecolor='black', linewidth=0.4, height=0.7)
        ax.axvline(0, color='gray', linewidth=0.6)
        ax.set_yticks(positions)
        ax.set_yticklabels(names, fontsize=9)
        ax.invert_yaxis()
        ax.set_title(f'{u} — top attention shifts bear minus bull', fontsize=10)
        ax.set_xlabel('Bear regime attention minus bull regime attention')
        ax.tick_params(axis='x', labelsize=7)
        ax.grid(True, axis='x', linestyle=':', linewidth=0.4, alpha=0.5)

    plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    pdf = os.path.join(OUT_DIR, 'fig_attention_diff.pdf')
    png = os.path.join(OUT_DIR, 'fig_attention_diff.png')
    fig.savefig(pdf, bbox_inches='tight')
    fig.savefig(png, dpi=180, bbox_inches='tight')
    print(f'Wrote {pdf}')
    print(f'Wrote {png}')
    plt.close(fig)


def main():
    fig_top_stocks_bar()
    fig_attention_diff()


if __name__ == '__main__':
    main()
