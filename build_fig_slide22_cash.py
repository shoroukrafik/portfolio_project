"""Generate the per-regime cash-over-time figure for Slide 22.

Three rows (DJI / SP100 / NDX), each showing the agent's cash weight across
the full 4-period evaluation timeline. Background shaded by HMM regime label.

Output: navy-background PNG matching the Canva template.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch


# Palette
BG = "#0E1B3A"
TEXT = "white"
TITLE = "#7DC8FF"
SUBTEXT = "#C9D5EA"
CASH_LINE = "#7DC8FF"
FLOOR_LINE = "#FFC97A"          # amber for floor reference
BULL_C = "#3FBF7F"
SIDEWAYS_C = "#8E9DBC"
BEAR_C = "#E66060"


REGIME_COLOR = {
    "bull": BULL_C,
    "sideways": SIDEWAYS_C,
    "bear": BEAR_C,
}

MARKETS = [
    ("DJI 30",      "DJI",   0.70),
    ("S&P 100",     "SP100", 0.30),
    ("NDX 2017",    "NDX",   0.00),
]

PERIOD_KEYS = ["val", "test_bear", "test_2023", "test_bull"]
PERIOD_LABELS = {
    "val":       "Val 2021",
    "test_bear": "Bear 2022",
    "test_2023": "Recovery 2023",
    "test_bull": "Bull 2024",
}


def main():
    proj = Path(__file__).parent
    with open(proj / "results" / "_canonical" / "per_regime_behavior.json") as f:
        data = json.load(f)

    fig, axes = plt.subplots(3, 1, figsize=(16, 9), dpi=240, sharex=True)
    fig.patch.set_facecolor(BG)

    for row_idx, (market_name, key, floor) in enumerate(MARKETS):
        ax = axes[row_idx]
        ax.set_facecolor(BG)

        # Concatenate all 4 periods
        all_dates = []
        all_cash = []
        all_regimes = []
        period_boundaries = []
        for p_key in PERIOD_KEYS:
            rows = data[key][p_key]
            dates = [pd.to_datetime(r["date"]) for r in rows]
            cash = [r["cash_weight"] for r in rows]
            regimes = [r["regime_name"] for r in rows]
            all_dates += dates
            all_cash += cash
            all_regimes += regimes
            period_boundaries.append((dates[0], dates[-1], PERIOD_LABELS[p_key]))

        all_dates = pd.to_datetime(all_dates)
        all_cash = np.array(all_cash)

        # Regime shading: find contiguous runs of same regime
        i = 0
        while i < len(all_regimes):
            j = i
            while j + 1 < len(all_regimes) and all_regimes[j + 1] == all_regimes[i]:
                j += 1
            ax.axvspan(all_dates[i], all_dates[j],
                       color=REGIME_COLOR[all_regimes[i]],
                       alpha=0.20, linewidth=0)
            i = j + 1

        # Cash line on top
        ax.plot(all_dates, all_cash,
                color=CASH_LINE, linewidth=1.6, alpha=1.0, zorder=5)

        # Floor reference line (if positive)
        if floor > 0:
            ax.axhline(floor, color=FLOOR_LINE, linewidth=1.0,
                       linestyle="--", alpha=0.8, zorder=4)
            ax.text(all_dates[-1], floor + 0.05, f"floor = {int(floor*100)}%",
                    color=FLOOR_LINE, fontsize=9, ha="right", va="bottom",
                    family="sans-serif", fontweight="bold")
        else:
            ax.text(all_dates[-1], 0.85, "no floor",
                    color=FLOOR_LINE, fontsize=9, ha="right", va="bottom",
                    family="sans-serif", fontweight="bold", style="italic")

        # Period boundary dashed verticals
        for start, _, _ in period_boundaries[1:]:
            ax.axvline(start, color=SUBTEXT, linewidth=0.5,
                       linestyle=":", alpha=0.4)

        # Axes formatting
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(all_dates[0], all_dates[-1])
        ax.tick_params(axis="x", colors=SUBTEXT, labelsize=9)
        ax.tick_params(axis="y", colors=SUBTEXT, labelsize=9)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("bottom", "left"):
            ax.spines[spine].set_color(SUBTEXT)
            ax.spines[spine].set_linewidth(0.5)

        ax.set_ylabel(f"{market_name}\ncash weight",
                      color=TEXT, fontsize=11, fontweight="bold")
        ax.grid(True, axis="y", linestyle=":", color=SUBTEXT, alpha=0.18)

        # Period labels above first row
        if row_idx == 0:
            for start, end, label in period_boundaries:
                mid = start + (end - start) / 2
                ax.text(mid, 1.12, label, ha="center", va="bottom",
                        color=TITLE, fontsize=11, fontweight="bold",
                        family="sans-serif")

    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Figure-level title
    fig.text(0.5, 0.97,
             "Agent cash weight over time   ·   background shaded by HMM regime",
             ha="center", va="center",
             color=TITLE, fontsize=16, fontweight="bold")

    # Legend at bottom of figure
    legend_handles = [
        Patch(facecolor=BULL_C, alpha=0.7, label="Bull"),
        Patch(facecolor=SIDEWAYS_C, alpha=0.7, label="Sideways"),
        Patch(facecolor=BEAR_C, alpha=0.7, label="Bear"),
    ]
    leg = fig.legend(handles=legend_handles,
                     loc="lower center", ncol=3, frameon=False,
                     fontsize=11, bbox_to_anchor=(0.5, 0.01),
                     handlelength=1.6, handleheight=1.0)
    for text in leg.get_texts():
        text.set_color(TEXT)
        text.set_fontweight("bold")

    plt.tight_layout(rect=[0, 0.05, 1, 0.93])

    out_path = proj / "slide22_cash_over_time.png"
    plt.savefig(out_path, dpi=240, bbox_inches="tight",
                facecolor=BG, pad_inches=0.20)
    plt.close(fig)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
