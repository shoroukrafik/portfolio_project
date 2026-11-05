"""
Generate Figure 3.2: S&P 500 index 2007-2024 with HMM regime shading.

Reads data/sp500_index_ext.csv and data/regime_labels_sp30.csv, then
draws the S&P 500 (rebased to 100) with each trading day's background
coloured by its regime label (bull = green, sideways = grey, bear = red).

Output: images/fig_hmm_regimes.pdf and images/fig_hmm_regimes.png
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
SP500_CSV = ROOT / "data" / "sp500_index_ext.csv"
LABELS_CSV = ROOT / "data" / "regime_labels_sp30.csv"
OUT_DIR = ROOT / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REGIME_COLORS = {
    "bull": "#cde7c8",
    "sideways": "#e6e6e6",
    "bear": "#f4c7c3",
}


def main() -> None:
    df = pd.read_csv(SP500_CSV, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    labels = pd.read_csv(LABELS_CSV, parse_dates=["date"])[["date", "regime_name"]]

    df = df.merge(labels, on="date", how="inner").sort_values("date").reset_index(drop=True)

    base_val = df.iloc[0]["close"]
    df["index"] = 100.0 * df["close"] / base_val

    fig, ax = plt.subplots(figsize=(11, 4.5))

    df["regime_block"] = (df["regime_name"] != df["regime_name"].shift()).cumsum()
    for _, block in df.groupby("regime_block"):
        regime = block["regime_name"].iloc[0]
        start = block["date"].iloc[0]
        end = block["date"].iloc[-1]
        ax.axvspan(start, end, color=REGIME_COLORS[regime], alpha=0.85, zorder=0)

    ax.plot(df["date"], df["index"], color="#1f3a5f", linewidth=1.2, zorder=2)

    ax.set_xlim(df["date"].min(), df["date"].max())
    ax.set_ylim(bottom=50)
    ax.set_xlabel("Date")
    ax.set_ylabel("S\\&P 500 index (rebased to 100 at start)")

    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.6)

    legend_handles = [
        mpatches.Patch(color=REGIME_COLORS["bull"], label="Bull regime"),
        mpatches.Patch(color=REGIME_COLORS["sideways"], label="Sideways regime"),
        mpatches.Patch(color=REGIME_COLORS["bear"], label="Bear regime"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8, framealpha=0.9)

    plt.tight_layout()
    out_pdf = OUT_DIR / "fig_hmm_regimes.pdf"
    out_png = OUT_DIR / "fig_hmm_regimes.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
