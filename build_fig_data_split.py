"""
Generate Figure 3.1: train/val/test split with S&P 500 index overlaid.

Reads data/sp500_index_ext.csv, normalises the close series to 100 at the
start of the training period, and produces a PDF + PNG showing the four
data periods as shaded backgrounds with the S&P 500 line over the top.

Output: images/fig_data_split.pdf and images/fig_data_split.png
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
SP500_CSV = ROOT / "data" / "sp500_index_ext.csv"
OUT_DIR = ROOT / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_START = pd.Timestamp("2007-02-01")
TRAIN_END = pd.Timestamp("2020-12-31")
VAL_START = pd.Timestamp("2021-01-01")
VAL_END = pd.Timestamp("2021-12-31")
BEAR_START = pd.Timestamp("2022-01-01")
BEAR_END = pd.Timestamp("2022-12-31")
RECOVERY_START = pd.Timestamp("2023-01-01")
RECOVERY_END = pd.Timestamp("2023-12-31")
BULL_START = pd.Timestamp("2024-01-01")
BULL_END = pd.Timestamp("2024-12-31")

COLORS = {
    "train": "#d9e6f2",
    "val": "#fff2cc",
    "bear": "#f9d4d4",
    "recovery": "#fbe5cd",
    "bull": "#d9ead3",
}


def main() -> None:
    df = pd.read_csv(SP500_CSV, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    base = df.loc[df["date"] == TRAIN_START, "close"]
    if base.empty:
        base_val = df.iloc[0]["close"]
    else:
        base_val = base.iloc[0]
    df["index"] = 100.0 * df["close"] / base_val

    fig, ax = plt.subplots(figsize=(11, 4.5))

    ax.axvspan(TRAIN_START, TRAIN_END, color=COLORS["train"], label="Train (2007--2020)")
    ax.axvspan(VAL_START, VAL_END, color=COLORS["val"], label="Validation (2021)")
    ax.axvspan(BEAR_START, BEAR_END, color=COLORS["bear"], label="Bear test (2022)")
    ax.axvspan(RECOVERY_START, RECOVERY_END, color=COLORS["recovery"], label="Recovery test (2023)")
    ax.axvspan(BULL_START, BULL_END, color=COLORS["bull"], label="Bull test (2024)")

    ax.plot(df["date"], df["index"], color="#1f3a5f", linewidth=1.4, label="S\\&P 500 (rebased to 100)")

    ax.set_xlim(TRAIN_START, BULL_END)
    ax.set_ylim(bottom=50)

    ax.set_xlabel("Date")
    ax.set_ylabel("S\\&P 500 index (rebased to 100 at 2007-02-01)")

    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.6)

    handles, labels = ax.get_legend_handles_labels()
    order = [5, 0, 1, 2, 3, 4]
    handles = [handles[i] for i in order]
    labels = [labels[i] for i in order]
    ax.legend(
        handles,
        labels,
        loc="upper left",
        fontsize=8,
        ncol=2,
        framealpha=0.9,
    )

    for boundary, label_text in [
        (TRAIN_END, "Train\nends"),
        (VAL_END, "Val\nends"),
        (BEAR_END, "2022"),
        (RECOVERY_END, "2023"),
    ]:
        ax.axvline(boundary, color="#555555", linewidth=0.6, linestyle="--", alpha=0.5)

    plt.tight_layout()

    out_pdf = OUT_DIR / "fig_data_split.pdf"
    out_png = OUT_DIR / "fig_data_split.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")

    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
