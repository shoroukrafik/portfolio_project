"""Generate a navy-background train/val/test timeline for Slide 14.

S&P 500 index 2007-2024, rebased to 100, with each period coloured:
training (grey-blue), validation (gold), bear test (red), recovery (orange),
bull test (green). Crisis annotations (2008 GFC, 2020 COVID) on the line.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


# Palette
BG = "#0E1B3A"
TEXT = "white"
TITLE = "#7DC8FF"
SUBTEXT = "#C9D5EA"
LINE = "white"

TRAIN_C = "#3B5D9C"          # cool blue for training (long, dominant)
VAL_C   = "#E8B057"          # gold for validation
BEAR_C  = "#E66060"          # soft red for bear
REC_C   = "#EE9554"          # warm orange for recovery
BULL_C  = "#3FBF7F"          # soft green for bull


def main():
    proj = Path(__file__).parent
    px = pd.read_csv(proj / "data" / "sp500_index_ext.csv",
                     parse_dates=["date"])
    df = px.sort_values("date").reset_index(drop=True)
    df = df[df["date"] >= "2007-01-01"].copy()
    df["rebased"] = df["close"] / df["close"].iloc[0] * 100.0

    fig = plt.figure(figsize=(16, 8.5), dpi=240)
    ax = fig.add_subplot(111)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    # --- Period shading -------------------------------------------------
    periods = [
        ("2007-01-01", "2020-12-31", TRAIN_C, "TRAINING\n2007 – 2020 (~14 y)"),
        ("2021-01-01", "2021-12-31", VAL_C,   "VAL\n2021"),
        ("2022-01-01", "2022-12-31", BEAR_C,  "BEAR\n2022"),
        ("2023-01-01", "2023-12-31", REC_C,   "RECOVERY\n2023"),
        ("2024-01-01", "2024-12-31", BULL_C,  "BULL\n2024"),
    ]
    for start, end, color, _ in periods:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   color=color, alpha=0.32, linewidth=0)

    # --- S&P 500 line ---------------------------------------------------
    ax.plot(df["date"], df["rebased"],
            color=LINE, linewidth=1.7, alpha=1.0, zorder=5)

    # --- Crisis annotations (just within training window) -------------
    def value_at(date):
        return float(df.loc[df["date"] >= date, "rebased"].iloc[0])

    crises = [
        ("2008 GFC",   pd.Timestamp("2008-11-15"), 60),
        ("2020 COVID", pd.Timestamp("2020-03-23"), 80),
    ]
    for label, when, dy in crises:
        y = value_at(when)
        ax.annotate(
            label,
            xy=(when, y),
            xytext=(when, y + dy),
            ha="center", va="bottom",
            color=TEXT, fontsize=11, fontweight="bold",
            zorder=6,
            arrowprops=dict(arrowstyle="-", color=TEXT,
                            linewidth=1.0, alpha=0.7),
        )

    # --- Period labels above each shaded region ------------------------
    label_y = 470
    for start, end, color, label in periods:
        mid = pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2
        ax.text(
            mid, label_y, label,
            ha="center", va="bottom",
            color=color, fontsize=11, fontweight="bold",
            family="sans-serif",
        )

    # --- Axes styling --------------------------------------------------
    ax.set_xlim(df["date"].min(), df["date"].max())
    ax.set_ylim(40, 520)

    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", colors=SUBTEXT, labelsize=11, length=4)
    ax.tick_params(axis="y", colors=SUBTEXT, labelsize=10, length=3)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color(SUBTEXT)
        ax.spines[spine].set_linewidth(0.7)

    ax.set_ylabel("S&P 500  (rebased to 100)", color=SUBTEXT, fontsize=11)

    # --- Title inside the figure ---------------------------------------
    fig.text(
        0.5, 0.96,
        "TRAIN  ·  VALIDATE  ·  STRESS-TEST",
        ha="center", va="center",
        color=TITLE, fontsize=18, fontweight="bold",
    )
    fig.text(
        0.5, 0.925,
        "S&P 500 · 2007–2024 · the four-period design",
        ha="center", va="center",
        color=SUBTEXT, fontsize=12, style="italic",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.90])

    out_path = proj / "slide14_timeline.png"
    plt.savefig(out_path, dpi=320, bbox_inches="tight",
                facecolor=BG, pad_inches=0.20)
    plt.close(fig)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
