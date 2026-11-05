"""Generate a transparent PNG figure for Slide 9: 'Detecting the market regime'.

S&P 500 index 2007-2024, rebased to 100, with each trading day shaded by the
HMM-assigned regime label (bull / sideways / bear).

Navy + cyan palette matched to the user's Canva template, with soft-green for
bull and soft-red for bear so the regime stripes read intuitively.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np


# Palette
BG = "#0E1B3A"          # solid navy, matches Canva template
TEXT = "white"
TITLE = "#7DC8FF"
SUBTEXT = "#C9D5EA"
LINE = "white"
BULL = "#3FBF7F"        # soft green
BEAR = "#E66060"        # soft red
SIDEWAYS = "#8E9DBC"    # soft blue-grey


def main():
    proj = Path(__file__).parent
    labels = pd.read_csv(proj / "data" / "regime_labels_sp30.csv",
                         parse_dates=["date"])
    px = pd.read_csv(proj / "data" / "sp500_index_ext.csv",
                     parse_dates=["date"])

    # Restrict to the labelled span (HMM labels run 2007 to 2024)
    df = labels.merge(px[["date", "close"]], on="date", how="inner")
    df = df.sort_values("date").reset_index(drop=True)

    # Rebase index to 100 at the start of the series
    df["rebased"] = df["close"] / df["close"].iloc[0] * 100.0

    fig = plt.figure(figsize=(16, 7.5), dpi=200)
    ax = fig.add_subplot(111)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    # Build contiguous regime spans for shading
    df["regime_block"] = (df["regime"] != df["regime"].shift()).cumsum()
    spans = df.groupby("regime_block").agg(
        start=("date", "first"),
        end=("date", "last"),
        regime=("regime", "first"),
        regime_name=("regime_name", "first"),
    )

    regime_color = {"bull": BULL, "sideways": SIDEWAYS, "bear": BEAR}
    for _, span in spans.iterrows():
        ax.axvspan(span["start"], span["end"],
                   color=regime_color[span["regime_name"]],
                   alpha=0.55, linewidth=0)

    # S&P 500 line
    ax.plot(df["date"], df["rebased"],
            color=LINE, linewidth=1.8, alpha=1.0, zorder=5)

    # Annotate key bear events (xy on the line, text above the line)
    def value_at(date):
        return float(df.loc[df["date"] >= date, "rebased"].iloc[0])

    annotations = [
        ("2008 GFC",       pd.Timestamp("2008-11-15"), 40),
        ("2020 COVID",     pd.Timestamp("2020-03-23"), 70),
        ("2022 rate-hike", pd.Timestamp("2022-10-15"), 80),
    ]
    for label, when, y_offset in annotations:
        y_pt = value_at(when)
        ax.annotate(
            label,
            xy=(when, y_pt),
            xytext=(when, y_pt + y_offset),
            ha="center", va="bottom",
            color=TEXT, fontsize=12, fontweight="bold",
            zorder=6,
            arrowprops=dict(arrowstyle="-", color=TEXT,
                            linewidth=1.2, alpha=0.85),
        )

    # Axes styling
    ax.set_xlim(df["date"].min(), df["date"].max())
    ax.set_ylim(40, df["rebased"].max() * 1.05)

    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", colors=SUBTEXT, labelsize=11, length=4)
    ax.tick_params(axis="y", colors=SUBTEXT, labelsize=10, length=3)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color(SUBTEXT)
        ax.spines[spine].set_linewidth(0.7)

    ax.set_ylabel("S&P 500  (rebased to 100)",
                  color=SUBTEXT, fontsize=11)

    # Title and subtitle inside the figure
    fig.text(
        0.5, 0.95,
        "HMM regime classification  ·  S&P 500, 2007–2024",
        ha="center", va="center",
        color=TITLE, fontsize=16, fontweight="bold",
    )

    # Custom legend with regime swatches (drawn manually)
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=BULL,     alpha=0.85, edgecolor="none", label="Bull"),
        Patch(facecolor=SIDEWAYS, alpha=0.85, edgecolor="none", label="Sideways"),
        Patch(facecolor=BEAR,     alpha=0.85, edgecolor="none", label="Bear"),
    ]
    leg = ax.legend(
        handles=legend_handles,
        loc="upper left", frameon=False,
        fontsize=12, handlelength=1.8, handleheight=1.2,
        borderaxespad=0.6,
    )
    for text in leg.get_texts():
        text.set_color(TEXT)
        text.set_fontweight("bold")

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    out_path = proj / "slide9_hmm_regimes.png"
    plt.savefig(out_path, dpi=220, bbox_inches="tight",
                facecolor=BG, pad_inches=0.15)
    plt.close(fig)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
