"""Generate a transparent PNG figure for Slide 8: 'What the agent sees each day'.

Navy + cyan palette matched to the user's Canva template.
Layout: four input boxes on the left feeding arrows into a central OBSERVATION
node on the right, with the total dimension printed below.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Palette tuned to the user's navy/cyan template
TEXT = "white"
TITLE = "#7DC8FF"            # bright cyan for headers/labels
ACCENT = "#7DC8FF"
BOX_FACE = "#0E1B3A"         # deep navy (low alpha so it sits softly over a navy slide)
BOX_EDGE = "#7DC8FF"         # cyan border
ARROW = "#7DC8FF"
SUBTEXT = "#C9D5EA"          # soft blue-grey for body copy


def main():
    fig = plt.figure(figsize=(18, 10), dpi=200)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 56.25)
    ax.set_aspect("equal")
    ax.axis("off")

    # ------------------------------ Title ------------------------------
    ax.text(
        50, 53.5,
        "WHAT THE AGENT SEES EACH DAY",
        fontsize=24, color=TITLE, ha="center", va="center",
        fontweight="bold", family="sans-serif",
    )

    # ---------------------- Source box helpers ------------------------
    def draw_source_box(x, y, w, h, header, body_lines, header_size=12, body_size=10):
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.4,rounding_size=0.8",
            facecolor=BOX_FACE, edgecolor=BOX_EDGE,
            linewidth=2.2, alpha=0.85,
        )
        ax.add_patch(box)
        ax.text(
            x + w / 2, y + h - 1.4,
            header,
            fontsize=header_size, color=TITLE, ha="center", va="top",
            fontweight="bold", family="sans-serif",
        )
        # body
        line_step = (h - 4.0) / max(len(body_lines), 1)
        start_y = y + h - 3.5
        for i, line in enumerate(body_lines):
            ax.text(
                x + 1.2, start_y - (i + 0.5) * line_step,
                line,
                fontsize=body_size, color=SUBTEXT, ha="left", va="center",
                family="sans-serif",
            )

    # Source box positions
    box_x = 4
    box_w = 33

    # 1) Per-stock features box (largest, top)
    draw_source_box(
        box_x, 33, box_w, 16,
        "PER STOCK   (22 × N features)",
        [
            " 5   prices & volume  (OHLCV)",
            "12   technical indicators",
            "        (RSI, MACD, Bollinger, ATR, SMA, ADX, CCI)",
            " 2   daily return & volatility",
            " 3   long-horizon momentum  (60d, 120d, 252d)",
        ],
        header_size=13, body_size=10.0,
    )

    # 2) Broad-market vol
    draw_source_box(
        box_x, 24, box_w, 7,
        "BROAD-MARKET VOLATILITY   (3)",
        ["vol_20d, vol_60d, vol_20/60 ratio"],
        header_size=12, body_size=10.5,
    )

    # 3) HMM regime info
    draw_source_box(
        box_x, 14.5, box_w, 7.5,
        "HMM REGIME   (6)",
        ["3  hard regime label (one-hot)",
         "3  posterior probabilities"],
        header_size=12, body_size=10.5,
    )

    # 4) Portfolio weights
    draw_source_box(
        box_x, 5, box_w, 7,
        "PORTFOLIO WEIGHTS   (N+1)",
        ["current weights (lets the agent factor in turnover cost)"],
        header_size=12, body_size=10.5,
    )

    # ------------------------- Observation node -----------------------
    obs_x, obs_y, obs_w, obs_h = 58, 18, 36, 24
    obs_box = FancyBboxPatch(
        (obs_x, obs_y), obs_w, obs_h,
        boxstyle="round,pad=0.6,rounding_size=1.2",
        facecolor=BOX_FACE, edgecolor=BOX_EDGE,
        linewidth=3.0, alpha=0.95,
    )
    ax.add_patch(obs_box)

    ax.text(
        obs_x + obs_w / 2, obs_y + obs_h - 3.4,
        "OBSERVATION",
        fontsize=20, color=TITLE, ha="center", va="center",
        fontweight="bold", family="sans-serif",
    )
    ax.text(
        obs_x + obs_w / 2, obs_y + obs_h - 6.6,
        r"$\mathbf{o}_t$",
        fontsize=18, color=TEXT, ha="center", va="center", style="italic",
    )
    ax.text(
        obs_x + obs_w / 2, obs_y + obs_h / 2 - 0.5,
        "22N + 10",
        fontsize=30, color=TITLE, ha="center", va="center",
        fontweight="bold", family="sans-serif",
    )
    ax.text(
        obs_x + obs_w / 2, obs_y + obs_h / 2 - 5.0,
        "features per day",
        fontsize=12, color=SUBTEXT, ha="center", va="center",
        family="sans-serif",
    )
    # tiny dim summary inside the obs box
    ax.text(
        obs_x + obs_w / 2, obs_y + 3.4,
        "→  700 (DJI)    2,080 (SP100, NDX)",
        fontsize=11, color=SUBTEXT, ha="center", va="center",
        family="sans-serif",
    )

    # ----------------------------- Arrows -----------------------------
    def arrow(x1, y1, x2, y2, rad=0.0):
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1), (x2, y2),
                arrowstyle="-|>,head_width=0.45,head_length=0.7",
                color=ARROW, linewidth=2.2,
                connectionstyle=f"arc3,rad={rad}",
                shrinkA=2, shrinkB=2,
            )
        )

    # arrows from each source box to observation box (entering left edge)
    src_right = box_x + box_w
    obs_left = obs_x
    arrow(src_right, 41.0, obs_left, 37.0, rad=0.05)   # per-stock
    arrow(src_right, 27.5, obs_left, 32.5, rad=0.04)   # broad-market
    arrow(src_right, 18.0, obs_left, 27.5, rad=-0.04)  # hmm
    arrow(src_right, 8.5,  obs_left, 22.5, rad=-0.06)  # portfolio

    # arrow out of observation node toward "the agent"
    out_x = obs_x + obs_w / 2
    arrow(out_x, obs_y, out_x, obs_y - 5.0, rad=0.0)
    ax.text(
        out_x, obs_y - 6.6,
        "to the agent",
        fontsize=12, color=SUBTEXT, ha="center", va="center",
        style="italic", family="sans-serif",
    )

    # ----------------------------- Footer -----------------------------
    ax.text(
        50, 1.5,
        "Total observation dimension = 22N + 10    ·    DJI: 700    ·    SP100 & NDX: 2,080",
        fontsize=12, color=TEXT, ha="center", va="center",
        family="sans-serif",
    )

    # ------------------------------ Save ------------------------------
    out_path = "/Users/shoroukrafeek/Desktop/portfolio_project/slide8_observation.png"
    plt.savefig(out_path, dpi=220, bbox_inches="tight", transparent=True, pad_inches=0.1)
    plt.close(fig)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
