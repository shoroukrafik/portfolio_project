"""Generate a navy-background figure for Slide 12: the canonical-model rule.

Top half — the 2-step selection pipeline (filter -> pick best val Sharpe ->
sweep cash floor -> canonical model).
Bottom half — per-market outcome (DJI 70% / SP100 30% / NDX no floor).

Designed to land the "same rule, different floors emerge automatically"
message in 5-7 seconds.
"""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


BG = "#0E1B3A"
TEXT = "white"
TITLE = "#7DC8FF"
SUBTEXT = "#C9D5EA"
NEUTRAL_FACE = "#16264F"
ACCENT_FACE = "#26477E"
EDGE = "#7DC8FF"
BULL = "#3FBF7F"
BEAR = "#E66060"
SIDEWAYS_GREY = "#8E9DBC"


def main():
    fig = plt.figure(figsize=(16, 10), dpi=200)
    ax = fig.add_subplot(111)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("auto")
    ax.axis("off")

    # ----------------------------- Title ------------------------------
    ax.text(
        50, 94,
        "THE CANONICAL MODEL",
        fontsize=22, color=TITLE, ha="center", va="center",
        fontweight="bold", family="sans-serif",
    )
    ax.text(
        50, 88,
        "A deterministic 2-step rule — same training run produces the same model",
        fontsize=13, color=SUBTEXT, ha="center", va="center",
        style="italic", family="sans-serif",
    )

    # --------------------------- Helpers ------------------------------
    def step_box(x, y, w, h, header, body_lines,
                 face=NEUTRAL_FACE, header_size=14, body_size=11,
                 highlight=False):
        edge = EDGE
        lw = 2.0
        b = FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.4,rounding_size=0.8",
            facecolor=face, edgecolor=edge, linewidth=lw, alpha=0.95,
        )
        ax.add_patch(b)
        ax.text(x, y + h / 2 - 2.6, header,
                fontsize=header_size, color=TITLE if not highlight else TEXT,
                ha="center", va="center",
                fontweight="bold", family="sans-serif")
        for i, line in enumerate(body_lines):
            ax.text(x, y + h / 2 - 5.8 - i * 2.6, line,
                    fontsize=body_size, color=SUBTEXT,
                    ha="center", va="center", family="sans-serif")

    def arrow(x1, y1, x2, y2):
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1), (x2, y2),
                arrowstyle="-|>,head_width=0.5,head_length=0.8",
                color=EDGE, linewidth=2.2,
                connectionstyle="arc3,rad=0",
                shrinkA=2, shrinkB=2,
            )
        )

    # ----------------------- Top: 2-step flow -------------------------
    flow_y = 68
    box_w, box_h = 23, 16

    step_box(
        15, flow_y, box_w, box_h,
        "STEP 1",
        [
            "Pick the DEFENSIVE",
            "checkpoint:",
            "best val Sharpe",
            "at ≥20% of training",
        ],
    )

    step_box(
        45, flow_y, box_w, box_h,
        "STEP 2",
        [
            "Sweep cash floor",
            "f ∈ {0, 30, 50, 70%}",
            "Pick the value that",
            "maximises wins",
        ],
    )

    step_box(
        75, flow_y, box_w, box_h,
        "CANONICAL",
        [
            "fully reproducible",
            "checkpoint",
            "+",
            "regime-conditioned",
            "cash floor",
        ],
        face=ACCENT_FACE,
        highlight=True,
    )

    # Arrows between flow boxes
    arrow(15 + box_w / 2, flow_y, 45 - box_w / 2, flow_y)
    arrow(45 + box_w / 2, flow_y, 75 - box_w / 2, flow_y)

    # ------------- Bottom: per-market outcome cards -------------------
    market_y = 30
    market_w, market_h = 22, 22
    market_xs = [20, 50, 80]

    # Section header above market cards
    ax.text(
        50, 50,
        "SAME RULE — APPLIED TO EACH MARKET — PRODUCES DIFFERENT FLOORS",
        fontsize=13, color=TITLE, ha="center", va="center",
        fontweight="bold", family="sans-serif",
    )

    markets = [
        ("DJI 30",      "70%",    "heavily defensive", BULL,            "30 stocks"),
        ("SP100",       "30%",    "moderate",          SIDEWAYS_GREY,   "90 stocks"),
        ("NDX 2017",    "no floor", "no overlay",      BEAR,            "90 stocks"),
    ]

    for cx, (mkt, floor, label, color, size) in zip(market_xs, markets):
        # market card
        rect = FancyBboxPatch(
            (cx - market_w / 2, market_y - market_h / 2),
            market_w, market_h,
            boxstyle="round,pad=0.4,rounding_size=0.8",
            facecolor=NEUTRAL_FACE, edgecolor=color,
            linewidth=2.4, alpha=0.95,
        )
        ax.add_patch(rect)

        # market name
        ax.text(cx, market_y + 7.8, mkt,
                fontsize=18, color=color, ha="center", va="center",
                fontweight="bold", family="sans-serif")
        # subtle size note
        ax.text(cx, market_y + 4.7, size,
                fontsize=10, color=SUBTEXT, ha="center", va="center",
                family="sans-serif")
        # floor value (the headline)
        ax.text(cx, market_y - 1.0, floor,
                fontsize=26, color=TEXT, ha="center", va="center",
                fontweight="bold", family="sans-serif")
        ax.text(cx, market_y - 5.8, "bear cash floor",
                fontsize=10, color=SUBTEXT, ha="center", va="center",
                family="sans-serif")
        # behaviour label at the bottom
        ax.text(cx, market_y - 8.6, label,
                fontsize=11, color=color, ha="center", va="center",
                style="italic", family="sans-serif")

    # --------------------- Footer key-message line --------------------
    ax.text(
        50, 11.5,
        "the rule is universal — the market decides what floor it needs",
        fontsize=12, color=SUBTEXT, ha="center", va="center",
        style="italic", family="sans-serif",
    )
    ax.text(
        50, 6.5,
        "addresses Gap 2: anyone running the same training reproduces the same model",
        fontsize=11, color=TITLE, ha="center", va="center",
        family="sans-serif",
    )

    # ---------------------- Save --------------------------------------
    out_path = Path(__file__).parent / "slide12_canonical.png"
    plt.savefig(out_path, dpi=220, bbox_inches="tight",
                facecolor=BG, pad_inches=0.25)
    plt.close(fig)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
