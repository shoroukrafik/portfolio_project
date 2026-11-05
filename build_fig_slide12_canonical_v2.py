"""Generate the canonical-model figure for Slide 12 (V2).

Improvements over V1:
- New title:  "THE CANONICAL MODEL  —  ONE RULE, THREE FLOORS"
  (captures the deterministic-pipeline AND the universe-adaptive-floor story)
- 16:9 landscape (19.2 × 10.8) — matches the rest of the Canva deck
- NDX card now shows "0%" + "no overlay" caption (consistent with other cards)
- Color coding: DJI green / SP100 cyan / NDX AMBER (red implied "bad" — wrong message)
- Bigger fonts throughout for projector readability
- Cleaner step boxes (2-3 lines max)
- Footer reworded — no jargon

Output: slide12_canonical_v2.png  (does NOT overwrite slide12_canonical.png)
"""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image


BG = "#0E1B3A"
TEXT = "white"
TITLE = "#7DC8FF"
SUBTEXT = "#C9D5EA"
NEUTRAL_FACE = "#16264F"
ACCENT_FACE = "#26477E"
EDGE = "#7DC8FF"
DJI_COLOR = "#3FBF7F"
SP100_COLOR = "#7DC8FF"
NDX_COLOR = "#FFC97A"


def main():
    fig = plt.figure(figsize=(19.2, 10.8), dpi=200)
    ax = fig.add_subplot(111)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("auto")
    ax.axis("off")

    # ----------------------------- Title ------------------------------
    ax.text(
        50, 95,
        "MODEL SELECTION",
        fontsize=32, color=TITLE, ha="center", va="center",
        fontweight="bold", family="sans-serif",
    )
    ax.text(
        50, 90,
        "deterministic 2-step pipeline  ·  same training run reproduces the same model",
        fontsize=16, color=SUBTEXT, ha="center", va="center",
        style="italic", family="sans-serif",
    )

    # --------------------------- Helpers ------------------------------
    def step_box(x, y, w, h, header, body_lines,
                 face=NEUTRAL_FACE, accent=EDGE,
                 header_size=17, body_size=13,
                 header_color=None):
        b = FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.5,rounding_size=0.9",
            facecolor=face, edgecolor=accent, linewidth=2.4, alpha=0.95,
        )
        ax.add_patch(b)
        ax.text(x, y + h / 2 - 3.0, header,
                fontsize=header_size,
                color=header_color if header_color else accent,
                ha="center", va="center",
                fontweight="bold", family="sans-serif")
        # body lines spaced from top down
        for i, line in enumerate(body_lines):
            ax.text(x, y + h / 2 - 7.0 - i * 3.0, line,
                    fontsize=body_size, color=SUBTEXT,
                    ha="center", va="center", family="sans-serif")

    def arrow(x1, y1, x2, y2, color=EDGE, lw=2.4):
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1), (x2, y2),
                arrowstyle="-|>,head_width=0.5,head_length=0.8",
                color=color, linewidth=lw,
                connectionstyle="arc3,rad=0",
                shrinkA=2, shrinkB=2,
            )
        )

    # ----------------------- Top: 2-step flow -------------------------
    flow_y = 73
    box_w, box_h = 24, 18

    step_box(
        15, flow_y, box_w, box_h,
        "STEP 1",
        [
            "Pick DEFENSIVE",
            "checkpoint",
            "best val Sharpe at ≥ 20%",
        ],
        header_size=22, body_size=15,
    )

    step_box(
        45, flow_y, box_w, box_h,
        "STEP 2",
        [
            "Sweep cash floor",
            "f ∈ {0, 30, 50, 70%}",
            "Pick best by wins",
        ],
        header_size=22, body_size=15,
    )

    step_box(
        75, flow_y, box_w, box_h,
        "FINAL MODEL",
        [
            "checkpoint  +  cash floor",
            "fully reproducible",
        ],
        face=ACCENT_FACE,
        header_color=TEXT,
        header_size=22, body_size=15,
    )

    # Horizontal arrows
    arrow(15 + box_w / 2 + 0.5, flow_y, 45 - box_w / 2 - 0.5, flow_y)
    arrow(45 + box_w / 2 + 0.5, flow_y, 75 - box_w / 2 - 0.5, flow_y)

    # ------------- Middle: transition header --------------------------
    ax.text(
        50, 58,
        "SAME RULE  ·  APPLIED TO EACH MARKET  ·  DIFFERENT FLOORS EMERGE",
        fontsize=18, color=TITLE, ha="center", va="center",
        fontweight="bold", family="sans-serif",
    )

    # Down arrow
    arrow(50, 54, 50, 50, lw=2.6)

    # ------------- Bottom: per-market outcome cards -------------------
    market_y = 30
    market_w, market_h = 24, 32
    market_xs = [18, 50, 82]

    markets = [
        {
            "name": "DJI 30",
            "size": "30 stocks  ·  low dispersion",
            "floor": "70%",
            "floor_caption": "bear cash floor",
            "behavior": "heavily defensive",
            "color": DJI_COLOR,
        },
        {
            "name": "S&P 100",
            "size": "90 stocks  ·  moderate dispersion",
            "floor": "30%",
            "floor_caption": "bear cash floor",
            "behavior": "moderately defensive",
            "color": SP100_COLOR,
        },
        {
            "name": "NDX 100",
            "size": "90 stocks  ·  high dispersion",
            "floor": "0%",
            "floor_caption": "no overlay  (architecture defends)",
            "behavior": "no floor needed",
            "color": NDX_COLOR,
        },
    ]

    for cx, m in zip(market_xs, markets):
        rect = FancyBboxPatch(
            (cx - market_w / 2, market_y - market_h / 2),
            market_w, market_h,
            boxstyle="round,pad=0.5,rounding_size=1.0",
            facecolor=NEUTRAL_FACE, edgecolor=m["color"],
            linewidth=2.8, alpha=0.95,
        )
        ax.add_patch(rect)

        # market name
        ax.text(cx, market_y + 12.0, m["name"],
                fontsize=26, color=m["color"], ha="center", va="center",
                fontweight="bold", family="sans-serif")
        # size note
        ax.text(cx, market_y + 8.0, m["size"],
                fontsize=13, color=SUBTEXT, ha="center", va="center",
                style="italic", family="sans-serif")

        # floor value — the headline number, big
        ax.text(cx, market_y - 0.5, m["floor"],
                fontsize=50, color=TEXT, ha="center", va="center",
                fontweight="bold", family="sans-serif")
        # floor caption
        ax.text(cx, market_y - 8.0, m["floor_caption"],
                fontsize=13, color=SUBTEXT, ha="center", va="center",
                family="sans-serif")
        # behavior label
        ax.text(cx, market_y - 12.5, m["behavior"],
                fontsize=16, color=m["color"], ha="center", va="center",
                style="italic", fontweight="bold", family="sans-serif")

    # --------------------- Footer key-message line --------------------
    ax.text(
        50, 9,
        "The rule is universal  ·  the MARKET decides what floor it needs",
        fontsize=17, color=TITLE, ha="center", va="center",
        fontweight="bold", family="sans-serif",
    )
    ax.text(
        50, 4.5,
        "fully reproducible — anyone running the same training reproduces the same model",
        fontsize=13, color=SUBTEXT, ha="center", va="center",
        style="italic", family="sans-serif",
    )

    # ---------------------- Save and flatten alpha --------------------
    out_path = Path(__file__).parent / "slide12_canonical_v2.png"
    plt.savefig(out_path, dpi=200, facecolor=BG, edgecolor=BG,
                transparent=False, pad_inches=0.20)
    plt.close(fig)

    img = Image.open(out_path).convert("RGBA")
    bg_rgb = tuple(int(BG[i:i + 2], 16) for i in (1, 3, 5))
    canvas = Image.new("RGB", img.size, bg_rgb)
    canvas.paste(img, mask=img.split()[-1])
    canvas.save(out_path, format="PNG")

    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
