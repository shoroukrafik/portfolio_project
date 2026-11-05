"""Generate the 4-card ablation infographic for Slide 23.

Each card carries:
  - numbered badge
  - component name + role subtitle
  - per-market breakdown (DJI / SP100 / NDX) — short metric lines
  - one-line takeaway

Solid navy background to match the Canva template.
"""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image


# Palette
BG = "#0E1B3A"
TEXT = "white"
TITLE = "#7DC8FF"
SUBTEXT = "#C9D5EA"
CARD_FACE = "#16264F"
ACCENT_CYAN = "#7DC8FF"
ACCENT_AMBER = "#FFC97A"
ACCENT_GREEN = "#3FBF7F"
ACCENT_RED = "#E66060"
ROW_GREY = "#A4B0C8"


CARDS = [
    {
        "badge": "1",
        "name": "CASH-FLOOR OVERLAY",
        "subtitle": "the safety mechanism",
        "accent": ACCENT_GREEN,
        "rows": [
            ("DJI",   "0% → 6 / 8 wins  ·  70% → 8 / 8  (canonical)"),
            ("SP100", "0% → 5 / 8 wins  ·  30% picked by Sharpe tiebreak"),
            ("NDX",   "0% (canonical) — any positive floor reduces wins"),
        ],
        "takeaway": "Strict Pareto improvement on DJI and SP100.\nNDX defends through architecture alone.",
    },
    {
        "badge": "2",
        "name": "REGIME-CONDITIONED λ",
        "subtitle": "the regime-aware reward",
        "accent": ACCENT_CYAN,
        "rows": [
            ("DJI",   "bear  −10.8%  →  −14.2%   (3pp worse with uniform λ)"),
            ("SP100", "bear   −7.6%  →  −10.6%   (3pp worse with uniform λ)"),
        ],
        "takeaway": "Specifically improves bear defence on both primary markets.",
    },
    {
        "badge": "3",
        "name": "MULTI-HEAD ATTENTION",
        "subtitle": "the architecture",
        "accent": ACCENT_AMBER,
        "rows": [
            ("DJI",   "val  −8pp  ·  recovery  −10pp  ·  bull  −9pp"),
            ("SP100", "val  ≈ unchanged  ·  bear  −2pp  ·  bull  +2.5pp"),
        ],
        "takeaway": "Load-bearing on DJI; much smaller on SP100.\nEncoder-capacity story comes next (RQ3).",
    },
    {
        "badge": "4",
        "name": "LONG-MOMENTUM FEATURES",
        "subtitle": "the observation enrichment",
        "accent": ACCENT_RED,
        "rows": [
            ("DJI",   "+8pp recovery  ·  −5pp bear"),
            ("SP100", "+8pp recovery / val  ·  −5pp bear"),
        ],
        "takeaway": "Trades bear for bull capture.\nCash floor compensates the bear cost exactly.",
    },
]


def main():
    fig = plt.figure(figsize=(18, 12), dpi=240)
    fig.patch.set_facecolor(BG)
    ax = fig.add_subplot(111)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("auto")
    ax.axis("off")

    # ----------------------------- Title ------------------------------
    ax.text(
        50, 96,
        "ABLATIONS  ·  WHAT EACH COMPONENT CONTRIBUTES",
        fontsize=22, color=TITLE, ha="center", va="center",
        fontweight="bold", family="sans-serif",
    )
    ax.text(
        50, 91,
        "removing each piece, one at a time — per market",
        fontsize=13, color=SUBTEXT, ha="center", va="center",
        style="italic", family="sans-serif",
    )

    # ----------------------------- Cards ------------------------------
    # 2x2 grid — taller cards now that each has per-market rows
    positions = [
        (25, 62),   # top-left
        (75, 62),   # top-right
        (25, 25),   # bottom-left
        (75, 25),   # bottom-right
    ]
    card_w, card_h = 45, 31

    for (cx, cy), card in zip(positions, CARDS):
        # Card border
        b = FancyBboxPatch(
            (cx - card_w / 2, cy - card_h / 2), card_w, card_h,
            boxstyle="round,pad=0.6,rounding_size=1.0",
            facecolor=CARD_FACE, edgecolor=card["accent"],
            linewidth=2.4, alpha=0.95,
        )
        ax.add_patch(b)

        # Numbered badge (circle in top-left corner, accent colour)
        badge_x = cx - card_w / 2 + 4.0
        badge_y = cy + card_h / 2 - 3.5
        circle = plt.Circle((badge_x, badge_y), 1.7,
                            facecolor=card["accent"], edgecolor=card["accent"],
                            linewidth=1.5, zorder=5)
        ax.add_patch(circle)
        ax.text(badge_x, badge_y, card["badge"],
                fontsize=15, color=BG, ha="center", va="center",
                fontweight="bold", family="sans-serif", zorder=6)

        # Component name (centred horizontally in card)
        ax.text(cx + 2.0, cy + card_h / 2 - 3.0, card["name"],
                fontsize=14, color=card["accent"], ha="center", va="center",
                fontweight="bold", family="sans-serif")
        # Subtitle (smaller, below name)
        ax.text(cx + 2.0, cy + card_h / 2 - 6.2, card["subtitle"],
                fontsize=10, color=SUBTEXT, ha="center", va="center",
                style="italic", family="sans-serif")

        # ---- Per-market rows ----
        row_label_x = cx - card_w / 2 + 4.5
        row_value_x = cx - card_w / 2 + 13.0
        # Centre the rows vertically in the card based on how many rows there are
        n_rows = len(card["rows"])
        row_step = 4.2
        rows_block_height = (n_rows - 1) * row_step
        row_top_y = cy + (rows_block_height / 2) - 1.0
        for i, (mkt, metric) in enumerate(card["rows"]):
            y = row_top_y - i * row_step
            ax.text(row_label_x, y, mkt,
                    fontsize=12, color=TEXT, ha="left", va="center",
                    fontweight="bold", family="sans-serif")
            ax.text(row_value_x, y, metric,
                    fontsize=11, color=TEXT, ha="left", va="center",
                    family="sans-serif")

        # Takeaway (bottom of card, white for readability, multi-line if needed)
        ax.text(cx, cy - card_h / 2 + 4.5, card["takeaway"],
                fontsize=10.5, color=TEXT, ha="center", va="center",
                style="italic", family="sans-serif",
                linespacing=1.3)

    # ---------------------- Bottom verdict line -----------------------
    ax.text(
        50, 6.0,
        "The clean separation between LEARNED POLICY and RULE-BASED OVERLAY",
        fontsize=12, color=TITLE, ha="center", va="center",
        fontweight="bold", family="sans-serif",
    )
    ax.text(
        50, 2.5,
        "is what enables RQ2 — the overlay is a Pareto improvement on top of an already-strong policy.",
        fontsize=11, color=SUBTEXT, ha="center", va="center",
        style="italic", family="sans-serif",
    )

    # ---------------------- Save and flatten alpha --------------------
    out_path = Path(__file__).parent / "slide23_ablations.png"
    plt.savefig(out_path, dpi=240, facecolor=BG, edgecolor=BG,
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
