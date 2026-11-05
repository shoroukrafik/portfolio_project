"""Generate the 4-card ablation infographic for Slide 23 (V2).

Improvements over V1:
- Cards 2, 3, 4 have explicit EXPERIMENTAL-SETUP subtitles
  (what was replaced, what it was replaced WITH, baseline values)
- Three OOS regimes per market shown: BEAR · RECOVERY · BULL
- Direction is explicit: +pp = HELPS canonical, -pp = HURTS canonical
- Numbers verified against the actual final_results.json files

Output: slide23_ablations_v2.png  (does NOT overwrite slide23_ablations.png)
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


CARDS = [
    {
        "badge": "1",
        "name": "CASH-FLOOR OVERLAY",
        "subtitle": "applied only when HMM says BEAR",
        "accent": ACCENT_GREEN,
        "rows": [
            ("DJI",   "0% → 6/8 wins  ·  70% → 8/8 (canonical)"),
            ("SP100", "0% → 5/8 wins  ·  30% → 5/8 canonical (Sharpe tiebreak)"),
            ("NDX",   "0% (canonical) — any floor REDUCES wins"),
        ],
        "takeaway": "8/8 wins on DJI with the floor.\nSP100 floor picked by Sharpe tiebreak  ·  NDX defends through architecture.",
    },
    {
        "badge": "2",
        "name": "REGIME-CONDITIONED  λ",
        "subtitle": "regime-λ (0.1 / 0.3 / 0.5)  →  uniform λ = 0.3",
        "accent": ACCENT_CYAN,
        "rows": [
            ("DJI",   "bear −3pp  ·  bull −3pp  ·  recovery ≈"),
            ("SP100", "bear −3pp  ·  bull / recovery ≈"),
        ],
        "takeaway": "Removing regime-λ HURTS bear defense\non BOTH markets (~3pp worse).",
    },
    {
        "badge": "3",
        "name": "MULTI-HEAD ATTENTION",
        "subtitle": "attention block  →  plain MLP feature extractor",
        "accent": ACCENT_AMBER,
        "rows": [
            ("DJI",   "bear +3pp  ·  recovery −10pp  ·  bull −9pp"),
            ("SP100", "bear −2pp  ·  recovery ≈  ·  bull +2.5pp"),
        ],
        "takeaway": "Load-bearing on DJI for upside regimes.\nSP100 mixed → encoder-capacity hypothesis (RQ3).",
    },
    {
        "badge": "4",
        "name": "LONG-MOMENTUM FEATURES",
        "subtitle": "add 120-day  +  252-day normalized momentum to the observation",
        "accent": ACCENT_RED,
        "rows": [
            ("DJI",   "bear −5pp  ·  recovery +8pp  ·  bull +3pp"),
            ("SP100", "bear −5pp  ·  recovery +8pp  ·  bull  ↑"),
        ],
        "takeaway": "Trades BEAR defense for BULL / RECOVERY capture.\nCash floor compensates the bear cost exactly.",
    },
]


def main():
    fig = plt.figure(figsize=(19.2, 10.8), dpi=200)
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
        fontsize=26, color=TITLE, ha="center", va="center",
        fontweight="bold", family="sans-serif",
    )
    ax.text(
        50, 91.5,
        "removing each piece, one at a time  ·  3 OOS regimes:  BEAR · RECOVERY · BULL",
        fontsize=14, color=SUBTEXT, ha="center", va="center",
        style="italic", family="sans-serif",
    )

    # ----------------------------- Cards ------------------------------
    positions = [
        (24.5, 62),   # top-left
        (75.5, 62),   # top-right
        (24.5, 25),   # bottom-left
        (75.5, 25),   # bottom-right
    ]
    card_w, card_h = 48, 32

    for (cx, cy), card in zip(positions, CARDS):
        # Card border
        b = FancyBboxPatch(
            (cx - card_w / 2, cy - card_h / 2), card_w, card_h,
            boxstyle="round,pad=0.6,rounding_size=1.0",
            facecolor=CARD_FACE, edgecolor=card["accent"],
            linewidth=2.4, alpha=0.95,
        )
        ax.add_patch(b)

        # Numbered badge
        badge_x = cx - card_w / 2 + 4.5
        badge_y = cy + card_h / 2 - 4.0
        circle = plt.Circle((badge_x, badge_y), 2.2,
                            facecolor=card["accent"], edgecolor=card["accent"],
                            linewidth=1.5, zorder=5)
        ax.add_patch(circle)
        ax.text(badge_x, badge_y, card["badge"],
                fontsize=19, color=BG, ha="center", va="center",
                fontweight="bold", family="sans-serif", zorder=6)

        # Component name (centred)
        ax.text(cx + 2.0, cy + card_h / 2 - 3.5, card["name"],
                fontsize=16, color=card["accent"],
                ha="center", va="center",
                fontweight="bold", family="sans-serif")
        # Subtitle — experimental setup (one line, italic, lighter)
        ax.text(cx, cy + card_h / 2 - 8.0, card["subtitle"],
                fontsize=12, color=SUBTEXT, ha="center", va="center",
                style="italic", family="sans-serif")

        # ---- Per-market rows ----
        row_label_x = cx - card_w / 2 + 4.5
        row_value_x = cx - card_w / 2 + 13.0
        n_rows = len(card["rows"])
        row_step = 4.6
        # Centre the rows in the middle of the card
        rows_block_height = (n_rows - 1) * row_step
        row_top_y = cy + (rows_block_height / 2) - 1.0
        for i, (mkt, metric) in enumerate(card["rows"]):
            y = row_top_y - i * row_step
            ax.text(row_label_x, y, mkt,
                    fontsize=14, color=TEXT, ha="left", va="center",
                    fontweight="bold", family="sans-serif")
            ax.text(row_value_x, y, metric,
                    fontsize=13, color=TEXT, ha="left", va="center",
                    family="sans-serif")

        # Takeaway (bottom of card)
        ax.text(cx, cy - card_h / 2 + 5.0, card["takeaway"],
                fontsize=12, color=TEXT, ha="center", va="center",
                style="italic", family="sans-serif",
                linespacing=1.3)

    # ---------------------- Bottom verdict line -----------------------
    ax.text(
        50, 5.5,
        "Clean separation between LEARNED POLICY and RULE-BASED OVERLAY",
        fontsize=14, color=TITLE, ha="center", va="center",
        fontweight="bold", family="sans-serif",
    )
    ax.text(
        50, 1.8,
        "overlay is a Pareto improvement on an already-strong policy  ·  answers RQ2",
        fontsize=12, color=SUBTEXT, ha="center", va="center",
        style="italic", family="sans-serif",
    )

    # ---------------------- Save and flatten alpha --------------------
    out_path = Path(__file__).parent / "slide23_ablations_v2.png"
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
