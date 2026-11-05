"""Generate the robustness infographic for Slide 26 (V4).

Same as V3 (MODEL ROBUSTNESS title + V2 substance) EXCEPT Card 2 is redesigned:
- Card 2 is now organized around WHAT EACH METRIC ADDRESSES, not market-by-market numbers.
  · SORTINO asks: is risk ASYMMETRIC? → YES, with the evidence
  · CALMAR  asks: return > worst drawdown? → YES, with the magnitude
  · BEAR caveat as third row (mathematical artifact, not failure)

Cards 1, 3, 4 unchanged from V3.

Output: slide26_robustness_v4.png  (does NOT overwrite v1 / v2 / v3)
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
        "name": "MARKOWITZ BASELINES",
        "subtitle": "classical portfolio optimization  ·  GMV  +  Max-Sharpe per period",
        "accent": ACCENT_CYAN,
        "rows": [
            ("DJI",   "agent wins every period vs BOTH baselines"),
            ("SP100", "wins val / recovery / bull  ·  loses bear to GMV"),
            ("NDX",   "wins val + bear  ·  loses recovery / bull"),
        ],
        "takeaway": "Universe-conditional story holds vs classical baselines.",
    },
    {
        "badge": "2",
        "name": "SORTINO  +  CALMAR",
        "subtitle": "two downside-focused lenses  ·  each addresses a different question",
        "accent": ACCENT_AMBER,
        "rows": [
            ("SORTINO", "is risk ASYMMETRIC?   →   YES, > Sharpe in every positive OOS"),
            ("CALMAR",  "return > worst drawdown?   →   YES, 1.7 – 8.3× across markets"),
            ("BEAR",    "negative by math (loss ÷ vol) — not a performance failure"),
        ],
        "takeaway": "Both downside-aware lenses confirm the headline:\nrisk IS asymmetric  AND  returns exceed worst drawdown.",
    },
    {
        "badge": "3",
        "name": "BOOTSTRAP  CONFIDENCE INTERVALS",
        "subtitle": "resample returns 10,000×  →  is the Sharpe reliably > 0, or could it be luck?",
        "accent": ACCENT_GREEN,
        "rows": [
            ("DJI",   "+56% (index +16%)   ·   CI [+0.03, +2.40]   excludes zero  ✓"),
            ("SP100", "+27% (EW +27%)   ·   CI overlaps zero"),
            ("NDX",   "+25% (index +32%)   ·   MDD  −23%  vs index  −35%"),
        ],
        "takeaway": "DJI is STATISTICALLY SIGNIFICANT.\nSP100 + NDX are directionally consistent.",
    },
    {
        "badge": "4",
        "name": "TRANSACTION-COST REALISM",
        "subtitle": "10 bps per dollar traded  →  all headline returns are NET, not gross",
        "accent": ACCENT_RED,
        "rows": [
            ("DJI",   "turnover 18.5×  ·  annual cost  1.85%"),
            ("SP100", "turnover 15.7×  ·  annual cost  1.57%"),
            ("NDX",   "turnover  3.9×  ·  annual cost  0.39%"),
        ],
        "takeaway": "Within active-management norms.\nNDX barely trades — defensive by design.",
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
        "MODEL ROBUSTNESS",
        fontsize=32, color=TITLE, ha="center", va="center",
        fontweight="bold", family="sans-serif",
    )
    ax.text(
        50, 91.5,
        "Markowitz  ·  Sortino + Calmar  ·  Bootstrap CIs  ·  Transaction-cost realism",
        fontsize=14, color=SUBTEXT, ha="center", va="center",
        style="italic", family="sans-serif",
    )

    # ----------------------------- Cards ------------------------------
    positions = [
        (25, 62),   # top-left
        (75, 62),   # top-right
        (25, 25),   # bottom-left
        (75, 25),   # bottom-right
    ]
    card_w, card_h = 48, 33

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

        # Card name (centred)
        ax.text(cx + 2.0, cy + card_h / 2 - 4.0, card["name"],
                fontsize=16, color=card["accent"],
                ha="center", va="center",
                fontweight="bold", family="sans-serif")
        # Subtitle — DEFINITION of the check (may be multi-line)
        ax.text(cx, cy + card_h / 2 - 9.0, card["subtitle"],
                fontsize=11.5, color=SUBTEXT, ha="center", va="center",
                style="italic", family="sans-serif",
                linespacing=1.4)

        # Per-row content
        row_label_x = cx - card_w / 2 + 4.0
        row_value_x = cx - card_w / 2 + 13.5
        n_rows = len(card["rows"])
        row_step = 4.4
        rows_block_height = (n_rows - 1) * row_step
        row_top_y = cy + (rows_block_height / 2) - 1.0
        for i, (label, value) in enumerate(card["rows"]):
            y = row_top_y - i * row_step
            # Label (market name OR metric name)
            ax.text(row_label_x, y, label,
                    fontsize=13.5, color=TEXT, ha="left", va="center",
                    fontweight="bold", family="sans-serif")
            # Value
            ax.text(row_value_x, y, value,
                    fontsize=11.5, color=TEXT, ha="left", va="center",
                    family="sans-serif")

        # Takeaway (bottom of card)
        ax.text(cx, cy - card_h / 2 + 5.0, card["takeaway"],
                fontsize=12.5, color=TEXT, ha="center", va="center",
                style="italic", family="sans-serif",
                linespacing=1.3)

    # ---------------------- Bottom verdict line -----------------------
    ax.text(
        50, 5.5,
        "Four independent checks  ·  same verdict:  strong on DJI, partial on SP100, defensive trade on NDX.",
        fontsize=15, color=TITLE, ha="center", va="center",
        fontweight="bold", family="sans-serif",
    )
    ax.text(
        50, 2.0,
        "Classical, statistical, and downside-aware lenses all agree with the headline.",
        fontsize=13, color=SUBTEXT, ha="center", va="center",
        style="italic", family="sans-serif",
    )

    # ---------------------- Save and flatten alpha --------------------
    out_path = Path(__file__).parent / "slide26_robustness_v4.png"
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
