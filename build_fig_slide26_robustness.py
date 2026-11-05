"""Generate the 4-card robustness infographic for Slide 26.

Same 2x2 card layout as Slide 23 ablations. Each card covers one
robustness check with a key highlight and a takeaway.
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
HYPO_FACE = "#26477E"
ACCENT_CYAN = "#7DC8FF"
ACCENT_AMBER = "#FFC97A"
ACCENT_GREEN = "#3FBF7F"
ACCENT_RED = "#E66060"
ROW_GREY = "#A4B0C8"


CARDS = [
    {
        "badge": "1",
        "name": "MARKOWITZ BASELINES",
        "subtitle": "agent vs GMV + Max-Sharpe (per period)",
        "accent": ACCENT_CYAN,
        "rows": [
            ("DJI",   "agent wins every period vs both Markowitz portfolios"),
            ("SP100", "wins val / recovery / bull · loses bear to GMV"),
            ("NDX",   "wins val + bear · loses 2023 + bull to Max-Sharpe (mega-cap)"),
        ],
        "takeaway": "Universe-conditional story holds\nagainst the classical baselines.",
    },
    {
        "badge": "2",
        "name": "SORTINO  +  CALMAR",
        "subtitle": "downside / drawdown risk-adjusted alternatives",
        "accent": ACCENT_AMBER,
        "rows": [
            ("DJI",   "val Sortino +2.76 (vs Sharpe +2.27)  ·  bull Calmar +5.70"),
            ("SP100", "val Sortino +4.48 (vs Sharpe +3.41)  ·  val Calmar +8.28"),
            ("NDX",   "val Sortino +2.50 (vs Sharpe +1.88)"),
        ],
        "takeaway": "Sortino > Sharpe on every non-bear period.\nAsymmetric reward works as designed.",
    },
    {
        "badge": "3",
        "name": "3-YEAR BACKTEST  +  BOOTSTRAP CIs",
        "subtitle": "compounded 2022–2024  ·  95% CI on Sharpe",
        "accent": ACCENT_GREEN,
        "rows": [
            ("DJI",   "+56% (vs index +16%)  ·  CI [+0.03, +2.40]  ←  excludes zero"),
            ("SP100", "+27% (vs EW +27%)  ·  CI overlaps zero"),
            ("NDX",   "+25% (vs index +32%)  ·  MDD smallest:  −23% vs −35%"),
        ],
        "takeaway": "DJI is statistically significant.\nSP100 and NDX directionally consistent.",
    },
    {
        "badge": "4",
        "name": "TRANSACTION-COST REALISM",
        "subtitle": "10 bps per unit of turnover  ·  net returns reported",
        "accent": ACCENT_RED,
        "rows": [
            ("DJI",   "annual turnover 18.5×  ·  annual cost 1.85%"),
            ("SP100", "annual turnover 15.7×  ·  annual cost 1.57%"),
            ("NDX",   "annual turnover 3.9×  ·  annual cost 0.39%"),
        ],
        "takeaway": "All headline returns are NET of these costs.\nWithin actively-managed strategy norms.",
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
        "ROBUSTNESS  ·  THE HEADLINE SURVIVES FOUR CHECKS",
        fontsize=24, color=TITLE, ha="center", va="center",
        fontweight="bold", family="sans-serif",
    )
    ax.text(
        50, 91,
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
    card_w, card_h = 47, 33

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
        badge_y = cy + card_h / 2 - 3.8
        circle = plt.Circle((badge_x, badge_y), 1.9,
                            facecolor=card["accent"], edgecolor=card["accent"],
                            linewidth=1.5, zorder=5)
        ax.add_patch(circle)
        ax.text(badge_x, badge_y, card["badge"],
                fontsize=17, color=BG, ha="center", va="center",
                fontweight="bold", family="sans-serif", zorder=6)

        # Component name
        ax.text(cx + 2.0, cy + card_h / 2 - 3.8, card["name"],
                fontsize=15, color=card["accent"], ha="center", va="center",
                fontweight="bold", family="sans-serif")
        # Subtitle
        ax.text(cx + 2.0, cy + card_h / 2 - 7.5, card["subtitle"],
                fontsize=12, color=SUBTEXT, ha="center", va="center",
                style="italic", family="sans-serif")

        # Per-market rows
        row_label_x = cx - card_w / 2 + 4.0
        row_value_x = cx - card_w / 2 + 13.0
        n_rows = len(card["rows"])
        row_step = 4.8
        rows_block_height = (n_rows - 1) * row_step
        row_top_y = cy + (rows_block_height / 2) - 1.5
        for i, (mkt, metric) in enumerate(card["rows"]):
            y = row_top_y - i * row_step
            ax.text(row_label_x, y, mkt,
                    fontsize=13, color=TEXT, ha="left", va="center",
                    fontweight="bold", family="sans-serif")
            ax.text(row_value_x, y, metric,
                    fontsize=11.5, color=TEXT, ha="left", va="center",
                    family="sans-serif")

        # Takeaway (bottom of card, multi-line)
        ax.text(cx, cy - card_h / 2 + 5.5, card["takeaway"],
                fontsize=12.5, color=TEXT, ha="center", va="center",
                style="italic", family="sans-serif",
                linespacing=1.35)

    # ---------------------- Bottom verdict line -----------------------
    ax.text(
        50, 5.5,
        "Four independent checks  ·  same verdict:  strong on DJI, partial on SP100, defensive trade on NDX.",
        fontsize=14, color=TITLE, ha="center", va="center",
        fontweight="bold", family="sans-serif",
    )
    ax.text(
        50, 1.5,
        "The classical, statistical, and downside-aware lenses all agree with the headline picture.",
        fontsize=13, color=SUBTEXT, ha="center", va="center",
        style="italic", family="sans-serif",
    )

    # ---------------------- Save and flatten alpha --------------------
    out_path = Path(__file__).parent / "slide26_robustness.png"
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
