"""Generate the Future Work infographic for Slide 31.

Three-column layout — one theme per column, three directions each.
Same navy/cyan visual language as Slides 23 / 26 / 29 / 30.
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


COLUMNS = [
    {
        "name": "ARCHITECTURAL EXTENSIONS",
        "subtitle": "scaling the recipe",
        "accent": ACCENT_CYAN,
        "items": [
            ("Multi-agent regime ensemble",
             "bull / sideways / bear specialists, routed by HMM"),
            ("Scale encoder capacity",
             "double embedding, heads, or stacked layers — tests RQ3 directly"),
            ("NDX architectural ablation pass",
             "give NDX the same programme DJI / SP100 received"),
        ],
    },
    {
        "name": "HMM REFINEMENTS",
        "subtitle": "3 variants already piloted",
        "accent": ACCENT_AMBER,
        "items": [
            ("Walk-forward retraining",
             "rolling 10y / expanding from 2010 — mixed gains"),
            ("Leading-indicator HMM",
             "+VIX-MA, +VIX-stress fraction — reveals 2025 luck"),
            ("Finer regime taxonomies",
             "4–6 states for slow / late-cycle / recovery"),
        ],
    },
    {
        "name": "METHODOLOGY + DEPLOYMENT",
        "subtitle": "tightening the rigor",
        "accent": ACCENT_GREEN,
        "items": [
            ("Multi-seed variance reporting",
             "addresses the biggest limitation"),
            ("Better validation criteria",
             "Sortino · Calmar · regime-stratified · composite"),
            ("International markets",
             "FTSE · DAX · Nikkei · MSCI — falsify or strengthen"),
        ],
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
        50, 95,
        "FUTURE WORK",
        fontsize=28, color=TITLE, ha="center", va="center",
        fontweight="bold", family="sans-serif",
    )
    ax.text(
        50, 90,
        "Three themes  ·  open research directions",
        fontsize=15, color=SUBTEXT, ha="center", va="center",
        style="italic", family="sans-serif",
    )

    # ----------------------------- Columns ----------------------------
    col_xs = [18.5, 50, 81.5]
    col_w, col_h = 30, 65
    col_top = 82
    col_bottom = col_top - col_h

    for cx, col in zip(col_xs, COLUMNS):
        b = FancyBboxPatch(
            (cx - col_w / 2, col_bottom),
            col_w, col_h,
            boxstyle="round,pad=0.5,rounding_size=0.9",
            facecolor=CARD_FACE, edgecolor=col["accent"],
            linewidth=2.6, alpha=0.95,
        )
        ax.add_patch(b)

        # Theme name (centred at top)
        ax.text(cx, col_top - 4.5, col["name"],
                fontsize=15, color=col["accent"],
                ha="center", va="center",
                fontweight="bold", family="sans-serif")
        ax.text(cx, col_top - 8.5, col["subtitle"],
                fontsize=11, color=SUBTEXT,
                ha="center", va="center",
                style="italic", family="sans-serif")

        # Items — bullet list
        item_top_y = col_top - 14.0
        item_step = 16.0
        for i, (item_name, item_sub) in enumerate(col["items"]):
            y = item_top_y - i * item_step
            # Number circle (small)
            badge_x = cx - col_w / 2 + 3.0
            circle = plt.Circle((badge_x, y), 1.4,
                                facecolor=col["accent"], edgecolor=col["accent"],
                                linewidth=1.2, zorder=5)
            ax.add_patch(circle)
            ax.text(badge_x, y, str(i + 1),
                    fontsize=12, color=BG, ha="center", va="center",
                    fontweight="bold", family="sans-serif", zorder=6)

            # Item name
            ax.text(badge_x + 3.0, y + 1.4, item_name,
                    fontsize=12, color=TEXT,
                    ha="left", va="center",
                    fontweight="bold", family="sans-serif")
            # Item sub
            ax.text(badge_x + 3.0, y - 1.7, item_sub,
                    fontsize=10, color=SUBTEXT,
                    ha="left", va="center",
                    style="italic", family="sans-serif")

    # ---------------------- Bottom verdict line -----------------------
    ax.text(
        50, 9.5,
        "TOP PRIORITIES",
        fontsize=14, color=TITLE, ha="center", va="center",
        fontweight="bold", family="sans-serif",
    )
    ax.text(
        50, 5.5,
        "Multi-seed variance reporting  ·  Scale encoder capacity (direct test of RQ3)",
        fontsize=14, color=TEXT, ha="center", va="center",
        fontweight="bold", family="sans-serif",
    )
    ax.text(
        50, 1.8,
        "the two next steps with the highest leverage on the thesis's open questions",
        fontsize=12, color=SUBTEXT, ha="center", va="center",
        style="italic", family="sans-serif",
    )

    # ---------------------- Save and flatten alpha --------------------
    out_path = Path(__file__).parent / "slide31_future_work.png"
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
