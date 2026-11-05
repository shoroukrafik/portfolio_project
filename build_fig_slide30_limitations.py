"""Generate the Limitations infographic for Slide 30.

Vertical-list layout (5 rows) — same style as Slide 29.
Honest acknowledgment of top 5 limitations from §4.8.2.
"""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image


# Palette — slightly muted accents since these are limitations
BG = "#0E1B3A"
TEXT = "white"
TITLE = "#7DC8FF"
SUBTEXT = "#C9D5EA"
CARD_FACE = "#16264F"
ACCENT_RED = "#E66060"        # most important = red
ACCENT_AMBER = "#FFC97A"
ACCENT_CYAN = "#7DC8FF"
ACCENT_GREEN = "#3FBF7F"
ACCENT_GREY = "#8E9DBC"


LIMITATIONS = [
    {
        "badge": "1",
        "name": "SINGLE TRAINING SEED",
        "tag": "most important",
        "body": "All canonical models trained with seed = 42  ·  multi-seed variance deferred",
        "accent": ACCENT_RED,
    },
    {
        "badge": "2",
        "name": "THREE-MARKET SAMPLE SIZE",
        "tag": "sample bound",
        "body": "Smallest sample producing asymmetry  ·  not yet a population-level claim",
        "accent": ACCENT_AMBER,
    },
    {
        "badge": "3",
        "name": "CASH-FLOOR REGIME-DISTRIBUTION SENSITIVITY",
        "tag": "evidence: 2025",
        "body": "Hard floor activated on 32 short corrections in 2025  ·  cost ~7 pp on DJI",
        "accent": ACCENT_CYAN,
    },
    {
        "badge": "4",
        "name": "NDX-SPECIFIC LIMITS",
        "tag": "scope",
        "body": "4-year training window  ·  uses S&P 500 HMM  ·  no architectural ablation pass",
        "accent": ACCENT_GREEN,
    },
    {
        "badge": "5",
        "name": "NO WALK-FORWARD RETRAINING",
        "tag": "deployment",
        "body": "HMM + PPO frozen  ·  clean held-out evaluation, NOT a deployment simulation",
        "accent": ACCENT_GREY,
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
        "LIMITATIONS  ·  HONEST ACKNOWLEDGMENTS",
        fontsize=26, color=TITLE, ha="center", va="center",
        fontweight="bold", family="sans-serif",
    )
    ax.text(
        50, 90,
        "What bounds the strength of the claims",
        fontsize=15, color=SUBTEXT, ha="center", va="center",
        style="italic", family="sans-serif",
    )

    # ----------------------------- Rows -------------------------------
    row_top = 79
    row_h = 12.5
    row_spacing = 0.8

    for i, lim in enumerate(LIMITATIONS):
        cy = row_top - i * (row_h + row_spacing)
        cx = 50
        row_w = 88

        b = FancyBboxPatch(
            (cx - row_w / 2, cy - row_h / 2), row_w, row_h,
            boxstyle="round,pad=0.4,rounding_size=0.8",
            facecolor=CARD_FACE, edgecolor=lim["accent"],
            linewidth=2.4, alpha=0.95,
        )
        ax.add_patch(b)

        # Number badge
        badge_x = cx - row_w / 2 + 5.0
        badge_y = cy
        circle = plt.Circle((badge_x, badge_y), 2.6,
                            facecolor=lim["accent"], edgecolor=lim["accent"],
                            linewidth=1.5, zorder=5)
        ax.add_patch(circle)
        ax.text(badge_x, badge_y, lim["badge"],
                fontsize=22, color=BG, ha="center", va="center",
                fontweight="bold", family="sans-serif", zorder=6)

        # Name
        name_x = badge_x + 5.5
        ax.text(name_x, cy + 2.0, lim["name"],
                fontsize=17, color=lim["accent"], ha="left", va="center",
                fontweight="bold", family="sans-serif")
        # Body
        ax.text(name_x, cy - 2.5, lim["body"],
                fontsize=14, color=TEXT, ha="left", va="center",
                family="sans-serif")

        # Tag on the right
        tag_x = cx + row_w / 2 - 5.0
        ax.text(tag_x, cy, lim["tag"],
                fontsize=13, color=lim["accent"], ha="right", va="center",
                style="italic", fontweight="bold", family="sans-serif")

    # ---------------------- Bottom verdict line -----------------------
    ax.text(
        50, 5.5,
        "These BOUND the strength of the claims",
        fontsize=18, color=TITLE, ha="center", va="center",
        fontweight="bold", family="sans-serif",
    )
    ax.text(
        50, 1.8,
        "none invalidate the headline contributions",
        fontsize=15, color=SUBTEXT, ha="center", va="center",
        style="italic", family="sans-serif",
    )

    # ---------------------- Save and flatten alpha --------------------
    out_path = Path(__file__).parent / "slide30_limitations.png"
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
