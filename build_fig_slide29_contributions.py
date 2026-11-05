"""Generate the 5-contributions infographic for Slide 29.

Vertical-list layout (5 rows) — name + tagline + tag.
Landscape 16:9 for Canva.
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
ACCENT_PURPLE = "#B58CE8"


CONTRIBUTIONS = [
    {
        "badge": "1",
        "name": "CROSS-MARKET VALIDATION",
        "tag": "RQ1",
        "tagline": "One recipe  ·  three markets  ·  fixed protocol",
        "accent": ACCENT_CYAN,
    },
    {
        "badge": "2",
        "name": "LEARNER + OVERLAY SEPARABILITY",
        "tag": "RQ2",
        "tagline": "Matched ablations isolate each piece",
        "accent": ACCENT_AMBER,
    },
    {
        "badge": "3",
        "name": "DEFENSIVE SELECTION RULE",
        "tag": "methodology",
        "tagline": "≥20% training-budget filter  ·  reproducible",
        "accent": ACCENT_GREEN,
    },
    {
        "badge": "4",
        "name": "ENCODER-CAPACITY HYPOTHESIS",
        "tag": "RQ3",
        "tagline": "Testable structural prediction for the field",
        "accent": ACCENT_RED,
    },
    {
        "badge": "5",
        "name": "SOFT POSTERIOR-WEIGHTED FLOOR",
        "tag": "methodology",
        "tagline": r"$f_t = f_{max} \cdot p_{bear}(t)$   ·  confidence-scaled refinement",
        "accent": ACCENT_PURPLE,
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
        "FIVE CONTRIBUTIONS",
        fontsize=28, color=TITLE, ha="center", va="center",
        fontweight="bold", family="sans-serif",
    )
    ax.text(
        50, 90,
        "What this thesis delivers",
        fontsize=16, color=SUBTEXT, ha="center", va="center",
        style="italic", family="sans-serif",
    )

    # ----------------------------- Rows -------------------------------
    # 5 vertical rows
    row_top = 79
    row_h = 12.5
    row_spacing = 0.8

    for i, contrib in enumerate(CONTRIBUTIONS):
        cy = row_top - i * (row_h + row_spacing)
        cx = 50
        row_w = 88

        # Card
        b = FancyBboxPatch(
            (cx - row_w / 2, cy - row_h / 2), row_w, row_h,
            boxstyle="round,pad=0.4,rounding_size=0.8",
            facecolor=CARD_FACE, edgecolor=contrib["accent"],
            linewidth=2.5, alpha=0.95,
        )
        ax.add_patch(b)

        # Number badge on the left
        badge_x = cx - row_w / 2 + 5.0
        badge_y = cy
        circle = plt.Circle((badge_x, badge_y), 2.6,
                            facecolor=contrib["accent"], edgecolor=contrib["accent"],
                            linewidth=1.5, zorder=5)
        ax.add_patch(circle)
        ax.text(badge_x, badge_y, contrib["badge"],
                fontsize=22, color=BG, ha="center", va="center",
                fontweight="bold", family="sans-serif", zorder=6)

        # Contribution name
        name_x = badge_x + 5.5
        ax.text(name_x, cy + 2.0, contrib["name"],
                fontsize=19, color=contrib["accent"], ha="left", va="center",
                fontweight="bold", family="sans-serif")
        # Tagline
        ax.text(name_x, cy - 2.5, contrib["tagline"],
                fontsize=15, color=TEXT, ha="left", va="center",
                family="sans-serif")

        # Tag on the right (RQ1 / methodology / etc.)
        tag_x = cx + row_w / 2 - 5.0
        ax.text(tag_x, cy, contrib["tag"],
                fontsize=14, color=contrib["accent"], ha="right", va="center",
                style="italic", fontweight="bold", family="sans-serif")

    # ---------------------- Bottom verdict line -----------------------
    ax.text(
        50, 6.0,
        "RQ1  ·  RQ2  ·  RQ3  answered",
        fontsize=18, color=TITLE, ha="center", va="center",
        fontweight="bold", family="sans-serif",
    )
    ax.text(
        50, 2.3,
        "brief delivered:  adaptivity  ·  explainability  ·  performance constraints",
        fontsize=15, color=SUBTEXT, ha="center", va="center",
        style="italic", family="sans-serif",
    )

    # ---------------------- Save and flatten alpha --------------------
    out_path = Path(__file__).parent / "slide29_contributions.png"
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
