"""Generate the Live Demo intro slide (Slide 28).

A minimal transition slide. Big title, framing tagline, 3 bullets
on what to watch for in the dashboard, and a visual cue indicating
the switch to the live demo.

Landscape 16:9 to match the rest of the deck.
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
HIGHLIGHT_FACE = "#26477E"
ACCENT_CYAN = "#7DC8FF"
ACCENT_AMBER = "#FFC97A"
ACCENT_GREEN = "#3FBF7F"


def main():
    fig = plt.figure(figsize=(19.2, 10.8), dpi=200)
    fig.patch.set_facecolor(BG)
    ax = fig.add_subplot(111)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("auto")
    ax.axis("off")

    # ------------------------- Big title ---------------------------
    ax.text(
        50, 78,
        "LIVE DEMO",
        fontsize=58, color=TITLE, ha="center", va="center",
        fontweight="bold", family="sans-serif",
    )
    ax.text(
        50, 68,
        "Investor Simulator  ·  the canonical agents on 2025 data",
        fontsize=20, color=SUBTEXT, ha="center", va="center",
        style="italic", family="sans-serif",
    )

    # ------------------------- Watch-for bullets ------------------
    bullets_box = FancyBboxPatch(
        (15, 26), 70, 32,
        boxstyle="round,pad=0.5,rounding_size=1.0",
        facecolor=CARD_FACE, edgecolor=ACCENT_CYAN,
        linewidth=2.4, alpha=0.95,
    )
    ax.add_patch(bullets_box)

    ax.text(50, 53, "WHAT TO WATCH",
            fontsize=20, color=ACCENT_CYAN, ha="center", va="center",
            fontweight="bold", family="sans-serif")

    bullets = [
        ("1", "Day-by-day cash adaptation",
              "watch cash step up on HMM-bear days and drop on bull days",
              ACCENT_CYAN),
        ("2", "Attention rotation across regimes",
              "different stocks light up in bear vs bull regimes",
              ACCENT_AMBER),
        ("3", "Compound wealth trajectory",
              "the agent's portfolio value evolves on a year it never saw",
              ACCENT_GREEN),
    ]
    for i, (num, headline, sub, col) in enumerate(bullets):
        y = 45 - i * 7.0
        # Coloured number badge
        circle = plt.Circle((20, y), 1.8,
                            facecolor=col, edgecolor=col,
                            linewidth=1.5, zorder=5)
        ax.add_patch(circle)
        ax.text(20, y, num,
                fontsize=18, color=BG, ha="center", va="center",
                fontweight="bold", family="sans-serif", zorder=6)
        ax.text(26, y + 0.8, headline,
                fontsize=18, color=TEXT, ha="left", va="center",
                fontweight="bold", family="sans-serif")
        ax.text(26, y - 2.0, sub,
                fontsize=14, color=SUBTEXT, ha="left", va="center",
                style="italic", family="sans-serif")

    # ------------------------- Transition cue ----------------------
    ax.text(
        50, 17,
        "→  switching to the dashboard",
        fontsize=22, color=ACCENT_GREEN, ha="center", va="center",
        fontweight="bold", style="italic", family="sans-serif",
    )

    ax.text(
        50, 9,
        "Same canonical models from Chapter 4  ·  frozen HMM  ·  fresh 2025 OHLCV",
        fontsize=14, color=SUBTEXT, ha="center", va="center",
        style="italic", family="sans-serif",
    )

    # ---------------------- Save and flatten alpha --------------------
    out_path = Path(__file__).parent / "slide28_demo.png"
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
