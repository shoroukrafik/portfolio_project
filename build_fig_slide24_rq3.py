"""Generate the RQ3 / encoder-capacity infographic for Slide 24.

Three columns (DJI / SP100 / NDX) showing the progression across the
dispersion gradient: wins, bull capture, cash share, ablation pattern.

Below the columns, a concise hypothesis statement explains the mechanism.

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
ACCENT_RED = "#E66060"
HYPO_FACE = "#26477E"


COLUMNS = [
    {
        "name": "DJI 30",
        "subtitle": "30 stocks  ·  low dispersion",
        "accent": ACCENT_CYAN,
        "wins": "8 / 8",
        "bull_cap": "233%",
        "bear_cash": "68%",
        "ablation": "Removing attention or regime\nhurts every period.",
        "verdict": "Recipe wins both ends:\nbull AND bear",
    },
    {
        "name": "S&P 100",
        "subtitle": "90 stocks  ·  moderate dispersion",
        "accent": ACCENT_AMBER,
        "wins": "5 / 8",
        "bull_cap": "74%",
        "bear_cash": "29%",
        "ablation": "Removing regime features\nactually helps non-bear.",
        "verdict": "Partial win:\nstrong bear, lagging bull",
    },
    {
        "name": "NDX 2017",
        "subtitle": "90 stocks  ·  high (mega-cap) dispersion",
        "accent": ACCENT_RED,
        "wins": "4 / 8",
        "bull_cap": "49%",
        "bear_cash": "0.7%",
        "ablation": "Bull capture limited by\nMag-7 concentration",
        "verdict": "Defensive trade:\nstrong bear, weak bull",
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
        50, 95,
        "RQ3  ·  WHY DOES THE RECIPE TRADE BULL FOR BEAR ON NDX ?",
        fontsize=21, color=TITLE, ha="center", va="center",
        fontweight="bold", family="sans-serif",
    )
    ax.text(
        50, 90,
        "the encoder-capacity hypothesis + dispersion gradient",
        fontsize=13, color=SUBTEXT, ha="center", va="center",
        style="italic", family="sans-serif",
    )

    # ----------------------------- Columns ----------------------------
    col_xs = [18, 50, 82]
    col_w = 28
    col_top = 82
    col_bottom = 38

    for cx, col in zip(col_xs, COLUMNS):
        # Card
        b = FancyBboxPatch(
            (cx - col_w / 2, col_bottom),
            col_w, col_top - col_bottom,
            boxstyle="round,pad=0.6,rounding_size=1.0",
            facecolor=CARD_FACE, edgecolor=col["accent"],
            linewidth=2.5, alpha=0.95,
        )
        ax.add_patch(b)

        # Header band
        ax.text(cx, col_top - 4.0, col["name"],
                fontsize=19, color=col["accent"],
                ha="center", va="center",
                fontweight="bold", family="sans-serif")
        ax.text(cx, col_top - 8.0, col["subtitle"],
                fontsize=12, color=SUBTEXT,
                ha="center", va="center",
                style="italic", family="sans-serif")

        # Wins (big headline number)
        ax.text(cx, col_top - 14.5, col["wins"],
                fontsize=24, color=TEXT,
                ha="center", va="center",
                fontweight="bold", family="sans-serif")
        ax.text(cx, col_top - 18.5, "wins",
                fontsize=12, color=SUBTEXT,
                ha="center", va="center", family="sans-serif")

        # Bull capture
        ax.text(cx - col_w * 0.20, col_top - 24.5, col["bull_cap"],
                fontsize=17, color=TEXT,
                ha="center", va="center",
                fontweight="bold", family="sans-serif")
        ax.text(cx - col_w * 0.20, col_top - 28.3, "bull capture\nvs index",
                fontsize=10.5, color=SUBTEXT,
                ha="center", va="center", family="sans-serif",
                linespacing=1.3)

        # Bear cash share
        ax.text(cx + col_w * 0.20, col_top - 24.5, col["bear_cash"],
                fontsize=17, color=TEXT,
                ha="center", va="center",
                fontweight="bold", family="sans-serif")
        ax.text(cx + col_w * 0.20, col_top - 28.3, "avg cash\non bear days",
                fontsize=10.5, color=SUBTEXT,
                ha="center", va="center", family="sans-serif",
                linespacing=1.3)

        # Ablation pattern line
        ax.text(cx, col_top - 35.5, col["ablation"],
                fontsize=11.5, color=SUBTEXT,
                ha="center", va="center",
                style="italic", family="sans-serif",
                linespacing=1.3)

        # Verdict line at the bottom
        ax.text(cx, col_bottom + 3.5, col["verdict"],
                fontsize=12, color=col["accent"],
                ha="center", va="center",
                fontweight="bold", family="sans-serif",
                linespacing=1.3)

    # ---------------------- Hypothesis explanation -------------------
    hypo_x, hypo_y, hypo_w, hypo_h = 50, 19, 88, 22
    hypo_box = FancyBboxPatch(
        (hypo_x - hypo_w / 2, hypo_y - hypo_h / 2), hypo_w, hypo_h,
        boxstyle="round,pad=0.5,rounding_size=1.0",
        facecolor=HYPO_FACE, edgecolor=TITLE,
        linewidth=2.0, alpha=0.95,
    )
    ax.add_patch(hypo_box)
    ax.text(hypo_x, hypo_y + 8.5,
            "THE ENCODER-CAPACITY HYPOTHESIS",
            fontsize=15, color=TITLE,
            ha="center", va="center",
            fontweight="bold", family="sans-serif")
    ax.text(hypo_x, hypo_y + 4.5,
            "The 64-dim attention extractor has a fixed representational budget.",
            fontsize=12, color=TEXT,
            ha="center", va="center", family="sans-serif")
    ax.text(hypo_x, hypo_y + 1.0,
            "As stocks-per-universe grows from 30 to 90, regime features compete with cross-sectional features for that budget.",
            fontsize=12, color=TEXT,
            ha="center", va="center", family="sans-serif")
    ax.text(hypo_x, hypo_y - 2.5,
            "The policy collapses to a defensive diversified allocation — bear defence remains, bull capture suffers.",
            fontsize=12, color=TEXT,
            ha="center", va="center", family="sans-serif")

    # Supporting evidence line — SP100 per-universe experiment (§4.3.2)
    ax.text(hypo_x, hypo_y - 6.8,
            "DIRECT TEST on SP100:  removing regime features → 6 / 8 wins (vs canonical 5 / 8)  ·  +4 pp on recovery",
            fontsize=12, color=ACCENT_AMBER,
            ha="center", va="center",
            fontweight="bold", family="sans-serif")

    # ---------------------- Bottom verdict line -----------------------
    ax.text(
        50, 4.0,
        "RQ3 answered:  the cross-market asymmetry has a structural cause, not an architectural flaw.",
        fontsize=13, color=ACCENT_AMBER, ha="center", va="center",
        fontweight="bold", style="italic", family="sans-serif",
    )

    # ---------------------- Save and flatten alpha --------------------
    out_path = Path(__file__).parent / "slide24_rq3.png"
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
