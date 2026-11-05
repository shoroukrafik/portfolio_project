"""Generate the RQ3 / encoder-capacity infographic for Slide 24 (V2).

Improvements over V1:
- Reformulated RQ (not NDX-specific) — focuses on the structural question
- Smaller top cards (only the gradient-defining metric: bull capture %)
  · Wins / bear-cash removed (covered in Slide 22)
- BIG hypothesis box dominates the slide
- Clarifies exactly what the SP100 direct-test experiment removed
- Bigger fonts throughout for projector readability

Output: slide24_rq3_v2.png  (does NOT overwrite slide24_rq3.png)
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
ACCENT_GREEN = "#3FBF7F"
HYPO_FACE = "#26477E"
TEST_FACE = "#1F3658"  # slightly distinct for the test box


COLUMNS = [
    {
        "name": "DJI 30",
        "subtitle": "30 stocks  ·  LOW dispersion",
        "accent": ACCENT_CYAN,
        "bull_cap": "233%",
        "verdict": "Recipe wins",
    },
    {
        "name": "S&P 100",
        "subtitle": "90 stocks  ·  MODERATE dispersion",
        "accent": ACCENT_AMBER,
        "bull_cap": "74%",
        "verdict": "Partial win",
    },
    {
        "name": "NDX 100",
        "subtitle": "90 stocks  ·  HIGH (mega-cap) dispersion",
        "accent": ACCENT_RED,
        "bull_cap": "49%",
        "verdict": "Defensive trade",
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
        "RQ3  ·  IS THE BULL-CAPTURE GAP ARCHITECTURAL OR STRUCTURAL ?",
        fontsize=24, color=TITLE, ha="center", va="center",
        fontweight="bold", family="sans-serif",
    )
    ax.text(
        50, 91.0,
        "the encoder-capacity hypothesis  ·  tested directly on S&P 100",
        fontsize=15, color=SUBTEXT, ha="center", va="center",
        style="italic", family="sans-serif",
    )

    # ----------------------------- Top cards --------------------------
    col_xs = [18, 50, 82]
    col_w = 28
    col_top = 86
    col_bottom = 68
    col_h = col_top - col_bottom

    for cx, col in zip(col_xs, COLUMNS):
        b = FancyBboxPatch(
            (cx - col_w / 2, col_bottom), col_w, col_h,
            boxstyle="round,pad=0.5,rounding_size=0.8",
            facecolor=CARD_FACE, edgecolor=col["accent"],
            linewidth=2.4, alpha=0.95,
        )
        ax.add_patch(b)

        # Universe name
        ax.text(cx, col_top - 3.5, col["name"],
                fontsize=20, color=col["accent"],
                ha="center", va="center",
                fontweight="bold", family="sans-serif")
        # Subtitle (dispersion level)
        ax.text(cx, col_top - 7.5, col["subtitle"],
                fontsize=11.5, color=SUBTEXT,
                ha="center", va="center",
                style="italic", family="sans-serif")

        # Bull capture (the gradient metric — centred big number)
        ax.text(cx, col_top - 12.5, col["bull_cap"],
                fontsize=30, color=TEXT,
                ha="center", va="center",
                fontweight="bold", family="sans-serif")
        ax.text(cx, col_bottom + 2.5,
                "bull capture vs index",
                fontsize=11.5, color=SUBTEXT,
                ha="center", va="center", family="sans-serif")

    # Gradient pointer between top cards and hypothesis
    ax.text(50, 65.0,
            "bull capture DECREASES as universe size & dispersion grow  →  WHY?",
            fontsize=14, color=ACCENT_AMBER, ha="center", va="center",
            style="italic", fontweight="bold", family="sans-serif")

    # ----------------------- BIG hypothesis box -----------------------
    hypo_x, hypo_y, hypo_w, hypo_h = 50, 34, 94, 54
    hypo_box = FancyBboxPatch(
        (hypo_x - hypo_w / 2, hypo_y - hypo_h / 2), hypo_w, hypo_h,
        boxstyle="round,pad=0.6,rounding_size=1.0",
        facecolor=HYPO_FACE, edgecolor=TITLE,
        linewidth=2.5, alpha=0.95,
    )
    ax.add_patch(hypo_box)

    # Box header
    ax.text(hypo_x, hypo_y + 23.0,
            "THE ENCODER-CAPACITY HYPOTHESIS",
            fontsize=22, color=TITLE,
            ha="center", va="center",
            fontweight="bold", family="sans-serif")

    # Three-bullet explanation
    ax.text(hypo_x, hypo_y + 16.5,
            "1.  Attention extractor has a FIXED 64-dim representational budget.",
            fontsize=15, color=TEXT,
            ha="center", va="center", family="sans-serif")
    ax.text(hypo_x, hypo_y + 11.5,
            "2.  As universe grows 30 → 90, regime features compete with",
            fontsize=15, color=TEXT,
            ha="center", va="center", family="sans-serif")
    ax.text(hypo_x, hypo_y + 8.0,
            "cross-sectional stock features for that same fixed budget.",
            fontsize=15, color=TEXT,
            ha="center", va="center", family="sans-serif")
    ax.text(hypo_x, hypo_y + 3.0,
            "3.  Budget runs out → policy defaults to a DEFENSIVE diversified allocation.",
            fontsize=15, color=TEXT,
            ha="center", va="center", family="sans-serif")

    # ---- Direct test sub-box (highlighted) ----
    test_x, test_y, test_w, test_h = 50, hypo_y - 11.5, 86, 16
    test_box = FancyBboxPatch(
        (test_x - test_w / 2, test_y - test_h / 2), test_w, test_h,
        boxstyle="round,pad=0.4,rounding_size=0.8",
        facecolor=TEST_FACE, edgecolor=ACCENT_AMBER,
        linewidth=2.0, alpha=0.95,
    )
    ax.add_patch(test_box)

    ax.text(test_x, test_y + 5.0,
            "DIRECT TEST  ·  retrained S&P 100 with HMM regime label + posteriors REMOVED from observation",
            fontsize=13, color=ACCENT_AMBER,
            ha="center", va="center",
            fontweight="bold", family="sans-serif")
    ax.text(test_x, test_y + 1.5,
            "(regime-λ in reward kept  ·  cash floor kept  ·  everything else identical)",
            fontsize=11.5, color=SUBTEXT,
            ha="center", va="center",
            style="italic", family="sans-serif")
    ax.text(test_x, test_y - 2.0,
            "Result:  6 / 8 wins   (vs canonical 5 / 8)   ·   recovery   +4 pp",
            fontsize=15, color=ACCENT_GREEN,
            ha="center", va="center",
            fontweight="bold", family="sans-serif")
    ax.text(test_x, test_y - 5.5,
            "freeing budget for stock features → MORE wins  ·  empirical support for the hypothesis",
            fontsize=11.5, color=TEXT,
            ha="center", va="center",
            style="italic", family="sans-serif")

    # ---------------------- Bottom verdict line -----------------------
    ax.text(
        50, 4.0,
        "RQ3 answered:  the bull-capture gap is STRUCTURAL (capacity bottleneck), not architectural.",
        fontsize=14, color=ACCENT_AMBER, ha="center", va="center",
        fontweight="bold", style="italic", family="sans-serif",
    )

    # ---------------------- Save and flatten alpha --------------------
    out_path = Path(__file__).parent / "slide24_rq3_v2.png"
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
