"""Generate the 2025 forward + soft-floor infographic for Slide 27 (landscape 16:9).

Layout:
  - Top row: Section 1 (results) | Section 2 (diagnostic) side by side
  - Bottom row: Section 3 (soft-floor refinement) full width
  - Verdict + caveat at very bottom

All fonts large enough for presentation (smallest ≈ 18-20pt).
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
    # 16:9 landscape — 1920 x 1080-ish at 200 dpi
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
        "FORWARD VALIDATION ON 2025  ·  SOFT-FLOOR REFINEMENT",
        fontsize=22, color=TITLE, ha="center", va="center",
        fontweight="bold", family="sans-serif",
    )
    ax.text(
        50, 90.5,
        "Out-of-sample stress test on data the agent never saw",
        fontsize=15, color=SUBTEXT, ha="center", va="center",
        style="italic", family="sans-serif",
    )

    # ----------------- TOP ROW: section 1 | section 2 ----------------
    # Section 1: 2025 results (left half)
    sec1_x, sec1_y, sec1_w, sec1_h = 25.5, 67, 46, 30
    box1 = FancyBboxPatch(
        (sec1_x - sec1_w / 2, sec1_y - sec1_h / 2), sec1_w, sec1_h,
        boxstyle="round,pad=0.5,rounding_size=0.9",
        facecolor=CARD_FACE, edgecolor=ACCENT_CYAN,
        linewidth=2.6, alpha=0.95,
    )
    ax.add_patch(box1)

    # Badge "1"
    badge_x = sec1_x - sec1_w / 2 + 4.0
    badge_y = sec1_y + sec1_h / 2 - 4.0
    circle = plt.Circle((badge_x, badge_y), 2.1,
                        facecolor=ACCENT_CYAN, edgecolor=ACCENT_CYAN,
                        linewidth=1.5, zorder=5)
    ax.add_patch(circle)
    ax.text(badge_x, badge_y, "1",
            fontsize=18, color=BG, ha="center", va="center",
            fontweight="bold", family="sans-serif", zorder=6)

    ax.text(sec1_x, sec1_y + 10.0, "THE RECIPE HOLDS ON 2025",
            fontsize=18, color=ACCENT_CYAN, ha="center", va="center",
            fontweight="bold", family="sans-serif")
    ax.text(sec1_x, sec1_y + 6.5,
            "Cross-market pattern persists",
            fontsize=14, color=SUBTEXT, ha="center", va="center",
            style="italic", family="sans-serif")

    # Three markets in section 1
    market_xs_s1 = [
        sec1_x - sec1_w * 0.30,
        sec1_x,
        sec1_x + sec1_w * 0.30,
    ]
    markets = [
        ("DJI",   "+15.5%", "vs BH +14.1%"),
        ("SP100", "+13.1%", "vs SP500 +17.5%"),
        ("NDX",   "+11.9%", "vs NDX +21.4%"),
    ]
    for mx, (mkt, agent_pct, benchmark) in zip(market_xs_s1, markets):
        ax.text(mx, sec1_y + 0.5, mkt,
                fontsize=17, color=TEXT, ha="center", va="center",
                fontweight="bold", family="sans-serif")
        ax.text(mx, sec1_y - 4.0, agent_pct,
                fontsize=26, color=ACCENT_CYAN, ha="center", va="center",
                fontweight="bold", family="sans-serif")
        ax.text(mx, sec1_y - 8.0, benchmark,
                fontsize=13, color=SUBTEXT, ha="center", va="center",
                family="sans-serif")

    # Section 2: Diagnostic (right half)
    sec2_x, sec2_y, sec2_w, sec2_h = 74.5, 67, 46, 30
    box2 = FancyBboxPatch(
        (sec2_x - sec2_w / 2, sec2_y - sec2_h / 2), sec2_w, sec2_h,
        boxstyle="round,pad=0.5,rounding_size=0.9",
        facecolor=CARD_FACE, edgecolor=ACCENT_AMBER,
        linewidth=2.6, alpha=0.95,
    )
    ax.add_patch(box2)

    badge_x = sec2_x - sec2_w / 2 + 4.0
    badge_y = sec2_y + sec2_h / 2 - 4.0
    circle = plt.Circle((badge_x, badge_y), 2.1,
                        facecolor=ACCENT_AMBER, edgecolor=ACCENT_AMBER,
                        linewidth=1.5, zorder=5)
    ax.add_patch(circle)
    ax.text(badge_x, badge_y, "2",
            fontsize=18, color=BG, ha="center", va="center",
            fontweight="bold", family="sans-serif", zorder=6)

    ax.text(sec2_x, sec2_y + 10.0, "DIAGNOSTIC",
            fontsize=18, color=ACCENT_AMBER, ha="center", va="center",
            fontweight="bold", family="sans-serif")
    ax.text(sec2_x, sec2_y + 6.0,
            "2025 was a regime-noise year",
            fontsize=14, color=SUBTEXT, ha="center", va="center",
            style="italic", family="sans-serif")

    ax.text(sec2_x, sec2_y + 1.5,
            "HMM flagged 32 days (~13%) as bear",
            fontsize=14, color=TEXT, ha="center", va="center",
            family="sans-serif")
    ax.text(sec2_x, sec2_y - 1.5,
            "but most were short corrections, not sustained downtrends.",
            fontsize=14, color=TEXT, ha="center", va="center",
            family="sans-serif")
    ax.text(sec2_x, sec2_y - 5.5,
            "Hard 70% floor activated on every one  ·  missed rebounds",
            fontsize=14, color=TEXT, ha="center", va="center",
            family="sans-serif")
    ax.text(sec2_x, sec2_y - 8.5,
            "DJI cost ≈ 7 pp",
            fontsize=14, color=ACCENT_AMBER, ha="center", va="center",
            fontweight="bold", family="sans-serif")

    # ----------------- BOTTOM: Section 3 full width ----------------
    sec3_x, sec3_y, sec3_w, sec3_h = 50, 33, 94, 30
    box3 = FancyBboxPatch(
        (sec3_x - sec3_w / 2, sec3_y - sec3_h / 2), sec3_w, sec3_h,
        boxstyle="round,pad=0.5,rounding_size=0.9",
        facecolor=HIGHLIGHT_FACE, edgecolor=ACCENT_GREEN,
        linewidth=2.8, alpha=0.95,
    )
    ax.add_patch(box3)

    badge_x = sec3_x - sec3_w / 2 + 4.0
    badge_y = sec3_y + sec3_h / 2 - 4.0
    circle = plt.Circle((badge_x, badge_y), 2.1,
                        facecolor=ACCENT_GREEN, edgecolor=ACCENT_GREEN,
                        linewidth=1.5, zorder=5)
    ax.add_patch(circle)
    ax.text(badge_x, badge_y, "3",
            fontsize=18, color=BG, ha="center", va="center",
            fontweight="bold", family="sans-serif", zorder=6)

    ax.text(sec3_x, sec3_y + 11.5,
            "SOFT FLOOR REFINEMENT  ·  scale with HMM confidence",
            fontsize=20, color=ACCENT_GREEN, ha="center", va="center",
            fontweight="bold", family="sans-serif")

    ax.text(sec3_x, sec3_y + 6.0,
            r"$f_t \;=\; f_{max} \,\cdot\, p_{bear}(t)$",
            fontsize=26, color=TEXT, ha="center", va="center",
            family="sans-serif")

    ax.text(sec3_x, sec3_y + 2.0,
            "No new hyperparameters  ·  recovers the canonical hard floor at the limits",
            fontsize=14, color=SUBTEXT, ha="center", va="center",
            style="italic", family="sans-serif")

    # Soft vs Hard comparison row
    soft_xs = [22, 50, 78]
    soft_data = [
        ("DJI 2025",       "+15.5% → +17.7%",  "+2.2 pp"),
        ("SP100 2025",     "+13.1% → +14.3%",  "+1.2 pp"),
        ("Thesis windows", "≈ neutral",         "± 0.7 pp"),
    ]
    for sx, (label, change, delta) in zip(soft_xs, soft_data):
        ax.text(sx, sec3_y - 2.5, label,
                fontsize=16, color=TEXT, ha="center", va="center",
                fontweight="bold", family="sans-serif")
        ax.text(sx, sec3_y - 7.0, change,
                fontsize=20, color=ACCENT_GREEN, ha="center", va="center",
                fontweight="bold", family="sans-serif")
        ax.text(sx, sec3_y - 11.0, delta,
                fontsize=15, color=SUBTEXT, ha="center", va="center",
                style="italic", family="sans-serif")

    # ---------------------- Bottom verdict + caveat -------------------
    ax.text(
        50, 13.5,
        "Cross-market pattern persists on out-of-sample 2025 data.",
        fontsize=15, color=TITLE, ha="center", va="center",
        fontweight="bold", family="sans-serif",
    )
    ax.text(
        50, 9.5,
        "Soft refinement adds ~2 pp on DJI 2025 without disturbing thesis-window canonical results.",
        fontsize=14, color=SUBTEXT, ha="center", va="center",
        style="italic", family="sans-serif",
    )
    # ---------------------- Save and flatten alpha --------------------
    out_path = Path(__file__).parent / "slide27_forward.png"
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
