"""Generate the attention-diagnostics infographic for Slide 25 (V2).

Improvements over V1:
- Bull/bear panels SHRUNK (40 → 30 units tall) so the interpretation box can grow
- Interpretation box EXPANDED (14 → 28 units tall) — full room for the WHY
- Visual hierarchy FLIPPED:
    SECTOR  (the "kind of market") = big, bold, primary
    ticker  (the "name")           = small, grey, secondary
- Bigger fonts everywhere for projector readability
- Bigger ATTENTION-SHARE numbers (the key quant evidence)

Output: slide25_attention_v2.png  (does NOT overwrite slide25_attention.png)
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
TICKER_DIM = "#8E9DBC"           # dim grey for ticker secondary text
CARD_FACE = "#16264F"
BULL_FACE = "#1B4530"
BEAR_FACE = "#4C2027"
BULL_ACCENT = "#3FBF7F"
BEAR_ACCENT = "#E66060"
HYPO_FACE = "#26477E"
ACCENT_CYAN = "#7DC8FF"


# (sector descriptor, ticker)
DJI_BULL = [
    ("CHIPS  ·  GROWTH",   "NVDA"),
    ("INDUSTRIALS",        "MMM"),
    ("CHIPS",              "INTC"),
    ("ENTERPRISE TECH",    "IBM"),
    ("INDUSTRIALS",        "BA"),
]
DJI_BEAR = [
    ("CHIPS  ·  GROWTH",   "NVDA"),
    ("INDUSTRIALS",        "MMM"),
    ("INDUSTRIALS",        "BA"),
    ("OIL  ·  ENERGY",     "CVX"),
    ("CONSUMER TECH",      "AAPL"),
]
SP100_BULL = [
    ("CHIPS  ·  GROWTH",   "NVDA"),
    ("HEALTHCARE",         "LLY"),
    ("CONSUMER",           "TGT"),
    ("AGRI-EQUIPMENT",     "DE"),
    ("ENTERPRISE TECH",    "ORCL"),
]
SP100_BEAR = [
    ("CHIPS  ·  GROWTH",   "NVDA"),
    ("HEALTHCARE",         "LLY"),
    ("OIL  ·  ENERGY",     "COP"),
    ("CONSUMER STAPLES",   "COST"),
    ("AGRI-EQUIPMENT",     "DE"),
]

COLUMNS = [
    {
        "name": "DJI 30",
        "bull": DJI_BULL, "bear": DJI_BEAR,
        "accent": ACCENT_CYAN,
        "bull_share": "71%", "bear_share": "52%",
        "uniform": "17%",
    },
    {
        "name": "S&P 100",
        "bull": SP100_BULL, "bear": SP100_BEAR,
        "accent": ACCENT_CYAN,
        "bull_share": "43%", "bear_share": "50%",
        "uniform": "6%",
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
        "ATTENTION DIAGNOSTICS  ·  WHERE THE AGENT LOOKS",
        fontsize=24, color=TITLE, ha="center", va="center",
        fontweight="bold", family="sans-serif",
    )
    ax.text(
        50, 91.5,
        "top-5 attended SECTORS in each regime  ·  same agent, different rotation",
        fontsize=14, color=SUBTEXT, ha="center", va="center",
        style="italic", family="sans-serif",
    )

    # ----------------------------- Columns ----------------------------
    col_xs = [27, 73]
    col_w = 44
    col_top = 87
    col_bottom = 49

    for cx, col in zip(col_xs, COLUMNS):
        # Outer card
        b = FancyBboxPatch(
            (cx - col_w / 2, col_bottom),
            col_w, col_top - col_bottom,
            boxstyle="round,pad=0.6,rounding_size=1.0",
            facecolor=CARD_FACE, edgecolor=col["accent"],
            linewidth=2.4, alpha=0.95,
        )
        ax.add_patch(b)

        # Market header (smaller — secondary)
        ax.text(cx, col_top - 3.5, col["name"],
                fontsize=16, color=col["accent"],
                ha="center", va="center",
                fontweight="bold", family="sans-serif")

        # ---- BULL panel ----
        bull_x = cx - col_w * 0.24
        bull_w = col_w * 0.46
        panel_top = col_top - 7.5
        panel_h = 28
        bull_box = FancyBboxPatch(
            (bull_x - bull_w / 2, panel_top - panel_h),
            bull_w, panel_h,
            boxstyle="round,pad=0.3,rounding_size=0.6",
            facecolor=BULL_FACE, edgecolor=BULL_ACCENT,
            linewidth=1.8, alpha=0.95,
        )
        ax.add_patch(bull_box)
        ax.text(bull_x, panel_top - 2.8, "TOP 5 IN BULL",
                fontsize=14, color=BULL_ACCENT,
                ha="center", va="center",
                fontweight="bold", family="sans-serif")

        for i, (sector, ticker) in enumerate(col["bull"]):
            y = panel_top - 7.5 - i * 3.7
            # SECTOR — primary, big, bold, white
            ax.text(bull_x - bull_w * 0.42, y, sector,
                    fontsize=12, color=TEXT,
                    ha="left", va="center",
                    fontweight="bold", family="sans-serif")
            # ticker — secondary, small, dim, right-aligned
            ax.text(bull_x + bull_w * 0.42, y, ticker,
                    fontsize=10, color=TICKER_DIM,
                    ha="right", va="center",
                    style="italic", family="sans-serif")

        # Share line (bigger, more prominent)
        ax.text(bull_x, panel_top - panel_h + 2.8,
                f"top-5 attention share:  {col['bull_share']}",
                fontsize=13, color=BULL_ACCENT,
                ha="center", va="center",
                fontweight="bold", family="sans-serif")

        # ---- BEAR panel ----
        bear_x = cx + col_w * 0.24
        bear_w = col_w * 0.46
        bear_box = FancyBboxPatch(
            (bear_x - bear_w / 2, panel_top - panel_h),
            bear_w, panel_h,
            boxstyle="round,pad=0.3,rounding_size=0.6",
            facecolor=BEAR_FACE, edgecolor=BEAR_ACCENT,
            linewidth=1.8, alpha=0.95,
        )
        ax.add_patch(bear_box)
        ax.text(bear_x, panel_top - 2.8, "TOP 5 IN BEAR",
                fontsize=14, color=BEAR_ACCENT,
                ha="center", va="center",
                fontweight="bold", family="sans-serif")

        for i, (sector, ticker) in enumerate(col["bear"]):
            y = panel_top - 7.5 - i * 3.7
            ax.text(bear_x - bear_w * 0.42, y, sector,
                    fontsize=12, color=TEXT,
                    ha="left", va="center",
                    fontweight="bold", family="sans-serif")
            ax.text(bear_x + bear_w * 0.42, y, ticker,
                    fontsize=10, color=TICKER_DIM,
                    ha="right", va="center",
                    style="italic", family="sans-serif")

        ax.text(bear_x, panel_top - panel_h + 2.8,
                f"top-5 attention share:  {col['bear_share']}",
                fontsize=13, color=BEAR_ACCENT,
                ha="center", va="center",
                fontweight="bold", family="sans-serif")

        # Uniform reference line beneath the two panels
        ax.text(cx, col_bottom - 2.5,
                f"uniform attention would be {col['uniform']}  ·  the agent concentrates well above that",
                fontsize=12, color=SUBTEXT,
                ha="center", va="center",
                style="italic", family="sans-serif")

    # ---------------------- BIG interpretation block ------------------
    hypo_x, hypo_y, hypo_w, hypo_h = 50, 22, 92, 28
    hypo_box = FancyBboxPatch(
        (hypo_x - hypo_w / 2, hypo_y - hypo_h / 2), hypo_w, hypo_h,
        boxstyle="round,pad=0.5,rounding_size=1.0",
        facecolor=HYPO_FACE, edgecolor=TITLE,
        linewidth=2.4, alpha=0.95,
    )
    ax.add_patch(hypo_box)

    # Header
    ax.text(hypo_x, hypo_y + 11.0,
            "ECONOMICALLY INTERPRETABLE ROTATION",
            fontsize=18, color=TITLE,
            ha="center", va="center",
            fontweight="bold", family="sans-serif")

    # Bull / Bear interpretation
    ax.text(hypo_x, hypo_y + 5.0,
            "BULL  →  attention favours growth tech and industrials",
            fontsize=14, color=BULL_ACCENT,
            ha="center", va="center",
            fontweight="bold", family="sans-serif")
    ax.text(hypo_x, hypo_y + 1.0,
            "BEAR  →  rotation toward oil, healthcare, consumer staples",
            fontsize=14, color=BEAR_ACCENT,
            ha="center", va="center",
            fontweight="bold", family="sans-serif")

    # Implication
    ax.text(hypo_x, hypo_y - 4.0,
            "This is what an interpretable policy SHOULD do  ·  delivers the brief's explainability requirement directly.",
            fontsize=13, color=TEXT,
            ha="center", va="center", family="sans-serif")
    ax.text(hypo_x, hypo_y - 8.0,
            "Same agent, no manual rules — the rotation emerges from training alone.",
            fontsize=13, color=TEXT,
            ha="center", va="center",
            style="italic", family="sans-serif")

    # ---------------------- Caveat ------------------------------------
    ax.text(
        50, 4.0,
        "Attention weights are correlational diagnostics, not causal explanations  (Jain 2019  ·  Wiegreffe 2019).",
        fontsize=11, color=SUBTEXT, ha="center", va="center",
        style="italic", family="sans-serif",
    )

    # ---------------------- Save and flatten alpha --------------------
    out_path = Path(__file__).parent / "slide25_attention_v2.png"
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
