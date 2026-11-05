"""Generate the attention-diagnostics infographic for Slide 25.

Two market cards (DJI / SP100). Each card has:
  - TOP 5 attended in BULL (green panel)
  - TOP 5 attended in BEAR (red panel)
Side by side so the audience reads both regimes AND the rotation
implied by which stocks appear/disappear between them.
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
BULL_FACE = "#1B4530"
BEAR_FACE = "#4C2027"
BULL_ACCENT = "#3FBF7F"
BEAR_ACCENT = "#E66060"
HYPO_FACE = "#26477E"
ACCENT_CYAN = "#7DC8FF"
BOTH_BADGE = "#FFC97A"           # amber for stocks attended in both regimes


# Tickers + sector descriptors from §4.4 / Table 4.6 of the thesis
# A "both" tag marks stocks that appear in the top-5 of both regimes,
# so the audience can see the always-attended core vs the regime-specific ones.
DJI_BULL = [
    ("NVDA", "semis  ·  growth",   "both"),
    ("MMM",  "industrials",        "both"),
    ("INTC", "semis",              "bull-only"),
    ("IBM",  "enterprise tech",    "bull-only"),
    ("BA",   "industrials",        "both"),
]
DJI_BEAR = [
    ("NVDA", "semis  ·  growth",   "both"),
    ("MMM",  "industrials",        "both"),
    ("BA",   "industrials",        "both"),
    ("CVX",  "oil  ·  energy",     "bear-only"),
    ("AAPL", "consumer tech",      "bear-only"),
]
SP100_BULL = [
    ("NVDA", "semis  ·  growth",   "both"),
    ("LLY",  "healthcare",         "both"),
    ("TGT",  "consumer",           "bull-only"),
    ("DE",   "agri-equipment",     "both"),
    ("ORCL", "enterprise tech",    "bull-only"),
]
SP100_BEAR = [
    ("NVDA", "semis  ·  growth",   "both"),
    ("LLY",  "healthcare",         "both"),
    ("COP",  "oil  ·  energy",     "bear-only"),
    ("COST", "consumer staples",   "bear-only"),
    ("DE",   "agri-equipment",     "both"),
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
    fig = plt.figure(figsize=(17, 11), dpi=240)
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
        "ATTENTION DIAGNOSTICS  ·  WHERE THE AGENT LOOKS",
        fontsize=21, color=TITLE, ha="center", va="center",
        fontweight="bold", family="sans-serif",
    )
    ax.text(
        50, 90,
        "Top 5 attended stocks in each regime — same agent, different rotation",
        fontsize=13, color=SUBTEXT, ha="center", va="center",
        style="italic", family="sans-serif",
    )

    # ----------------------------- Columns ----------------------------
    col_xs = [27, 73]
    col_w = 44
    col_top = 87
    col_bottom = 32

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

        # Market header
        ax.text(cx, col_top - 4.0, col["name"],
                fontsize=20, color=col["accent"],
                ha="center", va="center",
                fontweight="bold", family="sans-serif")

        # ---- BULL panel ----
        bull_x = cx - col_w * 0.24
        bull_w = col_w * 0.43
        panel_top = col_top - 9
        panel_h = 42
        bull_box = FancyBboxPatch(
            (bull_x - bull_w / 2, panel_top - panel_h),
            bull_w, panel_h,
            boxstyle="round,pad=0.3,rounding_size=0.6",
            facecolor=BULL_FACE, edgecolor=BULL_ACCENT,
            linewidth=1.6, alpha=0.95,
        )
        ax.add_patch(bull_box)
        ax.text(bull_x, panel_top - 3.5, "TOP 5 IN BULL",
                fontsize=12, color=BULL_ACCENT,
                ha="center", va="center",
                fontweight="bold", family="sans-serif")

        for i, (ticker, sector, _tag) in enumerate(col["bull"]):
            y = panel_top - 8.5 - i * 5.2
            ax.text(bull_x - bull_w * 0.36, y, ticker,
                    fontsize=13, color=TEXT,
                    ha="left", va="center",
                    fontweight="bold", family="sans-serif")
            ax.text(bull_x + bull_w * 0.02, y, sector,
                    fontsize=10, color=SUBTEXT,
                    ha="left", va="center",
                    family="sans-serif")

        # Bottom share line for bull panel
        ax.text(bull_x, panel_top - panel_h + 3.5,
                f"top-5 attention share:  {col['bull_share']}",
                fontsize=11, color=BULL_ACCENT,
                ha="center", va="center",
                fontweight="bold", family="sans-serif")

        # ---- BEAR panel ----
        bear_x = cx + col_w * 0.24
        bear_w = col_w * 0.43
        bear_box = FancyBboxPatch(
            (bear_x - bear_w / 2, panel_top - panel_h),
            bear_w, panel_h,
            boxstyle="round,pad=0.3,rounding_size=0.6",
            facecolor=BEAR_FACE, edgecolor=BEAR_ACCENT,
            linewidth=1.6, alpha=0.95,
        )
        ax.add_patch(bear_box)
        ax.text(bear_x, panel_top - 3.5, "TOP 5 IN BEAR",
                fontsize=12, color=BEAR_ACCENT,
                ha="center", va="center",
                fontweight="bold", family="sans-serif")

        for i, (ticker, sector, _tag) in enumerate(col["bear"]):
            y = panel_top - 8.5 - i * 5.2
            ax.text(bear_x - bear_w * 0.36, y, ticker,
                    fontsize=13, color=TEXT,
                    ha="left", va="center",
                    fontweight="bold", family="sans-serif")
            ax.text(bear_x + bear_w * 0.02, y, sector,
                    fontsize=10, color=SUBTEXT,
                    ha="left", va="center",
                    family="sans-serif")

        # Bottom share line for bear panel
        ax.text(bear_x, panel_top - panel_h + 3.5,
                f"top-5 attention share:  {col['bear_share']}",
                fontsize=11, color=BEAR_ACCENT,
                ha="center", va="center",
                fontweight="bold", family="sans-serif")

        # Uniform reference line beneath the two panels
        ax.text(cx, panel_top - panel_h - 2.5,
                f"uniform attention would be {col['uniform']}  ·  the agent concentrates well above that",
                fontsize=10, color=SUBTEXT,
                ha="center", va="center",
                style="italic", family="sans-serif")

    # ---------------------- Verdict block ----------------------------
    hypo_x, hypo_y, hypo_w, hypo_h = 50, 13, 86, 14
    hypo_box = FancyBboxPatch(
        (hypo_x - hypo_w / 2, hypo_y - hypo_h / 2), hypo_w, hypo_h,
        boxstyle="round,pad=0.5,rounding_size=1.0",
        facecolor=HYPO_FACE, edgecolor=TITLE,
        linewidth=2.0, alpha=0.95,
    )
    ax.add_patch(hypo_box)
    ax.text(hypo_x, hypo_y + 4.5,
            "ECONOMICALLY INTERPRETABLE ROTATION",
            fontsize=13, color=TITLE,
            ha="center", va="center",
            fontweight="bold", family="sans-serif")
    ax.text(hypo_x, hypo_y + 1.0,
            "Bull:  attention favours growth tech and industrials.    Bear:  rotation toward oil, healthcare, consumer staples.",
            fontsize=11, color=TEXT,
            ha="center", va="center", family="sans-serif")
    ax.text(hypo_x, hypo_y - 2.5,
            "This is what an interpretable policy should do — delivering the brief's explainability requirement directly.",
            fontsize=11, color=TEXT,
            ha="center", va="center", family="sans-serif")

    # ---------------------- Caveat ------------------------------------
    ax.text(
        50, 3,
        "Attention weights are correlational diagnostics, not causal explanations (Jain 2019, Wiegreffe 2019).",
        fontsize=9, color=SUBTEXT, ha="center", va="center",
        style="italic", family="sans-serif",
    )

    # ---------------------- Save and flatten alpha --------------------
    out_path = Path(__file__).parent / "slide25_attention.png"
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
