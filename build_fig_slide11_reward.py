"""Generate a navy-background figure for Slide 11: the reward equation.

Three-term equation visualised with annotations below each term and a
small λ-by-regime table hanging off the penalty term, which is the one
that adapts. Designed to teach the reward in 5-10 seconds.
"""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


# Palette
BG = "#0E1B3A"
TEXT = "white"
TITLE = "#7DC8FF"
SUBTEXT = "#C9D5EA"
ACCENT_FACE = "#26477E"          # highlight for the adaptive term
ACCENT_EDGE = "#7DC8FF"
NEUTRAL_FACE = "#16264F"


def main():
    fig = plt.figure(figsize=(16, 9), dpi=200)
    ax = fig.add_subplot(111)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("auto")
    ax.axis("off")

    # ----------------------------- Title ------------------------------
    ax.text(
        50, 93,
        "AN ADAPTIVE REWARD",
        fontsize=24, color=TITLE, ha="center", va="center",
        fontweight="bold", family="sans-serif",
    )
    ax.text(
        50, 87,
        "Three terms — only one adapts with the regime",
        fontsize=13, color=SUBTEXT, ha="center", va="center",
        style="italic", family="sans-serif",
    )

    # --------------------------- Helpers ------------------------------
    def term_box(x, y, w, h, label, face=NEUTRAL_FACE, edge=ACCENT_EDGE,
                 label_size=15, weight="bold", lw=1.8, alpha=0.95):
        b = FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.4,rounding_size=0.9",
            facecolor=face, edgecolor=edge, linewidth=lw, alpha=alpha,
        )
        ax.add_patch(b)
        ax.text(x, y, label,
                fontsize=label_size, color=TEXT,
                ha="center", va="center",
                fontweight=weight, family="sans-serif")

    def operator(x, y, sym, size=22):
        ax.text(x, y, sym,
                fontsize=size, color=TITLE,
                ha="center", va="center",
                fontweight="bold", family="sans-serif")

    def arrow(x1, y1, x2, y2):
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1), (x2, y2),
                arrowstyle="-|>,head_width=0.4,head_length=0.6",
                color=SUBTEXT, linewidth=1.5,
                connectionstyle="arc3,rad=0",
                shrinkA=2, shrinkB=2,
            )
        )

    # ---------------------------- Equation ----------------------------
    eq_y = 70
    term_h = 9
    # REWARD = on the far left
    ax.text(7, eq_y, "REWARD  =",
            fontsize=16, color=TITLE, ha="left", va="center",
            fontweight="bold", family="sans-serif")

    # Term 1: SMOOTH SHARPE
    term_box(28, eq_y, 22, term_h, "SMOOTH SHARPE")
    operator(43, eq_y, "−")

    # Term 2: λ × LOSS  (highlighted)
    term_box(58, eq_y, 22, term_h, "λ × LOSS",
             face=ACCENT_FACE, edge=ACCENT_EDGE, lw=2.6)
    operator(73, eq_y, "−")

    # Term 3: TRANSACTION COST
    term_box(86, eq_y, 25, term_h, "TRANSACTION COST")

    # ------------------------- Annotations -----------------------------
    annot_y = 50
    # arrows from each term down to annotation
    arrow(28, 65, 28, annot_y + 4)
    arrow(58, 65, 58, annot_y + 4)
    arrow(86, 65, 86, annot_y + 4)

    ax.text(28, annot_y,
            "rewards stable\nrisk-adjusted\nreturns",
            fontsize=12, color=SUBTEXT, ha="center", va="top",
            family="sans-serif")

    ax.text(58, annot_y,
            "ADAPTIVE\nchanges with\nthe regime",
            fontsize=12, color=TITLE, ha="center", va="top",
            fontweight="bold", family="sans-serif")

    ax.text(86, annot_y,
            "10 bps per\nunit of\nturnover",
            fontsize=12, color=SUBTEXT, ha="center", va="top",
            family="sans-serif")

    # ---------------------- λ-by-regime table -------------------------
    # Anchored beneath the penalty term to show clearly which term it explains
    table_y = 17
    table_w = 70
    table_h = 14
    table_box = FancyBboxPatch(
        (58 - table_w / 2, table_y - table_h / 2),
        table_w, table_h,
        boxstyle="round,pad=0.5,rounding_size=0.9",
        facecolor=ACCENT_FACE, edgecolor=ACCENT_EDGE,
        linewidth=2.0, alpha=0.92,
    )
    ax.add_patch(table_box)

    # Header
    ax.text(58, table_y + 4.5,
            "λ  (penalty coefficient)",
            fontsize=13, color=TITLE, ha="center", va="center",
            fontweight="bold", family="sans-serif")

    # Three columns inside the table
    col_x = [38, 58, 78]
    labels = ["BULL", "SIDEWAYS", "BEAR"]
    vals = ["0.1", "0.3", "0.5"]
    colors = ["#3FBF7F", "#C9D5EA", "#E66060"]
    for cx, name, val, c in zip(col_x, labels, vals, colors):
        ax.text(cx, table_y + 0.4, name,
                fontsize=12, color=c, ha="center", va="center",
                fontweight="bold", family="sans-serif")
        ax.text(cx, table_y - 3.0, val,
                fontsize=18, color=TEXT, ha="center", va="center",
                fontweight="bold", family="sans-serif")

    # Footer line under the table
    ax.text(58, 6,
            "bear is 5× heavier than bull — same loss penalised 5× more",
            fontsize=12, color=SUBTEXT, ha="center", va="center",
            style="italic", family="sans-serif")

    # ---------------------- Save --------------------------------------
    out_path = Path(__file__).parent / "slide11_reward.png"
    plt.savefig(out_path, dpi=220, bbox_inches="tight",
                facecolor=BG, pad_inches=0.25)
    plt.close(fig)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
