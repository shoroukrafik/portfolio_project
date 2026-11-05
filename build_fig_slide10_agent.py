"""Generate a navy-background architecture diagram for Slide 10.

Vertical flow showing the agent's pipeline:
observation -> per-stock path (encoder + attention + pool) | context path (MLP)
            -> combine -> feature summary
            -> policy head + value head -> portfolio weights, state value.

Labels are plain English; the only mathy notation is the highlighted attention
box which keeps "Multi-head attention" intact because it is the centrepiece
of the design.
"""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


# Palette matched to the user's Canva template
BG = "#0E1B3A"
TEXT = "white"
TITLE = "#7DC8FF"
SUBTEXT = "#C9D5EA"
BOX_EDGE = "#7DC8FF"
BOX_FACE = "#16264F"
ACCENT_FACE = "#26477E"        # brighter for the attention block
ARROW = "#7DC8FF"


def main():
    fig = plt.figure(figsize=(14, 12), dpi=200)
    ax = fig.add_subplot(111)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 100)
    ax.set_ylim(-12, 120)
    ax.set_aspect("equal")
    ax.axis("off")

    # ----------------------------- Title ------------------------------
    ax.text(
        50, 115,
        "AGENT ARCHITECTURE",
        fontsize=22, color=TITLE, ha="center", va="center",
        fontweight="bold", family="sans-serif",
    )
    ax.text(
        50, 110,
        "PPO with multi-head self-attention",
        fontsize=13, color=SUBTEXT, ha="center", va="center",
        family="sans-serif",
    )

    # --------------------------- Helpers ------------------------------
    def box(x, y, w, h, label, sub=None, face=BOX_FACE,
            label_size=12, sub_size=10):
        b = FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.4,rounding_size=0.8",
            facecolor=face, edgecolor=BOX_EDGE, linewidth=2.0, alpha=0.95,
        )
        ax.add_patch(b)

        if sub:
            # two-line layout: label upper third, sub lower third
            ax.text(x, y + h * 0.18, label,
                    fontsize=label_size, color=TITLE,
                    ha="center", va="center",
                    fontweight="bold", family="sans-serif")
            ax.text(x, y - h * 0.22, sub,
                    fontsize=sub_size, color=SUBTEXT,
                    ha="center", va="center", family="sans-serif")
        else:
            ax.text(x, y, label,
                    fontsize=label_size, color=TITLE,
                    ha="center", va="center",
                    fontweight="bold", family="sans-serif")

    def arrow(x1, y1, x2, y2):
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1), (x2, y2),
                arrowstyle="-|>,head_width=0.4,head_length=0.7",
                color=ARROW, linewidth=2.0,
                connectionstyle="arc3,rad=0",
                shrinkA=2, shrinkB=2,
            )
        )

    # -------------------------- Nodes ---------------------------------
    # Observation at top
    box(50, 100, 32, 7.5, "Observation",
        sub="per-stock features + context")

    # Split into two streams
    box(25, 85, 28, 7.5, "Per-stock matrix",
        sub="one row per stock")
    box(75, 85, 28, 7.5, "Context vector",
        sub="regime + market vol + weights")

    box(25, 72, 28, 7.5, "Shared encoder",
        sub="same weights for every stock")
    box(75, 72, 28, 7.5, "Small MLP")

    # Attention block — highlighted
    box(25, 58, 28, 9.5, "Multi-head attention",
        sub="stocks compare each other",
        face=ACCENT_FACE,
        label_size=13, sub_size=10)

    box(25, 46, 28, 6.5, "Average across stocks")

    box(25, 35, 28, 7, "Stock summary")
    box(75, 35, 28, 7, "Context summary")

    box(50, 22, 30, 7, "Combine")
    box(50, 12, 30, 7, "Feature summary")

    box(25, 1, 28, 7, "Policy head")
    box(75, 1, 28, 7, "Value head")

    # --------------------------- Arrows -------------------------------
    # Observation -> two branches
    arrow(45, 96, 28, 89.5)
    arrow(55, 96, 72, 89.5)

    # Per-stock path
    arrow(25, 81, 25, 76.5)
    arrow(25, 68, 25, 63.5)
    arrow(25, 53, 25, 49.5)
    arrow(25, 42.5, 25, 38.5)

    # Extras path
    arrow(75, 81, 75, 76.5)
    arrow(75, 68, 75, 38.5)

    # Both feed into Combine
    arrow(28, 31.5, 45, 25)
    arrow(72, 31.5, 55, 25)

    # Combine -> Feature summary
    arrow(50, 18.5, 50, 15.5)

    # Feature summary -> two heads
    arrow(45, 9, 28, 4.5)
    arrow(55, 9, 72, 4.5)

    # ---------------------- Output labels under heads -----------------
    ax.text(25, -6, "→  portfolio weights",
            fontsize=11, color=SUBTEXT, ha="center", va="center",
            family="sans-serif")
    ax.text(75, -6, "→  state value",
            fontsize=11, color=SUBTEXT, ha="center", va="center",
            family="sans-serif")

    # ---------------------- Save --------------------------------------
    out_path = Path(__file__).parent / "slide10_agent.png"
    plt.savefig(out_path, dpi=220, bbox_inches="tight",
                facecolor=BG, pad_inches=0.25)
    plt.close(fig)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
