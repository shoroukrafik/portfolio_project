"""Generate a clearer architecture diagram for Slide 10 (V2).

Improvements over V1:
- 16:9 landscape (matches the rest of the deck for Canva)
- Two visually-grouped paths with dashed panels and color-coded accents
  · LEFT (cyan)  = per-stock path : matrix → encoder → attention → pool → summary
  · RIGHT (amber)= context path   : vector → MLP → summary
  · MERGE (green)= concat → feature summary
  · OUTPUTS (purple)= policy head + value head
- Each box has a short "what it does" subtitle so the diagram is readable
  without narration.

Output: slide10_agent_v2.png  (does NOT overwrite slide10_agent.png)
"""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from PIL import Image


# Palette
BG = "#0E1B3A"
TEXT = "white"
TITLE = "#7DC8FF"
SUBTEXT = "#C9D5EA"
BOX_FACE = "#16264F"
ACCENT_CYAN = "#7DC8FF"
ACCENT_AMBER = "#FFC97A"
ACCENT_GREEN = "#3FBF7F"
ACCENT_PURPLE = "#B58CE8"
ATTENTION_FACE = "#26477E"


def main():
    fig = plt.figure(figsize=(19.2, 10.8), dpi=200)
    ax = fig.add_subplot(111)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("auto")
    ax.axis("off")

    # ------------------------- Title --------------------------------
    ax.text(
        50, 96,
        "AGENT ARCHITECTURE",
        fontsize=24, color=TITLE, ha="center", va="center",
        fontweight="bold", family="sans-serif",
    )
    ax.text(
        50, 91.5,
        "PPO actor-critic  ·  shared encoder  ·  multi-head self-attention",
        fontsize=14, color=SUBTEXT, ha="center", va="center",
        style="italic", family="sans-serif",
    )

    # ------------------------- Helpers ------------------------------
    def box(x, y, w, h, label, sub=None, accent=ACCENT_CYAN, face=BOX_FACE,
            label_size=13, sub_size=11):
        b = FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.4,rounding_size=0.8",
            facecolor=face, edgecolor=accent, linewidth=2.2, alpha=0.95,
        )
        ax.add_patch(b)
        if sub:
            ax.text(x, y + h * 0.20, label,
                    fontsize=label_size, color=accent,
                    ha="center", va="center",
                    fontweight="bold", family="sans-serif")
            ax.text(x, y - h * 0.22, sub,
                    fontsize=sub_size, color=SUBTEXT,
                    ha="center", va="center", family="sans-serif")
        else:
            ax.text(x, y, label,
                    fontsize=label_size, color=accent,
                    ha="center", va="center",
                    fontweight="bold", family="sans-serif")

    def arrow(x1, y1, x2, y2, color=ACCENT_CYAN, lw=2.0):
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1), (x2, y2),
                arrowstyle="-|>,head_width=0.4,head_length=0.7",
                color=color, linewidth=lw,
                connectionstyle="arc3,rad=0",
                shrinkA=2, shrinkB=2,
            )
        )

    # ------------------- OBSERVATION at top -------------------------
    box(50, 84, 50, 7, "OBSERVATION",
        sub="per-stock features  +  global context",
        accent=TITLE, label_size=16, sub_size=12)

    # ------------------ Path panels (grouping) ----------------------
    left_panel = Rectangle(
        (8, 26), 34, 53,
        facecolor="none", edgecolor=ACCENT_CYAN,
        linewidth=1.2, alpha=0.55, linestyle="--",
    )
    ax.add_patch(left_panel)
    ax.text(25, 76, "PER-STOCK PATH",
            fontsize=13, color=ACCENT_CYAN, ha="center", va="center",
            fontweight="bold", family="sans-serif")

    right_panel = Rectangle(
        (58, 47), 34, 32,
        facecolor="none", edgecolor=ACCENT_AMBER,
        linewidth=1.2, alpha=0.55, linestyle="--",
    )
    ax.add_patch(right_panel)
    ax.text(75, 76, "CONTEXT PATH",
            fontsize=13, color=ACCENT_AMBER, ha="center", va="center",
            fontweight="bold", family="sans-serif")

    # ------------------ LEFT path boxes -----------------------------
    box(25, 70, 30, 6, "Per-stock matrix",
        sub="N stocks  ×  M features", accent=ACCENT_CYAN)
    box(25, 61, 30, 6, "Shared encoder",
        sub="same MLP for every stock  →  64-dim", accent=ACCENT_CYAN)
    box(25, 51, 30, 8, "MULTI-HEAD ATTENTION",
        sub="stocks compare each other  ·  4 heads",
        accent=ACCENT_CYAN, face=ATTENTION_FACE,
        label_size=14, sub_size=11)
    box(25, 41, 30, 6, "Average across stocks",
        sub="N × 64 matrix  →  64-dim vector",
        accent=ACCENT_CYAN)
    box(25, 32, 30, 6, "Stock summary",
        sub="one 64-dim vector", accent=ACCENT_CYAN)

    # ------------------ RIGHT path boxes ----------------------------
    box(75, 70, 30, 6, "Context vector",
        sub="regime  +  market vol  +  current weights",
        accent=ACCENT_AMBER)
    box(75, 61, 30, 6, "Small MLP",
        sub="2 hidden layers", accent=ACCENT_AMBER)
    box(75, 52, 30, 6, "Context summary",
        sub="one 32-dim vector", accent=ACCENT_AMBER)

    # ------------------ COMBINE + Feature summary -------------------
    box(50, 22, 42, 6, "COMBINE",
        sub="concatenate stock summary  +  context summary",
        accent=ACCENT_GREEN, label_size=14, sub_size=11)
    box(50, 13.5, 42, 5, "Feature summary",
        sub="shared representation read by both heads",
        accent=ACCENT_GREEN, label_size=12, sub_size=10)

    # ------------------ Two heads -----------------------------------
    box(27, 6, 26, 5, "POLICY HEAD",
        sub="actor", accent=ACCENT_PURPLE, label_size=13, sub_size=10)
    box(73, 6, 26, 5, "VALUE HEAD",
        sub="critic", accent=ACCENT_PURPLE, label_size=13, sub_size=10)

    # ------------------ Output labels -------------------------------
    ax.text(27, 1.4,
            "→  portfolio weights  (stocks + cash, sum to 1)",
            fontsize=11, color=SUBTEXT, ha="center", va="center",
            family="sans-serif")
    ax.text(73, 1.4,
            "→  state value  (used by PPO for advantage)",
            fontsize=11, color=SUBTEXT, ha="center", va="center",
            family="sans-serif")

    # ------------------ Arrows --------------------------------------
    arrow(40, 81, 30, 73.5, color=TITLE)
    arrow(60, 81, 70, 73.5, color=TITLE)

    arrow(25, 67, 25, 64, color=ACCENT_CYAN)
    arrow(25, 58, 25, 55, color=ACCENT_CYAN)
    arrow(25, 47, 25, 44, color=ACCENT_CYAN)
    arrow(25, 38, 25, 35, color=ACCENT_CYAN)

    arrow(75, 67, 75, 64, color=ACCENT_AMBER)
    arrow(75, 58, 75, 55, color=ACCENT_AMBER)

    arrow(28, 29, 42, 25, color=ACCENT_GREEN)
    arrow(72, 49, 58, 25, color=ACCENT_GREEN)

    arrow(50, 19, 50, 16, color=ACCENT_GREEN)

    arrow(45, 11, 32, 8.5, color=ACCENT_PURPLE)
    arrow(55, 11, 68, 8.5, color=ACCENT_PURPLE)

    # ------------------ Save and flatten ----------------------------
    out_path = Path(__file__).parent / "slide10_agent_v2.png"
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
