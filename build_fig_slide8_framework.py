"""Generate the end-to-end framework diagram for Slide 8 (new).

Mirrors thesis Figure 3.1: a left-side HMM column feeding into a central
pipeline that runs from observation → agent → softmax → cash-floor →
environment → reward, with a dashed training-only feedback arrow from
reward back to agent.

Solid navy background matching the user's Canva template, high DPI.
"""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


# ---------------------------- Palette ----------------------------
BG = "#0E1B3A"
TEXT = "white"
TITLE = "#7DC8FF"
SUBTEXT = "#C9D5EA"

# Subtle category colours that all read on navy
DATA_FACE   = "#1A2A52"   # light steel for raw data
HMM_FACE    = "#2B2360"   # subtle violet for HMM column
AGENT_FACE  = "#26477E"   # brighter cyan-blue for the agent (highlighted)
PIPE_FACE   = "#16264F"   # neutral pipeline boxes
SAFETY_FACE = "#5A4218"   # warm amber for safety overlay
ENV_FACE    = "#194730"   # subtle green for environment
REWARD_FACE = "#5A3A18"   # warm orange-brown for reward

EDGE = "#7DC8FF"
EDGE_DIM = "#8E9DBC"
ARROW = "#7DC8FF"
TRAINING_ARROW = "#E8B057"  # warm tone for the dashed training loop


def main():
    fig = plt.figure(figsize=(16, 10), dpi=240)
    ax = fig.add_subplot(111)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("auto")
    ax.axis("off")

    # ----------------------------- Title ------------------------------
    ax.text(
        50, 96,
        "FRAMEWORK AT A GLANCE",
        fontsize=22, color=TITLE, ha="center", va="center",
        fontweight="bold", family="sans-serif",
    )
    ax.text(
        50, 91.5,
        "End-to-end pipeline of the regime-aware DRL portfolio agent",
        fontsize=13, color=SUBTEXT, ha="center", va="center",
        style="italic", family="sans-serif",
    )

    # --------------------------- Helpers ------------------------------
    def box(x, y, w, h, label, sub=None,
            face=PIPE_FACE, edge=EDGE, lw=1.8,
            label_size=12, sub_size=10, label_color=None, weight="bold"):
        b = FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.35,rounding_size=0.7",
            facecolor=face, edgecolor=edge, linewidth=lw, alpha=0.95,
        )
        ax.add_patch(b)
        col = label_color or TITLE
        if sub:
            ax.text(x, y + h * 0.18, label,
                    fontsize=label_size, color=col,
                    ha="center", va="center",
                    fontweight=weight, family="sans-serif")
            ax.text(x, y - h * 0.22, sub,
                    fontsize=sub_size, color=SUBTEXT,
                    ha="center", va="center", family="sans-serif")
        else:
            ax.text(x, y, label,
                    fontsize=label_size, color=col,
                    ha="center", va="center",
                    fontweight=weight, family="sans-serif")

    def arrow(x1, y1, x2, y2, color=ARROW, lw=1.8, style="-|>",
              rad=0.0, dashed=False, head_w=0.4, head_l=0.6):
        linestyle = "--" if dashed else "-"
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1), (x2, y2),
                arrowstyle=f"{style},head_width={head_w},head_length={head_l}",
                color=color, linewidth=lw, linestyle=linestyle,
                connectionstyle=f"arc3,rad={rad}",
                shrinkA=2, shrinkB=2,
            )
        )

    # ============== LEFT COLUMN: HMM path ==============
    hmm_x = 17
    box(hmm_x, 82, 24, 7, "Price + VIX",
        face=DATA_FACE, label_size=11)
    box(hmm_x, 71, 24, 7, "HMM regime classifier",
        face=HMM_FACE, edge=EDGE, label_size=12)
    box(hmm_x, 60, 24, 8, "Regime label",
        sub="+ posterior probabilities",
        face=HMM_FACE, edge=EDGE, label_size=12)

    # Arrows down the HMM column
    arrow(hmm_x, 78.5, hmm_x, 74.5)
    arrow(hmm_x, 67.5, hmm_x, 64)

    # ============== CENTER PIPELINE ==============
    cx = 58       # x-coord of the main pipeline
    in_y = 82     # row of input feature boxes

    # Three input feature boxes at the top of the pipeline (no overlap)
    input_offsets = (-16, 0, 16)
    box(cx + input_offsets[0], in_y, 14, 7, "Per-stock\nfeatures",
        face=DATA_FACE, label_size=10)
    box(cx + input_offsets[1], in_y, 14, 7, "Broad-mkt\nvol",
        face=DATA_FACE, label_size=10)
    box(cx + input_offsets[2], in_y, 14, 7, "Portfolio\nweights",
        face=DATA_FACE, label_size=10)

    # Observation
    obs_y = 70
    box(cx, obs_y, 38, 6.5, "Observation",
        sub=r"$\mathbf{o}_t$",
        face=PIPE_FACE, edge=EDGE, lw=2.2, label_size=13)

    # Agent (highlighted)
    agent_y = 58
    box(cx, agent_y, 38, 7.5, "Agent  ·  PPO + Attention",
        sub="policy + value heads",
        face=AGENT_FACE, edge=EDGE, lw=2.6, label_size=14)

    # Softmax
    sm_y = 46
    box(cx, sm_y, 38, 6.5,
        r"Softmax  →  proposed weights  $\tilde{\mathbf{w}}_t$",
        face=PIPE_FACE, edge=EDGE, lw=1.8, label_size=12)

    # Cash-floor (safety overlay) — amber
    cf_y = 34
    box(cx, cf_y, 38, 7, "Cash-floor overlay",
        sub="bear regime only",
        face=SAFETY_FACE, edge="#E8B057", lw=2.2, label_size=13,
        label_color="#FFC97A")

    # Environment
    env_y = 22
    box(cx, env_y, 38, 7, "Environment",
        sub="apply weights · charge cost · evolve portfolio value",
        face=ENV_FACE, edge="#6BCB94", lw=2.0, label_size=13,
        label_color="#9FE8BD")

    # Reward
    rew_y = 10
    box(cx, rew_y, 38, 7,
        "Reward",
        sub=r"$r_t = $ DSR  $-\ \lambda(\mathrm{regime}) \cdot $loss  $-$ cost",
        face=REWARD_FACE, edge="#E8B057", lw=2.0, label_size=13,
        label_color="#FFC97A")

    # ============== INTERNAL CENTER-PIPELINE ARROWS ==============
    # input rows → observation
    for off in input_offsets:
        arrow(cx + off, in_y - 3.5, cx + off, obs_y + 3.25)

    # observation → agent → softmax → cash-floor → env → reward
    arrow(cx, obs_y - 3.25, cx, agent_y + 3.75)
    arrow(cx, agent_y - 3.75, cx, sm_y + 3.25)
    arrow(cx, sm_y - 3.25, cx, cf_y + 3.5)
    arrow(cx, cf_y - 3.5, cx, env_y + 3.5)
    arrow(cx, env_y - 3.5, cx, rew_y + 3.5)

    # ============== HMM → CENTER LINKS ==============
    # Regime label → observation (the soft input)
    arrow(hmm_x + 12, 60, cx - 19, obs_y, rad=0.0)
    # Regime label → cash-floor (the hard switch for safety)
    arrow(hmm_x + 12, 60 - 0.8, cx - 19, cf_y + 0.5, rad=-0.18,
          color=EDGE_DIM, lw=1.6)
    # small label on the second arrow
    ax.text(35, 46, "(activates floor)",
            fontsize=9, color=SUBTEXT, ha="center", va="center",
            style="italic", family="sans-serif")

    # ============== TRAINING-ONLY DASHED FEEDBACK ==============
    # Reward → Agent loop (dashed, on the right side)
    fb_x = cx + 22
    # right segment from reward box to fb_x
    ax.plot([cx + 19, fb_x], [rew_y, rew_y],
            color=TRAINING_ARROW, linewidth=1.8, linestyle="--")
    # vertical segment up to agent height
    ax.plot([fb_x, fb_x], [rew_y, agent_y],
            color=TRAINING_ARROW, linewidth=1.8, linestyle="--")
    # arrowhead segment from fb_x back to the agent box
    arrow(fb_x, agent_y, cx + 19, agent_y, color=TRAINING_ARROW,
          lw=1.8, style="-|>", head_w=0.4, head_l=0.6)

    ax.text(fb_x + 1.5, (rew_y + agent_y) / 2,
            "PPO\nupdate\n(training\nonly)",
            fontsize=10, color=TRAINING_ARROW,
            ha="left", va="center", family="sans-serif",
            fontweight="bold")

    # (Legend removed — labels are self-explanatory and were crowding the
    # input boxes on the right.)

    # ---------------------- Save --------------------------------------
    out_path = Path(__file__).parent / "slide8_framework.png"
    plt.savefig(out_path, dpi=320, bbox_inches="tight",
                facecolor=BG, pad_inches=0.25)
    plt.close(fig)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
