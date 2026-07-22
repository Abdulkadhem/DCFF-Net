"""
Figure 1 -- the problem this paper addresses.

Deliberately non-technical: it states the task, then the two things that are
still missing once a detector is accurate. All imagery is real LEVIR-CD data
and the model's own output; the two numbers in the lower-right panel are our
measured in-domain and zero-shot scores.

    python paper/extract_flow_assets.py     # once, to make the imagery
    python paper/make_problem.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import *                                          # noqa: F401,F403

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
A = os.path.join(FIG, "_assets")


def tile(ax, name, cx, cy, edge, zoom, size):
    if not photo(ax, os.path.join(A, name), cx, cy, zoom, ec=edge, lw=2.8):
        slab(ax, cx, cy, size, size, 0.1, "#dddddd")


def main():
    fig, ax = plt.subplots(figsize=(19.5, 15.2))
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 156)
    ax.axis("off")

    IMG, ZOOM = 24.0, 0.505

    # ================================================== BAND A -- the task
    panel(ax, 4, 86, 192, 66, "#f8fafc", "#8f9bb0")
    ax.text(100, 149.5, "The task", ha="center", va="top", fontsize=18,
            fontweight="bold", color="#33465e", zorder=9)
    ax.text(100, 143.5,
            "given two images of the same place taken at different times, "
            "produce a map of what changed",
            ha="center", va="top", fontsize=13.5, style="italic",
            fontweight="bold", color="#5b6472", zorder=9)

    y1, y2, ym = 125.0, 100.0, 112.5
    tile(ax, "flow_t1.png", 46, y1, "#555555", ZOOM, IMG)
    tile(ax, "flow_t2.png", 46, y2, "#555555", ZOOM, IMG)
    ax.text(31, y1, "$T_1$\nearlier", ha="right", va="center", fontsize=14,
            fontweight="bold", zorder=9)
    ax.text(31, y2, "$T_2$\nlater", ha="right", va="center", fontsize=14,
            fontweight="bold", zorder=9)

    rbox(ax, 104, ym, 40, 16, "#ffffff", "#33465e", "change\ndetector", fs=15)
    arrow(ax, (46 + IMG / 2 + 2, y1), (84, ym + 3.5), lw=2.6, ms=17)
    arrow(ax, (46 + IMG / 2 + 2, y2), (84, ym - 3.5), lw=2.6, ms=17)

    tile(ax, "flow_gt.png", 158, ym, "#2f7a4a", ZOOM, IMG)
    arrow(ax, (124, ym), (158 - IMG / 2 - 2, ym), lw=3.0, ms=19)
    ax.text(158, ym - IMG / 2 - 3.0, "required output:\nthe change map",
            ha="center", va="top", fontsize=14, fontweight="bold",
            color="#2f7a4a", zorder=9)

    # ============================ BAND B -- what accuracy alone leaves open
    ax.text(100, 82.0, "What remains open once a detector is accurate",
            ha="center", va="top", fontsize=18, fontweight="bold",
            color="#33465e", zorder=9)

    # ---- B1: no control over false alarms --------------------------------
    panel(ax, 4, 6, 94, 68, BG_CONF, C_CONF)
    ax.text(51, 71.0, "1.  No control over false alarms", ha="center",
            va="top", fontsize=16, fontweight="bold", color="#c0324f", zorder=9)
    ax.text(51, 65.0, "the threshold is fixed at 0.5 by convention",
            ha="center", va="top", fontsize=12.5, style="italic",
            fontweight="bold", color="#5b6472", zorder=9)

    yB = 45.0
    tile(ax, "flow_pred.png", 28, yB, "#9aa0a6", ZOOM, IMG)
    tile(ax, "flow_err.png", 74, yB, C_CONF, ZOOM, IMG)
    arrow(ax, (28 + IMG / 2 + 2, yB), (74 - IMG / 2 - 2, yB), lw=2.4, ms=16)
    ax.text(28, yB - IMG / 2 - 2.6, "what the model flags", ha="center",
            va="top", fontsize=12.5, fontweight="bold", color=GREY, zorder=9)
    ax.text(74, yB - IMG / 2 - 2.6, "green correct,  red wrong", ha="center",
            va="top", fontsize=12.5, fontweight="bold", color="#c0324f", zorder=9)

    ax.text(51, 16.0,
            "How many of the flagged pixels are wrong?\n"
            "No existing detector can bound that fraction\n"
            "before it is deployed.",
            ha="center", va="center", fontsize=13.5, fontweight="bold",
            linespacing=1.55, color="#c0324f", zorder=9)

    # ---- B2: accuracy does not predict deployment ------------------------
    panel(ax, 102, 6, 94, 68, "#fff8e8", "#c39a2e")
    ax.text(149, 71.0, "2.  Accuracy does not predict deployment",
            ha="center", va="top", fontsize=16, fontweight="bold",
            color="#a8801c", zorder=9)
    ax.text(149, 65.0, "one trained model, two deployments", ha="center",
            va="top", fontsize=12.5, style="italic", fontweight="bold",
            color="#5b6472", zorder=9)

    b0, by, bw, bh = 128.0, 32.0, 20.0, 26.0
    for x, val, lab, col in [(b0, 90.0, "the imagery it\nwas trained on", "#3a8c4f"),
                             (b0 + 36, 3.0, "imagery from a\ndifferent sensor", "#d9534f")]:
        h = max(val / 100.0 * bh, 0.6)
        ax.add_patch(Rectangle((x, by), bw, h, fc=col, ec=INK, lw=1.5, zorder=5))
        ax.text(x + bw / 2, by + h + 1.4,
                ("%.0f" % val) if val > 10 else ("%.1f" % val),
                ha="center", va="bottom", fontsize=16, fontweight="bold",
                color=col, zorder=9)
        ax.text(x + bw / 2, by - 2.2, lab, ha="center", va="top", fontsize=12.5,
                fontweight="bold", color="#5b6472", zorder=9)
    ax.plot([b0 - 6, b0 + 62], [by, by], color=INK, lw=1.8, zorder=6)
    ax.text(b0 - 9.0, by + bh / 2, "F1 score", ha="center", va="center",
            fontsize=13, fontweight="bold", rotation=90, color="#5b6472", zorder=9)

    ax.text(149, 16.0,
            "An in-domain score says nothing about this collapse,\n"
            "and gives the user no warning that it has happened.",
            ha="center", va="center", fontsize=13.5, fontweight="bold",
            linespacing=1.55, color="#a8801c", zorder=9)

    fig.tight_layout(pad=0.3)
    out = os.path.join(FIG, "fig_problem.png")
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main()
