"""
Figure 2 -- what actually happens to an image inside DCFF-Net.

Every panel is real: one LEVIR-CD test tile is pushed through the trained
checkpoint and each intermediate tensor is rendered (L2 norm over channels,
contrast-stretched). Nothing here is a schematic stand-in.

    python paper/extract_flow_assets.py     # once, to make the imagery
    python paper/make_dualcue_v2.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import *                                          # noqa: F401,F403

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
A = os.path.join(FIG, "_assets")

IMG = 26.0                       # displayed edge of every panel, in axis units
Y_HI, Y_LO, Y_MID = 72.0, 35.0, 53.5
ZOOM = 0.547                     # 256 px source -> IMG units


def tile(ax, name, cx, cy, edge_col, caption, cap_col=INK, fs=13.5, above=False):
    """one real image panel with a coloured frame and a bold caption."""
    p = os.path.join(A, name)
    if not photo(ax, p, cx, cy, ZOOM, ec=edge_col, lw=2.6):
        slab(ax, cx, cy, IMG, IMG, 0.1, "#dddddd")
    dy, va = (IMG / 2 + 2.4, "bottom") if above else (-IMG / 2 - 2.4, "top")
    ax.text(cx, cy + dy, caption, ha="center", va=va,
            fontsize=fs, fontweight="bold", color=cap_col, zorder=9)


def header(ax, cx, txt, col):
    ax.text(cx, 101.0, txt, ha="center", va="center", fontsize=16.5,
            fontweight="bold", color=col, zorder=9)


def main():
    fig, ax = plt.subplots(figsize=(22.0, 10.0))
    ax.set_xlim(0, 244)
    ax.set_ylim(0, 108)
    ax.axis("off")

    C1, C2, C3, C4, C5, C6, C7 = 20, 58, 96, 134, 168, 202, 232

    # ------------------------------------------------------------ headers
    header(ax, C1, "Input", GREY)
    header(ax, C2, "Shared encoder", C_T2)
    header(ax, C3, "Dual-cue fusion  (ours)", "#a8701f")
    header(ax, C4, "Fused", "#a83232")
    header(ax, C5, "Decoder", C_DEC)
    header(ax, (C6 + C7) / 2, "Output", "#a8801c")

    # ------------------------------------------------------------- stage 1
    tile(ax, "flow_t1.png", C1, Y_HI, "#555555", "$T_1$   before")
    tile(ax, "flow_t2.png", C1, Y_LO, "#555555", "$T_2$   after")

    # ------------------------------------------------------------- stage 2
    tile(ax, "flow_enc1.png", C2, Y_HI, C_T1, "$F^{(1)}$", "#2c4d8f")
    tile(ax, "flow_enc2.png", C2, Y_LO, C_T2, "$F^{(2)}$", "#1e4a9c")
    arrow(ax, (C1 + IMG / 2 + 1, Y_HI), (C2 - IMG / 2 - 1, Y_HI), lw=2.4, ms=16)
    arrow(ax, (C1 + IMG / 2 + 1, Y_LO), (C2 - IMG / 2 - 1, Y_LO), lw=2.4, ms=16)

    # ------------------------------------------------------------- stage 3
    tile(ax, "flow_diff.png", C3, Y_HI, C_DIFF, "difference cue", "#a4560f")
    tile(ax, "flow_conc.png", C3, Y_LO, C_CONC, "concatenation cue", "#5f3a96")

    # a shared bus, because BOTH encoder streams feed BOTH cues
    bus = (C2 + C3) / 2.0
    ax.plot([bus, bus], [Y_LO, Y_HI], color="#555555", lw=2.4,
            solid_capstyle="round", zorder=5)
    arrow(ax, (C2 + IMG / 2 + 1, Y_HI), (bus, Y_HI), lw=2.4, ms=15)
    arrow(ax, (C2 + IMG / 2 + 1, Y_LO), (bus, Y_LO), lw=2.4, ms=15)
    arrow(ax, (bus, Y_HI), (C3 - IMG / 2 - 1, Y_HI), color="#a4560f", lw=2.4, ms=15)
    arrow(ax, (bus, Y_LO), (C3 - IMG / 2 - 1, Y_LO), color="#5f3a96", lw=2.4, ms=15)
    # (the bus reads for itself: two streams in, two cues out; the wording
    #  lives in the caption strip so the diagram stays uncluttered)

    # ------------------------------------------------------------- stage 4
    tile(ax, "flow_fused.png", C4, Y_MID, C_FUSED, "$\\widetilde{F}$", "#a83232",
         fs=15)
    arrow(ax, (C3 + IMG / 2 + 1, Y_HI - 3), (C4 - IMG / 2 - 1, Y_MID + 5),
          color="#a4560f", lw=2.6, ms=16)
    arrow(ax, (C3 + IMG / 2 + 1, Y_LO + 3), (C4 - IMG / 2 - 1, Y_MID - 5),
          color="#5f3a96", lw=2.6, ms=16)

    # ------------------------------------------------------------- stage 5
    tile(ax, "flow_dec.png", C5, Y_MID, C_DEC, "ASPP + U-Net", C_DEC)
    arrow(ax, (C4 + IMG / 2 + 1, Y_MID), (C5 - IMG / 2 - 1, Y_MID), lw=2.6, ms=16)

    # ------------------------------------------------------------- stage 6
    tile(ax, "flow_prob.png", C6, Y_HI, "#c39a2e", "probability  $\\hat{p}$",
         "#a8801c", above=True)
    tile(ax, "flow_pred.png", C6, Y_LO, "#c39a2e", "our prediction", "#a8801c")
    tile(ax, "flow_gt.png", C7, Y_LO, "#777777", "ground truth", GREY)
    arrow(ax, (C5 + IMG / 2 + 1, Y_MID + 4), (C6 - IMG / 2 - 1, Y_HI - 4),
          lw=2.6, ms=16)
    arrow(ax, (C6, Y_HI - IMG / 2 - 1), (C6, Y_LO + IMG / 2 + 1),
          lw=2.4, ms=15)
    ax.text(C6 + 2.4, Y_MID, "threshold", ha="left", va="center",
            fontsize=12, fontweight="bold", style="italic", color="#555555",
            zorder=9)

    # ------------------------------------------------------------- caption
    ax.add_patch(FancyBboxPatch((14, 1.5), 216, 14.0,
                                boxstyle="round,pad=0.7,rounding_size=2",
                                fc="#f7f9fc", ec="#9aa8bd", lw=1.5, zorder=1))
    ax.text(122, 8.5,
            "Every panel is the trained model's own response on this tile "
            "(F1 = 0.935).   Both encoder streams feed both cues:\n"
            "the difference cue reacts to change magnitude, "
            "the concatenation cue to relational context.",
            ha="center", va="center", fontsize=13.5, fontweight="bold",
            color="#33465e", zorder=9)

    fig.tight_layout(pad=0.3)
    out = os.path.join(FIG, "fig_dualcue.png")
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main()
