"""
Figure 3 -- the two cues at all four encoder scales.

Companion to Figure 2, which follows a single scale end to end. Here the same
tile is shown at every scale, so that the claim "the block is applied
identically at all four scales" can be checked rather than taken on trust.

Rendering is nearest-neighbour on purpose: the visible grid is the tensor's
true resolution, which is the point of a multi-scale figure.

    python paper/extract_flow_assets.py     # once, to make the imagery
    python paper/make_dualcue_scales.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import *                                          # noqa: F401,F403

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
A = os.path.join(FIG, "_assets")

IMG = 26.0
ZOOM = 0.508
ROWS = [("1", "1/4", "64 x 64"), ("2", "1/8", "32 x 32"),
        ("3", "1/16", "16 x 16"), ("4", "1/32", "8 x 8")]
COLS = [("enc1", "$F^{(1)}_i$", C_T1, "#2c4d8f"),
        ("enc2", "$F^{(2)}_i$", C_T2, "#1e4a9c"),
        ("diff", "difference cue", C_DIFF, "#a4560f"),
        ("conc", "concatenation cue", C_CONC, "#5f3a96"),
        ("fused", "fused  $\\widetilde{F}_i$", C_FUSED, "#a83232")]


def main():
    n_c = len(COLS)
    x0, dx = 52.0, 32.0
    y0, dy = 152.0, 33.0          # rows must clear the caption strip at y < 22
    W = x0 + (n_c - 1) * dx + IMG / 2 + 12
    Hgt = y0 + 26
    fig, ax = plt.subplots(figsize=(W / 12.5, Hgt / 12.5))
    ax.set_xlim(0, W)
    ax.set_ylim(0, Hgt)
    ax.axis("off")

    ax.text(W / 2, y0 + 23, "The dual-cue block at every encoder scale",
            ha="center", va="top", fontsize=18, fontweight="bold",
            color="#a8701f", zorder=9)
    ax.text(W / 2, y0 + 16.5,
            "same LEVIR-CD tile, same trained model — nearest-neighbour "
            "rendering, so each panel shows its true resolution",
            ha="center", va="top", fontsize=12.5, style="italic",
            fontweight="bold", color="#5b6472", zorder=9)

    # ------------------------------------------------------------- headers
    for j, (_, cap, col, tcol) in enumerate(COLS):
        cx = x0 + j * dx
        ax.text(cx, y0 + 9.0, cap, ha="center", va="center", fontsize=15,
                fontweight="bold", color=tcol, zorder=9)

    # ---------------------------------------------------------------- grid
    for i, (num, stride, res) in enumerate(ROWS):
        cy = y0 - IMG / 2 - i * dy
        # row label
        ax.text(x0 - IMG / 2 - 7.0, cy + 4.0, "$E_%s$" % num, ha="right",
                va="center", fontsize=16, fontweight="bold", color="#2c4d8f",
                zorder=9)
        ax.text(x0 - IMG / 2 - 7.0, cy - 2.0, "stride %s" % stride, ha="right",
                va="center", fontsize=12.5, fontweight="bold", color="#5b6472",
                zorder=9)
        ax.text(x0 - IMG / 2 - 7.0, cy - 8.0, res, ha="right", va="center",
                fontsize=11.5, style="italic", fontweight="bold",
                color="#9aa0a6", zorder=9)

        for j, (tag, _, col, _) in enumerate(COLS):
            cx = x0 + j * dx
            p = os.path.join(A, "scale%d_%s.png" % (i + 1, tag))
            if not photo(ax, p, cx, cy, ZOOM, ec=col, lw=2.4):
                slab(ax, cx, cy, IMG, IMG, 0.1, "#dddddd")


    # ------------------------------------------------------------- caption
    ax.add_patch(FancyBboxPatch((14, 2.5), W - 28, 17.0,
                                boxstyle="round,pad=0.7,rounding_size=2",
                                fc="#f7f9fc", ec="#9aa8bd", lw=1.5, zorder=1))
    ax.text(W / 2, 11.0,
            "The change signal survives all four scales.\n"
            "At 8 x 8, where one activation covers a 32 x 32 patch of the input,\n"
            "the difference cue still marks the replaced buildings and the "
            "concatenation cue still carries the context it discards.",
            ha="center", va="center", fontsize=13, fontweight="bold",
            linespacing=1.55, color="#33465e", zorder=9)

    fig.tight_layout(pad=0.3)
    out = os.path.join(FIG, "fig_dualcue_scales.png")
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main()
