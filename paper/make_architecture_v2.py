"""
Figure 1 -- the DCFF-Net framework, drawn as a block diagram.

One horizontal lane per encoder scale, so no two arrows cross. Feature tensors
are pseudo-3D slabs that shrink with the stride. Inputs and outputs are real
LEVIR-CD imagery: the probability map and the binary decision are this model's
actual prediction on the displayed tile (produced by extract_arch_assets.py).

    python paper/extract_arch_assets.py     # once, to make the imagery
    python paper/make_architecture_v2.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import *                                          # noqa: F401,F403

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
A = os.path.join(FIG, "_assets")

# ------------------------------------------------------------------- frame
BOT, TOP = 18.0, 124.0          # every column panel spans this band
H = TOP - BOT
Y_TITLE, Y_SUB = TOP - 2.0, TOP - 9.0
YC = [99.0, 76.0, 53.0, 30.0]   # one lane per encoder scale (23-unit pitch)
S = [1.00, 0.84, 0.69, 0.55]    # tensors shrink with the stride
BASE = 9.5                      # tensor edge at the finest scale
SC = ["1/4", "1/8", "1/16", "1/32"]


def main():
    fig, ax = plt.subplots(figsize=(24.6, 12.4))
    ax.set_xlim(0, 260)
    ax.set_ylim(0, 130)
    ax.axis("off")

    # ================================================================ INPUT
    panel(ax, 3, 24, 27, 100, BG_IN, "#9aa0a6", ls=(0, (6, 4)))
    label(ax, 16.5, Y_TITLE, "Bi-temporal input", 14.5, GREY, "bold", va="top")
    ok = photo(ax, os.path.join(A, "arch_t1.png"), 16.5, 97, 0.60)
    photo(ax, os.path.join(A, "arch_t2.png"), 16.5, 52, 0.60)
    if not ok:
        slab(ax, 16.5, 97, 17, 17, 2, "#c8d6e5")
        slab(ax, 16.5, 52, 17, 17, 2, "#c8d6e5")
    label(ax, 16.5, 83.5, "$T_1$   (before)", 14.5, INK, va="top")
    label(ax, 16.5, 38.5, "$T_2$   (after)", 14.5, INK, va="top")

    # ============================================================== ENCODER
    panel(ax, 34, BOT, 30, H, BG_ENC, C_T2, "Shared Encoder", 15,
          sub="ResNet-18 · ImageNet")
    for k, yc in enumerate(YC):
        w = h = BASE * S[k]
        stack(ax, 43.5, yc, w, h, 1.6 + 0.6 * k, C_T1, n=2, gap=0.9)
        stack(ax, 56.0, yc, w, h, 1.6 + 0.6 * k, C_T2, n=2, gap=0.9)
        label(ax, 49.7, yc - h / 2 - 3.4, "$E_%d$   stride %s" % (k + 1, SC[k]),
              12, "#2c4d8f", va="top")
    arrow(ax, (30.4, 97), (36.5, 100), rad=0.10, lw=1.8)
    arrow(ax, (30.4, 52), (36.5, 30), rad=-0.10, lw=1.8)

    # ==================================================== DUAL-CUE FUSION
    panel(ax, 68, BOT, 74, H, BG_FUSE, "#cf9838", "Dual-Cue Feature Fusion",
          16, tcol="#a8701f",
          sub="contribution 1")
    for k, yc in enumerate(YC):
        w = h = BASE * S[k]
        d = 1.6 + 0.6 * k
        off = h / 2.0 + 1.2
        yd, yl = yc + off, yc - off

        slab(ax, 84, yd, w, h, d, C_DIFF)                       # difference cue
        rbox(ax, 76, yl, 6.2, 5.2, C_WC, "#b8912a", "$W_c$", fs=10.5)
        slab(ax, 90, yl, w * 1.2, h, d, C_CONC)                 # concatenation cue
        slab(ax, 126, yc, w, h, d, C_FUSED)                     # fused
        label(ax, 126, yc - h / 2 - 1.4, "$\\widetilde{F}_%d$" % (k + 1),
              12, "#a83232", "bold", va="top")

        arrow(ax, (63.5, yc + 1.4), (84 - w / 2 - 1.2, yd - h * 0.2),
              color="#a4560f", lw=1.5)
        arrow(ax, (63.5, yc - 1.4), (72.6, yl + 0.5), color="#5f3a96", lw=1.5)
        arrow(ax, (79.4, yl), (90 - w * 0.6 - 1.2, yl), color="#5f3a96",
              lw=1.4, ms=11)
        arrow(ax, (84 + w / 2 + d, yd - h * 0.2), (126 - w / 2 - 1.2, yc + h * 0.3),
              color="#a4560f", lw=1.6)
        arrow(ax, (90 + w * 0.6 + d, yl + h * 0.2), (126 - w / 2 - 1.2, yc - h * 0.3),
              color="#5f3a96", lw=1.6)

    # ================================================================= ASPP
    panel(ax, 146, BOT, 22, 30, BG_ASPP, C_ASPP)
    slab(ax, 157, 31, 13, 11, 3.2, C_ASPP)
    ax.text(157, 31, "ASPP", ha="center", va="center", fontsize=12.5,
            fontweight="bold", color="white", zorder=8)
    label(ax, 157, 42.5, "context module", 12.5, "#2f7a4a", va="center")
    label(ax, 157, 21.5, "dilations 1, 2, 4", 12, "#2f7a4a", va="center")
    arrow(ax, (133, 30), (151.5, 31), color=C_ASPP, lw=2.2)
    label(ax, 141, 35.5, "deepest scale only", 11.5, "#2f7a4a", style="italic",
          va="bottom")

    # ============================================================== DECODER
    panel(ax, 172, BOT, 34, H, BG_DEC, C_DEC, "U-Net Decoder", 15,
          sub="deep supervision")
    for k, yc in enumerate(YC):
        w = 8.5 + 2.2 * (3 - k)
        slab(ax, 181, yc, w, BASE * S[k] * 1.15, 2.6, C_DEC)
        if k < 3:
            arrow(ax, (133, yc), (181 - w / 2 - 1.6, yc), color="#7060bb",
                  lw=1.5, ls=(0, (6, 3)), ms=12)
        else:
            arrow(ax, (164.5, 31), (181 - w / 2 - 1.6, yc), color=C_ASPP, lw=2.0)
        if k > 0:
            arrow(ax, (176, YC[k] + BASE * S[k] * 0.62 + 1.0),
                  (176, YC[k - 1] - BASE * S[k - 1] * 0.62 - 0.4),
                  color=C_DEC, lw=2.0, ms=12)
    label(ax, 154, 108.0, "skip connections", 12.5, "#7060bb", style="italic",
          va="center")
    for k in (1, 2, 3):
        rbox(ax, 199, YC[k], 11.0, 6.2, "#ddd7f7", C_DEC,
             "$P^{\\mathrm{aux}}_%d$" % k, fs=11)
        arrow(ax, (181 + (8.5 + 2.2 * (3 - k)) / 2 + 2.8, YC[k]), (193.2, YC[k]),
              color=C_DEC, lw=1.4, ms=11)

    # ================================================================ OUTPUT
    label(ax, 236, Y_TITLE, "Model output", 15, "#a8801c", "bold", va="top")
    got = photo(ax, os.path.join(A, "arch_prob.png"), 224, 101, 0.44,
                ec="#c39a2e", lw=1.8)
    photo(ax, os.path.join(A, "arch_pred.png"), 248, 101, 0.44, ec="#c39a2e", lw=1.8)
    if not got:
        slab(ax, 224, 101, 17, 17, 2, C_OUT); slab(ax, 248, 101, 17, 17, 2, C_OUT)
    label(ax, 224, 89.5, "probability map $\\hat{p}$", 11.5, INK, va="top")
    label(ax, 248, 89.5, "binary decision", 13, INK, va="top")
    arrow(ax, (192, 99), (215, 101), lw=2.0, rad=0.06)

    # ====================================================== CONFORMAL LAYER
    panel(ax, 214, 44, 42, 32, BG_CONF, C_CONF)
    label(ax, 235, 73.5, "Conformal-FDR layer", 14.5, "#c0324f", "bold", va="top")
    label(ax, 235, 68.8, "contribution 2", 11.5, "#c0324f", style="italic", va="top")
    label(ax, 235, 63.5, "post-hoc  ·  model-agnostic  ·  training-free",
          11.8, "#5b6472", va="top")
    ax.text(235, 54.5, "calibrate one threshold $\\hat{\\lambda}$ such that\n"
                       "$\\mathbb{E}[\\mathrm{FDP}_{\\mathrm{test}}]\\leq\\alpha$",
            ha="center", va="center", fontsize=13.5, fontweight="bold", zorder=8)
    arrow(ax, (235, 90.5), (235, 77.0), color=C_CONF, lw=2.2)

    rbox(ax, 235, 34, 42, 11, "#ffffff", "#9aa0a6",
         "certified change map    or    ABSTAIN", fs=12, fw="normal")
    arrow(ax, (235, 43), (235, 40.0), color=C_CONF, lw=2.0)

    # =============================================================== LEGEND
    ax.add_patch(FancyBboxPatch((4, 1.5), 252, 14,
                                boxstyle="round,pad=0.7,rounding_size=2",
                                fc="#fcfcfc", ec="#c9c9c9", lw=1.3, zorder=1))
    legend_row(ax, [(C_T1, "features from $T_1$"),
                    (C_T2, "features from $T_2$"),
                    (C_DIFF, "difference cue  $|F^{(1)}_i - F^{(2)}_i|$"),
                    (C_CONC, "concatenation cue  $[F^{(1)}_i ; F^{(2)}_i]$")],
               x0=12, y=12.6, dx=63)
    legend_row(ax, [(C_FUSED, "fused feature  $\\widetilde{F}_i$"),
                    (C_WC, "$W_c$   $1\\!\\times\\!1$ convolution"),
                    (C_ASPP, "ASPP context module"),
                    (C_DEC, "decoder / deep supervision")],
               x0=12, y=5.4, dx=63)

    fig.tight_layout(pad=0.3)
    out = os.path.join(FIG, "fig_architecture.png")
    fig.savefig(out, dpi=165, bbox_inches="tight", facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main()
