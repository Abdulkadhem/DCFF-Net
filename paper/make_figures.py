"""
Paper figures for DCFF-Net (schematics + result bars). Publication-sized fonts.
All numbers are the ADOPTED final results (see paper/tables.md):
  LEVIR 90.69 (MS-TTA) · SYSU 83.57/82.21 · EGY 81.39 · CLCD 77.25 · DSIFN 67.61
    python paper/make_figures.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

ENC = "#cfe0f0"; FUSE = "#f6c46b"; CTX = "#e0a0c0"; DEC = "#c9c9f0"; IN = "#eeeeee"
CONF = "#f4a6a6"   # conformal-FDR reliability layer (contribution 2)
OURS = "#2166ac"; BASE = "#9aa4ad"


def box(ax, x, y, w, h, t, fc, fs=12, weight="normal", ec="#555"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fc, ec=ec, lw=1.3, zorder=2))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, weight=weight, zorder=3)


def arrow(ax, x1, y1, x2, y2, color="#333", style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14,
                                 lw=1.5, color=color, zorder=1))


def fig_architecture():
    fig, ax = plt.subplots(figsize=(14, 7)); ax.set_xlim(0, 13); ax.set_ylim(0, 7); ax.axis("off")
    box(ax, 0.2, 4.7, 1.35, 1.0, "T1\n(before)", IN, 12)
    box(ax, 0.2, 1.3, 1.35, 1.0, "T2\n(after)", IN, 12)
    box(ax, 1.9, 3.1, 2.1, 2.5, "Siamese\nResNet-18 encoder\n(ImageNet-pretrained,\nshared weights)", ENC, 11.5, "bold")
    ax.text(2.95, 2.85, "4 scales", ha="center", fontsize=11, style="italic")
    arrow(ax, 1.55, 5.2, 1.9, 4.7); arrow(ax, 1.55, 1.8, 1.9, 3.5)

    ys = [5.2, 4.05, 2.9, 1.75]; labels = ["1/4", "1/8", "1/16", "1/32"]
    for i, y in enumerate(ys):
        box(ax, 4.3, y, 2.5, 0.95, f"Dual-Cue Fusion (ours)\n|F1−F2| + [F1,F2]  @{labels[i]}", FUSE, 11, "bold")
        arrow(ax, 4.0, 4.35, 4.3, y + 0.47)
    box(ax, 7.25, 1.75, 1.5, 0.95, "ASPP\n(CNN context)", CTX, 11.5, "bold")
    arrow(ax, 6.8, 2.22, 7.25, 2.22)
    box(ax, 9.2, 2.4, 1.85, 2.6, "U-Net decoder\n+ deep\nsupervision", DEC, 11.5, "bold")
    for y in ys[:3]:
        arrow(ax, 6.8, y + 0.47, 9.2, min(4.6, max(2.7, y + 0.2)), color="#8aa")
    arrow(ax, 8.75, 2.22, 9.2, 2.9, color="#a5a")
    box(ax, 11.35, 3.6, 1.25, 1.15, "change\nprobability\nmap", "#f9e79f", 11.5, "bold")
    arrow(ax, 11.05, 3.7, 11.35, 4.15)
    box(ax, 11.1, 1.4, 1.8, 1.4, "Conformal-FDR\nlayer (ours)\n$\\mathbb{E}[\\mathrm{FDP}]\\leq\\alpha$", CONF, 11.5, "bold")
    arrow(ax, 11.98, 3.6, 12.0, 2.8)
    ax.legend(handles=[Patch(fc=ENC, label="encoder (pretrained)"), Patch(fc=FUSE, label="dual-cue fusion (ours)"),
                       Patch(fc=CTX, label="ASPP context"), Patch(fc=DEC, label="decoder"),
                       Patch(fc=CONF, label="conformal-FDR (ours)")],
              loc="upper center", ncol=5, fontsize=11.5, framealpha=0.95, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    p = os.path.join(OUT, "fig_architecture.png"); fig.savefig(p, dpi=180, bbox_inches="tight"); print("wrote", p)


def fig_dualcue():
    fig, ax = plt.subplots(figsize=(9.5, 4.4)); ax.set_xlim(0, 8.8); ax.set_ylim(0, 4); ax.axis("off")
    box(ax, 0.15, 2.6, 1.35, 0.9, "$F^{(1)}_i$", ENC, 14)
    box(ax, 0.15, 0.5, 1.35, 0.9, "$F^{(2)}_i$", ENC, 14)
    box(ax, 2.3, 2.6, 2.1, 0.9, "difference cue\n$|F^{(1)}_i - F^{(2)}_i|$", FUSE, 11.5, "bold")
    box(ax, 2.3, 0.5, 2.1, 0.9, "concatenation cue\n$W_c[F^{(1)}_i ; F^{(2)}_i]$", FUSE, 11.5, "bold")
    arrow(ax, 1.5, 3.05, 2.3, 3.05); arrow(ax, 1.5, 0.95, 2.3, 0.95)
    arrow(ax, 1.5, 2.9, 2.3, 1.2); arrow(ax, 1.5, 0.7, 2.3, 2.9)
    box(ax, 5.2, 1.45, 1.9, 1.05, "ConvBlock $\\phi$\n(fuse)", "#b6d7a8", 12, "bold")
    arrow(ax, 4.4, 3.05, 5.2, 2.25); arrow(ax, 4.4, 0.95, 5.2, 1.75)
    box(ax, 7.5, 1.45, 1.2, 1.05, "fused$_i$", "#f9e79f", 12, "bold")
    arrow(ax, 7.1, 1.97, 7.5, 1.97)
    ax.set_title("Dual-Cue Fusion: change magnitude (difference) + relational context (concatenation)",
                 fontsize=12.5)
    fig.tight_layout()
    p = os.path.join(OUT, "fig_dualcue.png"); fig.savefig(p, dpi=180, bbox_inches="tight"); print("wrote", p)


def fig_comparison():
    """LEVIR-CD F1 vs published methods (ours = adopted 90.69)."""
    methods = ["FC-Siam-Diff\n(our repro)", "BIT", "ChangeFormer", "DCFF-Net\n(ours)+MS-TTA", "ChangeTitans"]
    f1 = [85.31, 89.30, 90.40, 90.69, 91.52]
    colors = [OURS if "ours" in m else BASE for m in methods]
    fig, ax = plt.subplots(figsize=(9.5, 5))
    ax.bar(range(len(methods)), f1, color=colors, edgecolor="k", linewidth=0.6)
    ax.set_xticks(range(len(methods))); ax.set_xticklabels(methods, fontsize=12)
    ax.set_ylabel("test F1 (%)", fontsize=13); ax.set_ylim(82, 93)
    ax.set_title("LEVIR-CD: DCFF-Net (13.6M) vs published methods", fontsize=13.5)
    ax.grid(alpha=0.3, axis="y"); ax.tick_params(labelsize=11.5)
    for i, v in enumerate(f1):
        ax.annotate(f"{v:.2f}", (i, v), textcoords="offset points", xytext=(0, 4), ha="center",
                    fontsize=12, weight="bold" if i == 3 else "normal")
    fig.tight_layout()
    p = os.path.join(OUT, "fig_comparison.png"); fig.savefig(p, dpi=180, bbox_inches="tight"); print("wrote", p)


def fig_progress():
    """Our iteration progress on LEVIR (adopted end point 90.69)."""
    iters = ["iter 1\npos_w=10", "iter 2\npos_w=2", "final\n(base)", "final\n+MS-TTA"]
    f1 = [87.52, 89.40, 90.02, 90.69]
    xs = list(range(len(iters)))
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.plot(xs, f1, "o-", lw=2.5, color=OURS, markersize=11)
    ax.set_xticks(xs); ax.set_xticklabels(iters, fontsize=12)
    ax.set_ylabel("test F1 (%)", fontsize=13); ax.set_title("DCFF-Net development on LEVIR-CD", fontsize=13.5)
    ax.grid(alpha=0.3); ax.tick_params(labelsize=11.5)
    for x, v in zip(xs, f1):
        ax.annotate(f"{v:.2f}", (x, v), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=12)
    fig.tight_layout()
    p = os.path.join(OUT, "fig_progress.png"); fig.savefig(p, dpi=180, bbox_inches="tight"); print("wrote", p)


def fig_multidataset():
    """DCFF-Net vs FC-Siam-Diff in-domain test F1 across five datasets (adopted numbers)."""
    datasets = ["LEVIR-CD", "SYSU-CD", "EGY-BCD", "CLCD", "DSIFN-CD"]
    dcff = [90.69, 83.57, 81.39, 77.25, 67.61]
    fcsiam = [85.31, 79.24, 68.10, 50.46, 57.31]
    x = np.arange(len(datasets)); w = 0.36
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    b1 = ax.bar(x - w / 2, fcsiam, w, label="FC-Siam-Diff (baseline)", color=BASE, edgecolor="k", linewidth=0.6)
    b2 = ax.bar(x + w / 2, dcff, w, label="DCFF-Net (ours)", color=OURS, edgecolor="k", linewidth=0.6)
    for bars in (b1, b2):
        for r in bars:
            ax.annotate(f"{r.get_height():.1f}", (r.get_x() + r.get_width() / 2, r.get_height()),
                        textcoords="offset points", xytext=(0, 4), ha="center", fontsize=11.5)
    for i, (d, f) in enumerate(zip(dcff, fcsiam)):
        ax.annotate(f"+{d - f:.1f}", (i, max(d, f) + 6), ha="center", fontsize=13, color=OURS, weight="bold")
    ax.set_xticks(x); ax.set_xticklabels(datasets, fontsize=12.5)
    ax.set_ylabel("test F1 (%)", fontsize=13); ax.set_ylim(0, 102)
    ax.set_title("In-domain change detection: DCFF-Net beats the baseline on all five datasets", fontsize=13.5)
    ax.legend(fontsize=12, loc="upper right"); ax.grid(alpha=0.3, axis="y"); ax.tick_params(labelsize=11.5)
    fig.tight_layout()
    p = os.path.join(OUT, "fig_multidataset.png"); fig.savefig(p, dpi=180, bbox_inches="tight"); print("wrote", p)


def fig_optimisation():
    """Optimisation study: seven avenues probed on LEVIR; only MS-TTA improved."""
    names = ["Model\nsoup", "RS-SSL\npretrain", "Lovász\n(scratch)", "longer +\nstrong aug",
             "FixRes\n384", "Lovász\npolish", "baseline\nrecipe", "+ MS-TTA\n(adopted)"]
    f1 = [86.57, 89.14, 89.34, 89.80, 90.19, 90.00, 90.02, 90.69]
    cols = [OURS if n.startswith("+ MS") else ("#7f9fbf" if "baseline" in n else "#d9a5a5") for n in names]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(range(len(names)), f1, color=cols, edgecolor="k", linewidth=0.6)
    ax.axhline(90.02, color="#444", ls="--", lw=1.3)
    ax.text(0.05, 90.12, "baseline recipe (90.02)", fontsize=11, color="#444")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("test F1 (%)", fontsize=13); ax.set_ylim(84, 92)
    ax.set_title("Optimisation study on LEVIR-CD: 7 avenues probed, only MS-TTA improved", fontsize=13.5)
    ax.grid(alpha=0.3, axis="y"); ax.tick_params(labelsize=11.5)
    for i, v in enumerate(f1):
        ax.annotate(f"{v:.2f}", (i, v), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=11.5,
                    weight="bold" if i == len(f1) - 1 else "normal")
    fig.tight_layout()
    p = os.path.join(OUT, "fig_optimisation.png"); fig.savefig(p, dpi=180, bbox_inches="tight"); print("wrote", p)


if __name__ == "__main__":
    # Figures 1 and 2 are drawn by make_architecture_v2.py and
    # make_dualcue_v2.py. The fig_architecture()/fig_dualcue() functions above
    # are superseded and deliberately not called, so that running this script
    # cannot overwrite the current versions.
    fig_comparison(); fig_progress()
    fig_multidataset(); fig_optimisation()
