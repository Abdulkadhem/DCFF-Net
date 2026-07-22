"""
Ablation & baseline convergence figures for DCFF-Net, parsed from the rescued logs
(results/training_logs/*.log). No GPU needed. Produces:
  fig_ablation_curves.png   — val F1 vs epoch for the 7 ablation variants (+ final)
  fig_baseline_curves.png   — val F1 vs epoch: FC-Siam baseline vs DCFF-Net variants
Demonstrates design robustness visually (convergence), honestly, from our own runs.
    python paper/make_ablation_curves.py
"""
import os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
LOGDIR = os.path.join(HERE, "..", "results", "training_logs")
OUT = os.path.join(HERE, "figures")

EPAT = re.compile(r"\[E(\d+)/\d+\]\s+loss=([\d.]+)\s+val_F1=([\d.]+)\s+val_IoU=([\d.]+)")
TPAT = re.compile(r"=== TEST \(([^)]+)\) ===")


def parse_runs(path):
    """Split a log into named runs: accumulate epoch curves until a TEST line names them."""
    runs = {}
    cur = []
    for line in open(path, encoding="utf-8", errors="ignore"):
        m = EPAT.search(line)
        if m:
            cur.append((int(m.group(1)), float(m.group(3)) * 100, float(m.group(4)) * 100))
            continue
        t = TPAT.search(line)
        if t and cur:
            runs[t.group(1)] = cur
            cur = []
    return runs


def collect():
    runs = {}
    for fn in ["abl.log", "final.log", "sweep.log", "i3.log", "train.log"]:
        p = os.path.join(LOGDIR, fn)
        if os.path.exists(p):
            runs.update(parse_runs(p))
    return runs


def plot_curves(runs, keys_labels_styles, title, out, ymin=78):
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    for key, lab, color, lw, ls in keys_labels_styles:
        if key not in runs:
            print("  (missing run:", key, ")"); continue
        ep = [e for e, f, i in runs[key]]
        f1 = [f for e, f, i in runs[key]]
        ax.plot(ep, f1, label=lab, color=color, lw=lw, ls=ls)
    ax.set_xlabel("epoch", fontsize=13); ax.set_ylabel("validation F1 (%)", fontsize=13)
    ax.set_ylim(ymin, 91); ax.grid(alpha=0.3)
    ax.set_title(title, fontsize=13.5)
    ax.legend(fontsize=11, loc="lower right", framealpha=0.95, ncol=2)
    ax.tick_params(labelsize=11.5)
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print("wrote", out)


def main():
    runs = collect()
    print("parsed runs:", {k: len(v) for k, v in runs.items()})

    # Fig A: ablation variants (our design choices)
    abl = [
        ("dcff_final", "DCFF-Net (full, 100ep)", "#2166ac", 2.4, "-"),
        ("abl_dual",   "dual-cue (60ep ref)",    "#4a90d9", 1.8, "-"),
        ("abl_conc",   "conc-only",              "#8e6fb0", 1.6, "--"),
        ("abl_diff",   "diff-only",              "#d98c5f", 1.6, "--"),
        ("abl_noaspp", "− ASPP",                 "#5aa469", 1.4, ":"),
        ("abl_nocbam", "− CBAM",                 "#9aa0a6", 1.4, ":"),
        ("abl_nobnd",  "− boundary",             "#c0b060", 1.4, ":"),
        ("abl_scratch","no pretrain (scratch)",  "#d9534f", 2.2, "-"),
    ]
    plot_curves(runs, abl, "DCFF-Net ablation: validation convergence on LEVIR-CD",
                os.path.join(OUT, "fig_ablation_curves.png"), ymin=78)

    # Fig B: baseline vs ours
    base = [
        ("fcsiam",     "FC-Siam-Diff (baseline)", "#d9534f", 2.2, "-"),
        ("dcff_pw2",   "DCFF-Net (pw=2)",         "#4a90d9", 1.8, "--"),
        ("dcff_r34",   "DCFF-Net (ResNet-34)",    "#5aa469", 1.8, "--"),
        ("dcff_final", "DCFF-Net (full, 100ep)",  "#2166ac", 2.4, "-"),
    ]
    plot_curves(runs, base, "DCFF-Net vs FC-Siam baseline: validation convergence",
                os.path.join(OUT, "fig_baseline_curves.png"), ymin=80)


if __name__ == "__main__":
    main()
