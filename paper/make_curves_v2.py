"""
Two NEW curve figures built from the full multi-session training logs:
  fig_convergence_multidataset.png — DCFF-Net vs FC-Siam validation convergence on all
                                     five datasets (5 panels): our model dominates the
                                     baseline throughout training, everywhere.
  fig_optimisation_curves.png      — the LEVIR optimisation study as curves: every probed
                                     avenue plotted against the adopted recipe.
    python paper/make_curves_v2.py
"""
import os, re, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "results")
OUT = os.path.join(HERE, "figures")
EPAT = re.compile(r"\[E(\d+)/\d+\]\s+loss=([\d.]+)\s+val_F1=([\d.]+)\s+val_IoU=([\d.]+)")
TPAT = re.compile(r"=== TEST \(([^)]+)\) ===")
OURS, BASE = "#2166ac", "#9aa4ad"


def parse_single(path):
    """(epoch, valF1%) for a log holding one run."""
    out = []
    if not os.path.exists(path):
        return out
    for line in open(path, encoding="utf-8", errors="ignore"):
        m = EPAT.search(line)
        if m:
            out.append((int(m.group(1)), float(m.group(3)) * 100))
    return out


def parse_named(path):
    """{run_name: [(epoch, valF1%)]} for a log holding several runs (split at TEST lines)."""
    runs, cur = {}, []
    if not os.path.exists(path):
        return runs
    for line in open(path, encoding="utf-8", errors="ignore"):
        m = EPAT.search(line)
        if m:
            cur.append((int(m.group(1)), float(m.group(3)) * 100)); continue
        t = TPAT.search(line)
        if t and cur:
            runs[t.group(1)] = cur; cur = []
    return runs


def L(*parts):
    return os.path.join(RES, *parts)


def fig_convergence_multidataset():
    sweep = parse_named(L("training_logs", "sweep.log"))
    panels = [
        ("LEVIR-CD",  parse_single(L("training_logs", "final.log")),          sweep.get("fcsiam", []),                       "90.69 vs 85.31"),
        ("SYSU-CD",   parse_single(L("final", "logs", "sysu_dcff.log")),      parse_single(L("final", "logs", "sysu_fcsiam.log")),   "83.57 vs 79.24"),
        ("EGY-BCD",   parse_single(L("final", "logs", "egy_dcff_v2.log")),    parse_single(L("final", "logs", "egy_fcsiam.log")),    "81.39 vs 68.10"),
        ("CLCD",      parse_single(L("final", "logs", "clcd_dcff.log")),      parse_single(L("final", "logs", "clcd_fcsiam.log")),   "77.25 vs 50.46"),
        ("DSIFN-CD",  parse_single(L("final", "logs", "dsifn512_dcff_v2.log")), parse_single(L("final", "logs", "dsifn512_fcsiam.log")), "67.61 vs 57.31"),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.6), sharey=False)
    for ax, (name, ours, base, note) in zip(axes, panels):
        if ours:
            ax.plot([e for e, _ in ours], [f for _, f in ours], color=OURS, lw=2.2, label="DCFF-Net (ours)")
        if base:
            ax.plot([e for e, _ in base], [f for _, f in base], color=BASE, lw=2.0, ls="--", label="FC-Siam-Diff")
        ax.set_title(f"{name}\n(test F1 {note})", fontsize=13)
        ax.set_xlabel("epoch", fontsize=12.5); ax.grid(alpha=0.3); ax.tick_params(labelsize=11)
        ax.legend(fontsize=10.5, loc="lower right")
        if not ours and not base:
            ax.text(0.5, 0.5, "log missing", ha="center", transform=ax.transAxes)
    axes[0].set_ylabel("validation F1 (%)", fontsize=13)
    fig.suptitle("DCFF-Net dominates the baseline throughout training on every dataset", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = os.path.join(OUT, "fig_convergence_multidataset.png")
    fig.savefig(p, dpi=170, bbox_inches="tight"); print("wrote", p)


def fig_optimisation_curves():
    series = [
        ("adopted recipe (100 ep)", parse_single(L("training_logs", "final.log")),                    OURS,      "-",  2.6),
        ("longer + strong aug",     parse_single(L("final", "logs", "levir_dcff_long.log")),          "#e08a2b", "--", 1.7),
        ("RS-SSL pretraining",      parse_single(L("final", "logs", "levir_rspre.log")),              "#8e6fb0", "--", 1.7),
        ("Lovász (from scratch)",   parse_single(L("final", "logs", "levir_lovasz.log")),             "#5aa469", ":",  1.9),
        ("Lovász polish (low lr)",  parse_single(L("final", "logs", "levir_polish.log")),             "#d9534f", "-.", 1.9),
        ("FixRes 384 fine-tune",    parse_single(L("final", "logs", "levir_fixres384.log")),          "#7f7f7f", "-.", 1.7),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for lab, data, c, ls, lw in series:
        if not data:
            print("  (missing:", lab, ")"); continue
        ax.plot([e for e, _ in data], [f for _, f in data], label=lab, color=c, ls=ls, lw=lw)
    ax.axhline(90.42, color="#444", ls=":", lw=1.4)
    ax.text(2, 90.5, "best validation of the adopted recipe (90.42)", fontsize=11, color="#444")
    ax.set_xlabel("epoch", fontsize=13); ax.set_ylabel("validation F1 (%)", fontsize=13)
    ax.set_ylim(84, 91.4); ax.grid(alpha=0.3); ax.tick_params(labelsize=11.5)
    ax.set_title("Optimisation study on LEVIR-CD: no probed avenue surpasses the adopted recipe", fontsize=13.5)
    ax.legend(fontsize=11, loc="lower right", ncol=2, framealpha=0.95)
    fig.tight_layout()
    p = os.path.join(OUT, "fig_optimisation_curves.png")
    fig.savefig(p, dpi=170, bbox_inches="tight"); print("wrote", p)


if __name__ == "__main__":
    fig_convergence_multidataset()
    fig_optimisation_curves()
