"""
Conformal FDR across three datasets — two panels:
 (left)  SOUND: with exchangeable calibration the certificate is valid — realized
         FDR <= target (LEVIR, EGY hold); on DSIFN the layer correctly ABSTAINS
         (FDR<=0.2 infeasible: model precision too low) rather than over-promising.
 (right) DIAGNOSTIC: calibrating on a distribution-shifted split (official val)
         yields false certificates (realized FDR > target), proportional to the shift.
    python paper/make_conformal_multi.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "figures")

# (B) exchangeable (proper) calibration — realized test FDR (None = abstains/infeasible)
SOUND = {
    "LEVIR-CD": (dict(zip([0.05, 0.10, 0.20], [0.045, 0.097, 0.125])), "#3a8c4f", "o"),
    "EGY_BCD":  (dict(zip([0.10, 0.20], [0.081, 0.166])),               "#2166ac", "s"),
    # DSIFN abstains at all levels -> plotted as annotation, no points
}
# (A) shifted (official-val) calibration — false certificates
SHIFT = {
    "EGY_BCD (val calib)":  (dict(zip([0.10, 0.20], [0.102, 0.178])),            "#e08a2b", "s"),
    "DSIFN-CD (val calib)": (dict(zip([0.05, 0.10, 0.20], [0.355, 0.427, 0.487])), "#d9534f", "^"),
}

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.8))

for ax, title in [(axL, "Sound: exchangeable calibration →\ncertificate valid or abstain (never false)"),
                  (axR, "Diagnostic: shifted calibration →\nfalse certificates track the shift")]:
    ax.plot([0, 0.22], [0, 0.22], "--", color="#888", lw=1.2, label="target (y = x)")
    ax.fill_between([0, 0.22], [0, 0.22], 0, color="#3a8c4f", alpha=0.06)
    ax.set_xlabel("target FDR level  α", fontsize=13); ax.set_xlim(0, 0.22); ax.grid(alpha=0.3); ax.tick_params(labelsize=11.5)
    ax.set_title(title, fontsize=12.5)

axL.set_ylabel("realized test FDR", fontsize=13); axL.set_ylim(0, 0.22)
for name, (pts, c, mk) in SOUND.items():
    xs = sorted(pts); ys = [pts[a] for a in xs]
    axL.plot(xs, ys, "-", color=c, marker=mk, ms=9, lw=1.9, label=name + " ✓ holds")
    for a, y in pts.items():
        axL.annotate(f"{y:.3f}", (a, y), textcoords="offset points", xytext=(6, -3), fontsize=10, color=c)
axL.annotate("DSIFN-CD: layer ABSTAINS\n(FDR≤0.2 infeasible — honest\nrefusal, not a false certificate)",
             (0.055, 0.17), fontsize=11, color="#d9534f",
             bbox=dict(boxstyle="round,pad=0.3", fc="#fdeaea", ec="#d9534f", lw=0.8))
axL.legend(fontsize=11, loc="lower right")

axR.set_ylim(0, 0.52)
for name, (pts, c, mk) in SHIFT.items():
    xs = sorted(pts); ys = [pts[a] for a in xs]
    axR.plot(xs, ys, "--", color=c, marker=mk, ms=9, lw=1.8, label=name)
    for a, y in pts.items():
        axR.annotate(f"{y:.3f}", (a, y), textcoords="offset points", xytext=(6, -3), fontsize=10, color=c)
axR.text(0.15, 0.03, "target region\n(y = x)", fontsize=8, color="#3a8c4f")
axR.legend(fontsize=11, loc="upper left")

fig.suptitle("Conformal-FDR: trustworthy under exchangeability, and a distribution-shift diagnostic", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])
p = os.path.join(OUT, "fig_conformal_multi.png"); fig.savefig(p, dpi=170, bbox_inches="tight")
print("wrote", p)
