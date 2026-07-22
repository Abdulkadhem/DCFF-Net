"""
Training-curve figure for DCFF-Net final model, parsed from results/training_logs/final.log.
Left axis: training loss. Right axis: validation F1 / IoU. Marks the best-val epoch.
    python paper/make_training_curve.py
"""
import os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
LOG = os.path.join(HERE, "..", "results", "training_logs", "final.log")
OUT = os.path.join(HERE, "figures", "fig_training_curve.png")

pat = re.compile(r"\[E(\d+)/\d+\]\s+loss=([\d.]+)\s+val_F1=([\d.]+)\s+val_IoU=([\d.]+)")
ep, loss, f1, iou = [], [], [], []
best_ep = None
for line in open(LOG, encoding="utf-8", errors="ignore"):
    m = pat.search(line)
    if m:
        ep.append(int(m.group(1))); loss.append(float(m.group(2)))
        f1.append(float(m.group(3)) * 100); iou.append(float(m.group(4)) * 100)
    mb = re.search(r"best val@ep(\d+)", line)
    if mb:
        best_ep = int(mb.group(1))

fig, axL = plt.subplots(figsize=(9, 5.2))
axL.plot(ep, loss, color="#d9534f", lw=1.8, label="train loss")
axL.set_xlabel("epoch", fontsize=13); axL.set_ylabel("training loss", color="#d9534f", fontsize=13)
axL.tick_params(axis="y", labelcolor="#d9534f"); axL.set_xlim(1, max(ep))
axL.grid(alpha=0.25)

axR = axL.twinx()
axR.plot(ep, f1, color="#2166ac", lw=2.0, label="val F1")
axR.plot(ep, iou, color="#5aa469", lw=1.8, ls="--", label="val IoU")
axR.set_ylabel("validation metric (%)", color="#2166ac", fontsize=13)
axR.tick_params(axis="y", labelcolor="#2166ac")

if best_ep:
    axL.axvline(best_ep, color="#888", ls=":", lw=1.2)
    ymax = max(f1)
    axR.annotate(f"best val @ ep{best_ep}", xy=(best_ep, ymax),
                 xytext=(best_ep - 34, ymax - 6), fontsize=11, color="#444",
                 arrowprops=dict(arrowstyle="->", color="#888", lw=0.9))

lines = [l for l in (axL.get_lines() + axR.get_lines()) if not l.get_label().startswith("_")]
axL.legend(lines, [l.get_label() for l in lines], loc="center right", fontsize=11.5, framealpha=0.95)
axL.tick_params(labelsize=11.5); axR.tick_params(labelsize=11.5)
axL.set_title("DCFF-Net training on LEVIR-CD (final model, 100 epochs)", fontsize=13.5)
fig.tight_layout()
fig.savefig(OUT, dpi=170, bbox_inches="tight")
print("wrote", OUT, "| epochs parsed:", len(ep), "| best_ep:", best_ep,
      "| final val_F1:", f1[-1], "| peak val_F1:", max(f1))
