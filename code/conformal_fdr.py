"""
Image-level conformal risk control for FDR on DCFF-Net change maps.
Calibrate a probability threshold on LEVIR-CD val so that the EXPECTED per-image
false-discovery proportion (FDP = FP/(TP+FP)) on unseen data is <= alpha, using the
conformal-risk-control bound (Angelopoulos et al. 2023):

    lambda_hat = inf { lambda : (n * Rhat(lambda) + B) / (n + 1) <= alpha },  B = 1

with per-image loss L_i(lambda) = FDP_i(lambda) (0 if no pixels predicted positive),
which is monotone non-increasing in the threshold lambda. Guarantee: E[L_test] <= alpha.

Produces Table 4 numbers + fig_conformal.png. Pure local, no GPU.
    python conformal_fdr.py
"""
import os, numpy as np

CDIR = "../results/conformal"


def load(split):
    d = np.load(f"{CDIR}/counts_{split}.npz")
    return d["tp"].astype(np.float64), d["fp"].astype(np.float64), d["fn"].astype(np.float64), d["grid"]


def per_image_fdp(tp, fp):
    denom = tp + fp
    fdp = np.where(denom > 0, fp / np.maximum(denom, 1), 0.0)
    return fdp  # [N, G]


def per_image_recall(tp, fn):
    denom = tp + fn
    # undefined when no true positives; mark nan to exclude from power mean
    rec = np.where(denom > 0, tp / np.maximum(denom, 1), np.nan)
    return rec  # [N, G]


def crc_threshold(fdp_cal, grid, alpha):
    """Smallest lambda (max power) whose CRC-corrected empirical risk <= alpha.
    Loss decreasing in lambda => valid region is grid indices >= g*. Returns g*, lambda."""
    n = fdp_cal.shape[0]
    Rhat = fdp_cal.mean(0)                      # [G] empirical per-image FDP, decreasing in lambda
    corrected = (n * Rhat + 1.0) / (n + 1.0)
    valid = np.where(corrected <= alpha)[0]
    if len(valid) == 0:
        return None, None, Rhat
    g = int(valid.min())                        # smallest lambda that controls FDR
    return g, float(grid[g]), Rhat


def main():
    tpc, fpc, fnc, grid = load("val")
    tpt, fpt, fnt, gridt = load("test")
    assert np.allclose(grid, gridt)
    fdp_cal = per_image_fdp(tpc, fpc)
    fdp_test = per_image_fdp(tpt, fpt)
    rec_test = per_image_recall(tpt, fnt)

    # reference: fixed threshold 0.5 on test
    g05 = int(np.argmin(np.abs(grid - 0.5)))
    ref_fdr = float(np.nanmean(fdp_test[:, g05])); ref_pow = float(np.nanmean(rec_test[:, g05]))
    print(f"[REF] fixed thr=0.50 -> test mean-FDP={ref_fdr:.4f}  power(recall)={ref_pow:.4f}\n")

    rows = []
    for alpha in [0.05, 0.10, 0.20]:
        g, lam, Rhat = crc_threshold(fdp_cal, grid, alpha)
        if g is None:
            print(f"[alpha={alpha}] no threshold controls FDR at this level"); continue
        test_fdr = float(np.nanmean(fdp_test[:, g]))
        test_pow = float(np.nanmean(rec_test[:, g]))
        cal_fdr = float(fdp_cal[:, g].mean())
        rows.append((alpha, lam, cal_fdr, test_fdr, test_pow))
        ok = "OK" if test_fdr <= alpha + 1e-9 else "VIOLATED"
        print(f"[alpha={alpha:.2f}] lambda={lam:.3f} | calib FDP={cal_fdr:.4f} "
              f"| TEST FDP={test_fdr:.4f} ({ok}) | power(recall)={test_pow:.4f}")

    # ---- figure: guarantee (realized FDR vs alpha) + power price ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        al = [r[0] for r in rows]; tfdr = [r[3] for r in rows]; tpow = [r[4] for r in rows]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5)) 
        a1.plot([0, 0.22], [0, 0.22], "--", color="#888", lw=1, label="target (y=x)")
        a1.scatter(al, tfdr, color="#2166ac", s=70, zorder=3, label="realized test FDR")
        for x, y in zip(al, tfdr):
            a1.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(6, -3), fontsize=11)
        a1.scatter([0.5 if False else None], [None])  # noop keep axes
        a1.set_xlabel("target FDR level  α", fontsize=13); a1.set_ylabel("realized test FDR", fontsize=13)
        a1.set_xlim(0, 0.22); a1.set_ylim(0, 0.22); a1.grid(alpha=0.3)
        a1.set_title("Conformal FDR guarantee holds (points ≤ y=x)", fontsize=13); a1.legend(fontsize=11); a1.tick_params(labelsize=11.5)

        a2.plot(al, [p * 100 for p in tpow], "o-", color="#5aa469", lw=2, label="conformal power")
        a2.axhline(ref_pow * 100, ls=":", color="#d9534f", lw=1.2,
                   label=f"fixed thr=0.5 (FDR={ref_fdr:.2f})")
        a2.set_xlabel("target FDR level  α", fontsize=13); a2.set_ylabel("power = recall on change (%)", fontsize=13)
        a2.grid(alpha=0.3); a2.set_title("Power vs FDR budget", fontsize=13); a2.legend(fontsize=11, loc="lower right"); a2.tick_params(labelsize=11.5)
        fig.tight_layout()
        out = "../paper/figures/fig_conformal.png"; fig.savefig(out, dpi=170, bbox_inches="tight")
        print("\nwrote", out)
    except Exception as e:
        print("fig skipped:", e)

    # ---- emit markdown table (utf-8 file; ascii-safe console) ----
    lines = ["| alpha (target FDR) | threshold lambda | calib FDP | **test FDP** | power (recall) |",
             "|---|---|---|---|---|"]
    for alpha, lam, cfdr, tfdr, tpow in rows:
        lines.append(f"| {alpha:.2f} | {lam:.3f} | {cfdr:.4f} | **{tfdr:.4f}** | {tpow:.4f} |")
    lines.append(f"| fixed 0.50 (baseline) | 0.500 | - | {ref_fdr:.4f} | {ref_pow:.4f} |")
    with open(f"{CDIR}/table4.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n=== Table 4 (also saved to results/conformal/table4.md) ===")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
