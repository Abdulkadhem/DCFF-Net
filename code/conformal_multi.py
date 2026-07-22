"""
Conformal-FDR across datasets: calibrate on a dataset's val, verify the FDR guarantee
on its test, using that dataset's own in-domain DCFF-Net checkpoint. Shows the
distribution-free guarantee is NOT LEVIR-specific. CPU-only.

    python conformal_multi.py --dataset egy   --ckpt ../results/multidataset/egy_dcff_best.pt
    python conformal_multi.py --dataset dsifn --ckpt ../results/multidataset/dsifn_dcff_best.pt
Writes results/conformal/{dataset}_conformal.json (+ appends to a combined summary).
"""
import argparse, json, os, numpy as np, torch
from torch.utils.data import DataLoader
from data_cd import make_cd
from models import DCFFNet

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GRID = np.round(np.linspace(0.02, 0.98, 49), 4)


@torch.no_grad()
def counts(model, split_ds):
    """per-image TP/FP/FN over the threshold grid."""
    ld = DataLoader(split_ds, batch_size=1, num_workers=0)
    TP, FP, FN = [], [], []
    for k, (xa, xb, y) in enumerate(ld):
        prob = torch.sigmoid(model(xa.to(DEV), xb.to(DEV)))[0, 0].cpu().numpy().ravel()
        yy = y[0, 0].numpy().ravel().astype(bool)
        pos, neg, npos = prob[yy], prob[~yy], int(yy.sum())
        tp = np.array([(pos > l).sum() for l in GRID], np.float64)
        fp = np.array([(neg > l).sum() for l in GRID], np.float64)
        TP.append(tp); FP.append(fp); FN.append(npos - tp)
        if (k + 1) % 300 == 0:
            print(f"    {k+1}/{len(ld)}", flush=True)
    return np.array(TP), np.array(FP), np.array(FN)


def per_image_fdp(tp, fp):
    d = tp + fp
    return np.where(d > 0, fp / np.maximum(d, 1), 0.0)


def crc(fdp_cal, alpha):
    n = fdp_cal.shape[0]
    R = fdp_cal.mean(0)
    corrected = (n * R + 1.0) / (n + 1.0)
    valid = np.where(corrected <= alpha)[0]
    return int(valid.min()) if len(valid) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()
    outdir = "../results/conformal"; os.makedirs(outdir, exist_ok=True)

    model = DCFFNet(backbone="resnet18", pretrained=False, use_cbam=False, use_aspp=True).to(DEV).eval()
    model.load_state_dict(torch.load(args.ckpt, map_location=DEV))
    print(f"[LOG] {args.dataset} | {args.ckpt}", flush=True)

    print("  [val counts]", flush=True); tpc, fpc, fnc = counts(model, make_cd(args.dataset, "val", augment=False))
    print("  [test counts]", flush=True); tpt, fpt, fnt = counts(model, make_cd(args.dataset, "test", augment=False))
    fdp_cal = per_image_fdp(tpc, fpc); fdp_test = per_image_fdp(tpt, fpt)
    rec_test = np.where((tpt + fnt) > 0, tpt / np.maximum(tpt + fnt, 1), np.nan)

    # (A) calibrate on official val (reveals sensitivity to val->test shift)
    # (B) calibrate on a random half of TEST, evaluate on the other half (exchangeable
    #     by construction -> the *correct* conformal deployment; guarantee should hold)
    rng = np.random.default_rng(0)
    perm = rng.permutation(fdp_test.shape[0]); h = len(perm) // 2
    cal_i, ev_i = perm[:h], perm[h:]

    def eval_rows(fdp_calib, fdp_eval, rec_eval):
        rr = []
        for a in [0.05, 0.10, 0.20]:
            g = crc(fdp_calib, a)
            if g is None:
                rr.append(dict(alpha=a, lam=None)); continue
            rr.append(dict(alpha=a, lam=float(GRID[g]),
                           test_fdp=float(np.nanmean(fdp_eval[:, g])),
                           power=float(np.nanmean(rec_eval[:, g]))))
        return rr

    rows_val = eval_rows(fdp_cal, fdp_test, rec_test)                       # (A) official val
    rows_split = eval_rows(fdp_test[cal_i], fdp_test[ev_i], rec_test[ev_i])  # (B) within-test split

    out = dict(dataset=args.dataset, ckpt=args.ckpt, calib_val=rows_val, calib_testsplit=rows_split)
    json.dump(out, open(f"{outdir}/{args.dataset}_conformal.json", "w"), indent=2)
    for tag, rows in [("(A) calib=official-val", rows_val), ("(B) calib=within-test split [exchangeable]", rows_split)]:
        print(f"\n=== {args.dataset} conformal {tag} ===")
        for r in rows:
            if r.get("lam") is None:
                print(f"  alpha={r['alpha']}: no valid threshold"); continue
            ok = "OK" if r["test_fdp"] <= r["alpha"] + 1e-9 else "VIOLATED"
            print(f"  alpha={r['alpha']:.2f} lam={r['lam']:.3f} FDP={r['test_fdp']:.4f} ({ok}) power={r['power']:.4f}")
    print("[ALL_DONE]", flush=True)


if __name__ == "__main__":
    main()
