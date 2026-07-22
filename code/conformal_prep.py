"""
Prepare per-image, per-threshold detection counts for conformal FDR control.
Runs DCFF-Net locally (CPU) on LEVIR-CD calibration (val) and evaluation (test),
and for a grid of probability thresholds records per-image TP/FP/FN pixel counts.
Output: results/conformal/counts_{split}.npz  (arrays [N, G] of int64: tp, fp, fn; + grid)

No GPU needed. This is the raw material for image-level conformal risk control.
    python conformal_prep.py --ckpt ../weights/dcff_final_best.pt
"""
import argparse, glob, io, os, numpy as np, torch
import pandas as pd
from PIL import Image
from models import DCFFNet

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# dataset locations come from paths.py, never from a literal in this file


def parquet_for(split):
    # dataset uses 'validation' and 'test'; accept 'val' as alias
    names = {"val": ["validation", "val"], "validation": ["validation", "val"], "test": ["test"]}[split]
    for nm in names:
        hits = find_parquet("levir", nm, required=False)
        if hits:
            return hits[0]
    raise FileNotFoundError(split)


def img_from(cell):
    b = cell["bytes"] if isinstance(cell, dict) else cell
    return Image.open(io.BytesIO(b))


def norm(a):
    return torch.from_numpy(((a / 255.0 - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1).astype(np.float32))


@torch.no_grad()
def run_split(model, split, grid):
    df = pd.read_parquet(parquet_for(split)).reset_index(drop=True)
    N, G = len(df), len(grid)
    TP = np.zeros((N, G), np.int64); FP = np.zeros((N, G), np.int64); FN = np.zeros((N, G), np.int64)
    for i in range(N):
        row = df.iloc[i]
        a = np.array(img_from(row["imageA"]).convert("RGB"), np.uint8)
        b = np.array(img_from(row["imageB"]).convert("RGB"), np.uint8)
        y = (np.array(img_from(row["label"]).convert("L"), np.uint8) > 127)
        prob = torch.sigmoid(model(norm(a)[None].to(DEV), norm(b)[None].to(DEV)))[0, 0].cpu().numpy()
        yp = y.ravel(); pp = prob.ravel()
        pos = pp[yp]; neg = pp[~yp]           # probs at true-change / true-nochange pixels
        npos = yp.sum()
        for g, lam in enumerate(grid):
            tp = int((pos > lam).sum()); fp = int((neg > lam).sum())
            TP[i, g] = tp; FP[i, g] = fp; FN[i, g] = int(npos - tp)
        if (i + 1) % 256 == 0:
            print(f"  {split} {i+1}/{N}", flush=True)
    return TP, FP, FN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--backbone", default="resnet18")
    args = ap.parse_args()
    grid = np.round(np.linspace(0.02, 0.98, 49), 4)
    outdir = "../results/conformal"; os.makedirs(outdir, exist_ok=True)
    model = DCFFNet(backbone=args.backbone, pretrained=False, use_cbam=False, use_aspp=True).to(DEV).eval()
    model.load_state_dict(torch.load(args.ckpt, map_location=DEV))
    print(f"[LOG] loaded {args.ckpt} | grid={len(grid)} thresholds | device={DEV}", flush=True)
    for split in ["val", "test"]:
        TP, FP, FN = run_split(model, split, grid)
        p = f"{outdir}/counts_{split}.npz"
        np.savez_compressed(p, tp=TP, fp=FP, fn=FN, grid=grid)
        print(f"[SAVED] {p} | N={TP.shape[0]}", flush=True)
    print("[ALL_DONE]", flush=True)


if __name__ == "__main__":
    main()
