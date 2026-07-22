"""
Local (CPU) evaluation of DCFF-Net on any bitemporal change-detection parquet.
Reads imageA/imageB/label columns directly from a local HF parquet cache
(no Hub contact). Runs plain + TTA evaluation, reports P/R/F1/IoU on the change class.

Default target = local LEVIR-CD test (sanity / reproduce). Point --parquet elsewhere
for cross-dataset generalization (e.g. WHU-CD) once that parquet is available locally.

    python eval_local.py --ckpt ../weights/dcff_final_best.pt
    python eval_local.py --ckpt ../weights/dcff_final_best.pt --parquet <whu_test.parquet> --tta
"""
import argparse, glob, io, numpy as np, torch
import pandas as pd
from models import DCFFNet
from paths import find_parquet

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# dataset locations come from paths.py, never from a literal in this file


def _img(cell):
    b = cell["bytes"] if isinstance(cell, dict) else cell
    return np.array(Image_open(b).convert("RGB"), np.uint8)


def Image_open(b):
    from PIL import Image
    return Image.open(io.BytesIO(b))


def norm(a):
    return torch.from_numpy(((a / 255.0 - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1).astype(np.float32))


def load_df(parquet):
    if parquet:
        path = parquet
    else:
        path = find_parquet("levir", "test")[0]
    print("[LOG] parquet:", path, flush=True)
    return pd.read_parquet(path).reset_index(drop=True)


def tta_views(x):
    # (transform, inverse) pairs on NCHW
    return [
        (lambda t: t, lambda t: t),
        (lambda t: torch.flip(t, [3]), lambda t: torch.flip(t, [3])),
        (lambda t: torch.flip(t, [2]), lambda t: torch.flip(t, [2])),
        (lambda t: torch.rot90(t, 1, [2, 3]), lambda t: torch.rot90(t, -1, [2, 3])),
        (lambda t: torch.rot90(t, 2, [2, 3]), lambda t: torch.rot90(t, -2, [2, 3])),
        (lambda t: torch.rot90(t, 3, [2, 3]), lambda t: torch.rot90(t, -3, [2, 3])),
    ]


@torch.no_grad()
def evaluate(model, df, use_tta):
    tp = fp = fn = 0.0
    tfs = tta_views(None) if use_tta else [(lambda t: t, lambda t: t)]
    for i in range(len(df)):
        row = df.iloc[i]
        a = _img(row["imageA"]); b = _img(row["imageB"])
        lab = row["label"]["bytes"] if isinstance(row["label"], dict) else row["label"]
        y = (np.array(Image_open(lab).convert("L"), np.uint8) > 127).astype(np.float32)
        xa = norm(a)[None].to(DEV); xb = norm(b)[None].to(DEV)
        prob = torch.zeros(1, 1, y.shape[0], y.shape[1], device=DEV)
        for fwd, inv in tfs:
            prob += inv(torch.sigmoid(model(fwd(xa), fwd(xb))))
        prob /= len(tfs)
        pred = (prob[0, 0].cpu().numpy() > 0.5).astype(np.float32)
        tp += float(((pred == 1) & (y == 1)).sum())
        fp += float(((pred == 1) & (y == 0)).sum())
        fn += float(((pred == 0) & (y == 1)).sum())
        if (i + 1) % 256 == 0:
            print(f"  ..{i+1}/{len(df)}", flush=True)
    prec = tp / (tp + fp + 1e-9); rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9); iou = tp / (tp + fp + fn + 1e-9)
    return prec, rec, f1, iou


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--parquet", default=None)
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--tta", action="store_true")
    args = ap.parse_args()

    df = load_df(args.parquet)
    model = DCFFNet(backbone=args.backbone, pretrained=False, use_cbam=False, use_aspp=True).to(DEV).eval()
    model.load_state_dict(torch.load(args.ckpt, map_location=DEV))
    print(f"[LOG] loaded {args.ckpt} | N={len(df)} | TTA={args.tta} | device={DEV}", flush=True)
    p, r, f1, iou = evaluate(model, df, args.tta)
    print(f"[RESULT] P={p:.4f} R={r:.4f} F1={f1:.4f} IoU={iou:.4f}", flush=True)


if __name__ == "__main__":
    main()
