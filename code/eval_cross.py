"""
Zero-shot cross-dataset generalization: LEVIR-trained DCFF-Net evaluated on another
change-detection dataset (default DSIFN-CD test). Fully-convolutional, so it runs at
native resolution; optional resize for scale study. Reports P/R/F1/IoU (+ optional TTA).

    python eval_cross.py --ckpt ../weights/dcff_final_best.pt --size 512
    python eval_cross.py --ckpt ../weights/dcff_final_best.pt --size 256 --tta
"""
import argparse, glob, io, numpy as np, torch
import pandas as pd
from PIL import Image
from models import DCFFNet

IM = np.array([0.485, 0.456, 0.406], np.float32)
IS = np.array([0.229, 0.224, 0.225], np.float32)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# dataset locations come from paths.py, never from a literal in this file


def op(c):
    return Image.open(io.BytesIO(c["bytes"] if isinstance(c, dict) else c))


def norm(a):
    return torch.from_numpy(((a / 255.0 - IM) / IS).transpose(2, 0, 1).astype(np.float32))


def tta(x):
    return [(lambda t: t, lambda t: t),
            (lambda t: torch.flip(t, [3]), lambda t: torch.flip(t, [3])),
            (lambda t: torch.flip(t, [2]), lambda t: torch.flip(t, [2])),
            (lambda t: torch.rot90(t, 1, [2, 3]), lambda t: torch.rot90(t, -1, [2, 3])),
            (lambda t: torch.rot90(t, 2, [2, 3]), lambda t: torch.rot90(t, -2, [2, 3])),
            (lambda t: torch.rot90(t, 3, [2, 3]), lambda t: torch.rot90(t, -3, [2, 3]))]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--t1", default="t1_image"); ap.add_argument("--t2", default="t2_image")
    ap.add_argument("--mask", default="change_mask")
    ap.add_argument("--parquet", default=None)
    args = ap.parse_args()

    path = args.parquet or glob.glob(DSIFN)[0]
    df = pd.read_parquet(path).reset_index(drop=True)
    model = DCFFNet(backbone="resnet18", pretrained=False, use_cbam=False, use_aspp=True).to(DEV).eval()
    model.load_state_dict(torch.load(args.ckpt, map_location=DEV))
    tfs = tta(None) if args.tta else [(lambda t: t, lambda t: t)]
    print(f"[LOG] {path.split('/')[-1]} | N={len(df)} | size={args.size} | TTA={args.tta}", flush=True)

    tp = fp = fn = 0.0
    for i in range(len(df)):
        r = df.iloc[i]
        S = (args.size, args.size)
        a = np.array(op(r[args.t1]).convert("RGB").resize(S), np.uint8)
        b = np.array(op(r[args.t2]).convert("RGB").resize(S), np.uint8)
        y = (np.array(op(r[args.mask]).convert("L").resize(S, Image.NEAREST), np.uint8) > 127).astype(np.float32)
        xa = norm(a)[None].to(DEV); xb = norm(b)[None].to(DEV)
        prob = torch.zeros(1, 1, args.size, args.size, device=DEV)
        for fwd, inv in tfs:
            prob += inv(torch.sigmoid(model(fwd(xa), fwd(xb))))
        prob /= len(tfs)
        pred = (prob[0, 0].cpu().numpy() > 0.5).astype(np.float32)
        tp += float(((pred == 1) & (y == 1)).sum())
        fp += float(((pred == 1) & (y == 0)).sum())
        fn += float(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp + 1e-9); rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9); iou = tp / (tp + fp + fn + 1e-9)
    print(f"[RESULT size={args.size} tta={args.tta}] P={prec:.4f} R={rec:.4f} F1={f1:.4f} IoU={iou:.4f}", flush=True)


if __name__ == "__main__":
    main()
