"""
Multi-scale test-time augmentation: averages probabilities over {0.75x, 1.0x, 1.25x}
scales x {identity, hflip, vflip, rot90/180/270}. Inference only — no retraining.
Reports P/R/F1/IoU and compares against plain + single-scale TTA.

    python eval_mstta.py --ckpt results/dcff_final_best.pt --dataset levir
"""
import argparse, numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data_cd import make_cd
from models import DCFFNet

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def views():
    return [(lambda t: t, lambda t: t),
            (lambda t: torch.flip(t, [3]), lambda t: torch.flip(t, [3])),
            (lambda t: torch.flip(t, [2]), lambda t: torch.flip(t, [2])),
            (lambda t: torch.rot90(t, 1, [2, 3]), lambda t: torch.rot90(t, -1, [2, 3])),
            (lambda t: torch.rot90(t, 2, [2, 3]), lambda t: torch.rot90(t, -2, [2, 3])),
            (lambda t: torch.rot90(t, 3, [2, 3]), lambda t: torch.rot90(t, -3, [2, 3]))]


@torch.no_grad()
def run(model, loader, scales, use_views=True):
    vs = views() if use_views else [views()[0]]
    tp = fp = fn = 0.0
    for xa, xb, y in loader:
        xa, xb, y = xa.to(DEV), xb.to(DEV), y.to(DEV)
        H, W = y.shape[-2:]
        prob = torch.zeros_like(y); n = 0
        for s in scales:
            if s == 1.0:
                a, b = xa, xb
            else:
                a = F.interpolate(xa, scale_factor=s, mode="bilinear", align_corners=False)
                b = F.interpolate(xb, scale_factor=s, mode="bilinear", align_corners=False)
            for fwd, inv in vs:
                p = torch.sigmoid(model(fwd(a), fwd(b)))
                p = inv(p)
                if p.shape[-2:] != (H, W):
                    p = F.interpolate(p, size=(H, W), mode="bilinear", align_corners=False)
                prob += p; n += 1
        prob /= n
        pr = (prob > 0.5).float()
        tp += float((pr * y).sum()); fp += float((pr * (1 - y)).sum()); fn += float(((1 - pr) * y).sum())
    prec = tp / (tp + fp + 1e-9); rec = tp / (tp + fn + 1e-9)
    return prec, rec, 2 * prec * rec / (prec + rec + 1e-9), tp / (tp + fp + fn + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default="levir")
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    model = DCFFNet(backbone="resnet18", pretrained=False, use_cbam=False, use_aspp=True).to(DEV).eval()
    model.load_state_dict(torch.load(args.ckpt, map_location=DEV))
    te = DataLoader(make_cd(args.dataset, "test", augment=False, crop=args.crop),
                    batch_size=args.batch, num_workers=8)
    print(f"[LOG] {args.dataset} test N={len(te.dataset)} | {args.ckpt} | device={DEV}", flush=True)

    configs = [("plain (no TTA)", [1.0], False),
               ("TTA 6 views", [1.0], True),
               ("MS-TTA .75/1/1.25", [0.75, 1.0, 1.25], True),
               ("MS-TTA 1/1.25/1.5", [1.0, 1.25, 1.5], True)]
    for name, sc, uv in configs:
        p, r, f1, iou = run(model, te, sc, uv)
        print(f"[{name:20s}] P={p:.4f} R={r:.4f} F1={f1:.4f} IoU={iou:.4f}", flush=True)
    print("[ALL_DONE]", flush=True)


if __name__ == "__main__":
    main()
