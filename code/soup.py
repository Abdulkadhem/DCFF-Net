"""
Model soup: average the weights of several DCFF-Net checkpoints and evaluate.
Works only if the members share a loss basin (same init / same seed) — otherwise the
average collapses, which is itself an informative (cheap) negative result.

    python soup.py --ckpts results/dcff_final_best.pt results/levir_dcff_long_best.pt --dataset levir
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
def evaluate(model, loader, scales=(1.0,), use_views=False):
    vs = views() if use_views else [views()[0]]
    tp = fp = fn = 0.0
    for xa, xb, y in loader:
        xa, xb, y = xa.to(DEV), xb.to(DEV), y.to(DEV)
        H, W = y.shape[-2:]
        prob = torch.zeros_like(y); n = 0
        for s in scales:
            a, b = (xa, xb) if s == 1.0 else (
                F.interpolate(xa, scale_factor=s, mode="bilinear", align_corners=False),
                F.interpolate(xb, scale_factor=s, mode="bilinear", align_corners=False))
            for fwd, inv in vs:
                p = inv(torch.sigmoid(model(fwd(a), fwd(b))))
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
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--dataset", default="levir")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--mstta", action="store_true")
    args = ap.parse_args()

    sds = [torch.load(c, map_location="cpu") for c in args.ckpts]
    avg = {k: sum(sd[k].float() for sd in sds) / len(sds) for k in sds[0]}
    model = DCFFNet(backbone="resnet18", pretrained=False, use_cbam=False, use_aspp=True).to(DEV).eval()
    model.load_state_dict({k: v.to(sds[0][k].dtype) for k, v in avg.items()})

    te = DataLoader(make_cd(args.dataset, "test", augment=False), batch_size=args.batch, num_workers=8)
    print(f"[SOUP] {len(sds)} members: {[c.split('/')[-1] for c in args.ckpts]} | N={len(te.dataset)}", flush=True)

    p, r, f1, iou = evaluate(model, te)
    print(f"[soup plain ] P={p:.4f} R={r:.4f} F1={f1:.4f} IoU={iou:.4f}", flush=True)
    if args.mstta:
        p, r, f1, iou = evaluate(model, te, scales=(1.0, 1.25, 1.5), use_views=True)
        print(f"[soup MS-TTA] P={p:.4f} R={r:.4f} F1={f1:.4f} IoU={iou:.4f}", flush=True)
    print("[ALL_DONE]", flush=True)


if __name__ == "__main__":
    main()
