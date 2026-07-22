"""
Train + evaluate DCFF-Net (or FC-Siam baseline) on LEVIR-CD.
Loss = BCE + Dice + Boundary, with deep supervision. Metrics: P/R/F1/IoU on change class.

    python code/train.py --model dcff --epochs 120 --batch 16
    python code/train.py --model fcsiam --epochs 120     # baseline
"""
import os, argparse, time, json
import numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data_cd import make_cd
from models import build_model

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------- losses ----------------
def dice_loss(logit, y, eps=1.0):
    p = torch.sigmoid(logit)
    num = 2 * (p * y).sum((2, 3)) + eps
    den = (p + y).sum((2, 3)) + eps
    return (1 - num / den).mean()


def boundary_map(y):
    """edges of the mask via a Laplacian-like kernel."""
    k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=y.dtype, device=y.device).view(1, 1, 3, 3)
    e = F.conv2d(y, k, padding=1).abs()
    return (e > 0).float()


USE_BOUNDARY = True  # set from CLI in main()
USE_LOVASZ = False   # set from CLI in main(); Lovasz-hinge directly optimises IoU


def _lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    inter = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jac = 1.0 - inter / union.clamp(min=1e-9)
    if p > 1:
        jac[1:p] = jac[1:p] - jac[0:-1].clone()
    return jac


def lovasz_hinge(logits, y):
    """binary Lovasz-hinge, averaged per image (surrogate for IoU)."""
    losses = []
    for lo, la in zip(logits, y):
        lo = lo.reshape(-1); la = la.reshape(-1)
        if la.sum() == 0:                     # no positives -> skip (keeps it stable)
            losses.append(lo.sum() * 0.0); continue
        signs = 2.0 * la - 1.0
        errors = 1.0 - lo * signs
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        grad = _lovasz_grad(la[perm])
        losses.append(torch.dot(F.relu(errors_sorted), grad))
    return torch.stack(losses).mean()


def seg_loss(logit, y, pos_weight):
    bce = F.binary_cross_entropy_with_logits(logit, y, pos_weight=pos_weight)
    dl = dice_loss(logit, y)
    if USE_LOVASZ:
        dl = dl + 0.5 * lovasz_hinge(logit, y)
    if not USE_BOUNDARY:
        return bce + dl
    bnd = F.binary_cross_entropy_with_logits(logit, boundary_map(y))
    return bce + dl + 0.5 * bnd


# ---------------- metrics ----------------
class CM:
    def __init__(self): self.tp = self.fp = self.fn = self.tn = 0
    def add(self, logit, y):
        p = (torch.sigmoid(logit) > 0.5).float()
        self.tp += float((p * y).sum()); self.fp += float((p * (1 - y)).sum())
        self.fn += float(((1 - p) * y).sum()); self.tn += float(((1 - p) * (1 - y)).sum())
    def scores(self):
        tp, fp, fn = self.tp, self.fp, self.fn
        prec = tp / (tp + fp + 1e-9); rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9); iou = tp / (tp + fp + fn + 1e-9)
        return dict(precision=prec, recall=rec, f1=f1, iou=iou)


@torch.no_grad()
def evaluate(model, loader):
    model.eval(); cm = CM()
    for xa, xb, y in loader:
        xa, xb, y = xa.to(DEV), xb.to(DEV), y.to(DEV)
        cm.add(model(xa, xb), y)
    return cm.scores()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dcff", choices=["dcff", "fcsiam"])
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--pretrained", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--pos_weight", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--fusion", default="dual", choices=["dual", "diff", "conc"])
    ap.add_argument("--no_cbam", action="store_true")
    ap.add_argument("--no_aspp", action="store_true")
    ap.add_argument("--no_boundary", action="store_true")
    ap.add_argument("--out", default="results")
    ap.add_argument("--tag", default="dcff")
    ap.add_argument("--dataset", default="levir",
                    choices=["levir", "egy", "clcd", "sysu", "gvlm", "dsifn"])
    ap.add_argument("--crop", type=int, default=256)   # 512 = train DSIFN at full resolution
    ap.add_argument("--strong", action="store_true")   # stronger augmentation (small datasets)
    ap.add_argument("--lovasz", action="store_true")   # add Lovasz-hinge (IoU surrogate)
    ap.add_argument("--rs_pretrain", action="store_true")  # remote-sensing SSL encoder (torchgeo)
    ap.add_argument("--resume", default=None)          # polish an existing checkpoint (low-lr fine-tune)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    global USE_LOVASZ; USE_LOVASZ = args.lovasz
    os.makedirs(args.out, exist_ok=True)
    global USE_BOUNDARY; USE_BOUNDARY = not args.no_boundary

    tr = DataLoader(make_cd(args.dataset, "train", crop=args.crop, strong=args.strong), batch_size=args.batch, shuffle=True,
                    num_workers=args.workers, drop_last=True, pin_memory=True)
    va = DataLoader(make_cd(args.dataset, "val", augment=False, crop=args.crop), batch_size=args.batch, num_workers=args.workers)
    te = DataLoader(make_cd(args.dataset, "test", augment=False, crop=args.crop), batch_size=args.batch, num_workers=args.workers)

    kw = (dict(backbone=args.backbone, pretrained=bool(args.pretrained), fusion=args.fusion,
               use_cbam=not args.no_cbam, use_aspp=not args.no_aspp,
               rs_pretrain=args.rs_pretrain) if args.model == "dcff" else {})
    model = build_model(args.model, **kw).to(DEV)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=DEV))
        print(f"[RESUME] polishing from {args.resume}", flush=True)
    nparam = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[LOG] {args.model} params={nparam:.2f}M device={DEV} train={len(tr.dataset)} "
          f"val={len(va.dataset)} test={len(te.dataset)}", flush=True)

    # backbone gets a smaller lr
    enc_ids = set(id(p) for n, p in model.named_parameters() if n.startswith("enc."))
    groups = [{"params": [p for p in model.parameters() if id(p) in enc_ids], "lr": args.lr * 0.1},
              {"params": [p for p in model.parameters() if id(p) not in enc_ids], "lr": args.lr}]
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEV.type == "cuda"))
    pos_weight = torch.tensor([args.pos_weight], device=DEV)  # tuned; change is rare

    best_f1, best = -1, {}
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); tot = 0
        for xa, xb, y in tr:
            xa, xb, y = xa.to(DEV), xb.to(DEV), y.to(DEV)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=(DEV.type == "cuda")):
                out = model(xa, xb)
                if isinstance(out, tuple):
                    main_out, aux = out
                    loss = seg_loss(main_out, y, pos_weight)
                    for a in aux: loss = loss + 0.4 * seg_loss(a, y, pos_weight)
                else:
                    loss = seg_loss(out, y, pos_weight)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += float(loss)
        sched.step()
        vs = evaluate(model, va)
        print(f"[E{ep+1}/{args.epochs}] loss={tot/len(tr):.3f} val_F1={vs['f1']:.4f} "
              f"val_IoU={vs['iou']:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if vs["f1"] > best_f1:
            best_f1 = vs["f1"]
            torch.save(model.state_dict(), os.path.join(args.out, f"{args.tag}_best.pt"))
            best = {"epoch": ep + 1, "val": vs}

    # final test with best checkpoint
    model.load_state_dict(torch.load(os.path.join(args.out, f"{args.tag}_best.pt"), map_location=DEV))
    ts = evaluate(model, te)
    print(f"\n=== TEST ({args.tag}) === F1={ts['f1']:.4f} IoU={ts['iou']:.4f} "
          f"P={ts['precision']:.4f} R={ts['recall']:.4f}  (best val@ep{best.get('epoch')})", flush=True)
    json.dump({"args": vars(args), "params_M": nparam, "best_val": best, "test": ts},
              open(os.path.join(args.out, f"{args.tag}_result.json"), "w"), indent=2)
    print("[ALL_DONE]", flush=True)


if __name__ == "__main__":
    main()
