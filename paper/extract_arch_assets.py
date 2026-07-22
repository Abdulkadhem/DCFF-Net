"""
Prepare the real imagery embedded in the architecture figure.

Picks one LEVIR-CD test tile from the local parquet cache, runs the trained
DCFF-Net checkpoint on it, and writes the inputs, the ground truth and the
model's ACTUAL prediction (both the probability map and the binary decision).

    python paper/extract_arch_assets.py
"""
import os, io, sys, glob
import numpy as np
import pandas as pd
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "figures", "_assets")
CKPT = os.path.join(ROOT, "weights", "dcff_final_best.pt")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(ROOT, "code"))

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def decode(cell):
    if hasattr(cell, "convert"):
        return cell
    b = cell["bytes"] if isinstance(cell, dict) else cell
    return Image.open(io.BytesIO(b))


def heat(prob):
    """probability map -> RGB heat image (dark blue .. yellow)."""
    import matplotlib
    return (matplotlib.colormaps["magma"](prob)[..., :3] * 255).astype(np.uint8)


def main():
    pq = glob.glob(os.path.join(ROOT, "data", "hf_cache", "hub", "**",
                                "test-*.parquet"), recursive=True)
    if not pq:
        raise SystemExit("LEVIR-CD test parquet not found under data/hf_cache/")
    df = pd.read_parquet(pq[0])

    # a tile whose change is large enough to read at thumbnail size
    pick, best = 0, -1.0
    for i in range(min(400, len(df))):
        gt = np.array(decode(df.iloc[i]["label"]).convert("L"))
        r = (gt > 127).mean()
        if 0.08 <= r <= 0.22 and r > best:
            pick, best = i, r
    print("selected tile %d (change ratio %.3f)" % (pick, best))
    row = df.iloc[pick]

    a = np.array(decode(row["imageA"]).convert("RGB"), np.uint8)
    b = np.array(decode(row["imageB"]).convert("RGB"), np.uint8)
    gt = (np.array(decode(row["label"]).convert("L"), np.uint8) > 127)

    Image.fromarray(a).save(os.path.join(OUT, "arch_t1.png"))
    Image.fromarray(b).save(os.path.join(OUT, "arch_t2.png"))
    Image.fromarray(np.dstack([(gt * 255).astype(np.uint8)] * 3)).save(
        os.path.join(OUT, "arch_gt.png"))

    # ---- run the real model ------------------------------------------
    if not os.path.exists(CKPT):
        print("checkpoint missing (%s) -- inputs written, prediction skipped" % CKPT)
        return
    import torch
    from models import DCFFNet

    def norm(x):
        return torch.from_numpy(
            ((x / 255.0 - IMAGENET_MEAN) / IMAGENET_STD)
            .transpose(2, 0, 1).astype(np.float32))[None]

    model = DCFFNet(backbone="resnet18", pretrained=False,
                    use_cbam=False, use_aspp=True).eval()
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    with torch.no_grad():
        prob = torch.sigmoid(model(norm(a), norm(b)))[0, 0].numpy()

    pred = prob > 0.5
    tp = int((pred & gt).sum()); fp = int((pred & ~gt).sum()); fn = int((~pred & gt).sum())
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    print("tile F1 = %.4f  (TP %d, FP %d, FN %d)" % (f1, tp, fp, fn))

    Image.fromarray(heat(prob)).save(os.path.join(OUT, "arch_prob.png"))
    Image.fromarray(np.dstack([(pred * 255).astype(np.uint8)] * 3)).save(
        os.path.join(OUT, "arch_pred.png"))

    # error map: green TP, red FP, blue FN -- the paper's colour convention
    err = np.zeros(gt.shape + (3,), np.uint8)
    err[pred & gt] = (56, 176, 88)
    err[pred & ~gt] = (214, 69, 69)
    err[~pred & gt] = (58, 110, 214)
    Image.fromarray(err).save(os.path.join(OUT, "arch_err.png"))

    for n in ("arch_prob", "arch_pred", "arch_err"):
        print("wrote", os.path.join(OUT, n + ".png"))


if __name__ == "__main__":
    main()
