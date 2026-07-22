"""
Run the trained DCFF-Net on one real LEVIR-CD tile and save a visualisation of
every intermediate stage, so Figure 2 can show what actually happens to an
image as it passes through the network.

Each feature tensor is reduced to a single map by the L2 norm over channels,
contrast-stretched between its 2nd and 98th percentile, and coloured.

    python paper/extract_flow_assets.py
"""
import os, io, sys, glob
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "figures", "_assets")
CKPT = os.path.join(ROOT, "weights", "dcff_final_best.pt")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(ROOT, "code"))

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
SIZE = 256


def decode(cell):
    if hasattr(cell, "convert"):
        return cell
    b = cell["bytes"] if isinstance(cell, dict) else cell
    return Image.open(io.BytesIO(b))


def colour(t, cmap="viridis", nearest=False):
    """(C,H,W) tensor -> contrast-stretched RGB image at SIZE x SIZE.

    nearest=True keeps the true grid visible, which matters for the
    multi-scale figure where the point is that resolution drops.
    """
    m = np.linalg.norm(t, axis=0)
    lo, hi = np.percentile(m, 2), np.percentile(m, 98)
    m = np.clip((m - lo) / max(hi - lo, 1e-8), 0, 1)
    rgb = (matplotlib.colormaps[cmap](m)[..., :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb).resize(
        (SIZE, SIZE), Image.NEAREST if nearest else Image.BICUBIC)


def main():
    pq = glob.glob(os.path.join(ROOT, "data", "hf_cache", "hub", "**",
                                "test-*.parquet"), recursive=True)
    if not pq:
        raise SystemExit("LEVIR-CD test parquet not found under data/hf_cache/")
    df = pd.read_parquet(pq[0])

    pick, best = 0, -1.0
    for i in range(min(400, len(df))):
        g = np.array(decode(df.iloc[i]["label"]).convert("L"))
        r = (g > 127).mean()
        if 0.08 <= r <= 0.22 and r > best:
            pick, best = i, r
    print("tile %d (change ratio %.3f)" % (pick, best))
    row = df.iloc[pick]

    a = np.array(decode(row["imageA"]).convert("RGB"), np.uint8)
    b = np.array(decode(row["imageB"]).convert("RGB"), np.uint8)
    gt = (np.array(decode(row["label"]).convert("L"), np.uint8) > 127)

    Image.fromarray(a).save(os.path.join(OUT, "flow_t1.png"))
    Image.fromarray(b).save(os.path.join(OUT, "flow_t2.png"))
    Image.fromarray(np.dstack([(gt * 255).astype(np.uint8)] * 3)).save(
        os.path.join(OUT, "flow_gt.png"))

    if not os.path.exists(CKPT):
        raise SystemExit("checkpoint not found: %s" % CKPT)

    import torch
    import torch.nn.functional as F
    from models import DCFFNet

    def norm(x):
        return torch.from_numpy(((x / 255.0 - MEAN) / STD)
                                .transpose(2, 0, 1).astype(np.float32))[None]

    net = DCFFNet(backbone="resnet18", pretrained=False,
                  use_cbam=False, use_aspp=True).eval()
    net.load_state_dict(torch.load(CKPT, map_location="cpu"))

    xa, xb = norm(a), norm(b)
    with torch.no_grad():
        # ---- replay the forward pass, keeping every intermediate ----------
        fa, fb = net.enc(xa), net.enc(xb)
        blk = net.fuse[0]                                  # finest scale, 1/4
        diff = torch.abs(fa[0] - fb[0])
        conc = blk.conc(torch.cat([fa[0], fb[0]], dim=1))
        fused0 = blk.fuse(torch.cat([diff, conc], dim=1))

        p = [net.fuse[i](fa[i], fb[i]) for i in range(4)]
        d4 = net.aspp(p[3])
        d3 = net.up3(torch.cat([net._up(d4, p[2]), p[2]], 1))
        d2 = net.up2(torch.cat([net._up(d3, p[1]), p[1]], 1))
        d1 = net.up1(torch.cat([net._up(d2, p[0]), p[0]], 1))
        logit = F.interpolate(net.head(d1), size=xa.shape[-2:],
                              mode="bilinear", align_corners=False)
        prob = torch.sigmoid(logit)[0, 0].numpy()

    n = lambda t: t[0].numpy()
    saves = [
        ("flow_enc1", n(fa[0]), "Blues"),      # encoder features, T1
        ("flow_enc2", n(fb[0]), "Blues"),      # encoder features, T2
        ("flow_diff", n(diff), "Oranges"),     # difference cue
        ("flow_conc", n(conc), "Purples"),     # concatenation cue
        ("flow_fused", n(fused0), "Reds"),     # fused feature
        ("flow_aspp", n(d4), "Greens"),        # ASPP output
        ("flow_dec", n(d1), "Purples"),        # decoder output
    ]
    for name, t, cm in saves:
        colour(t, cm).save(os.path.join(OUT, "%s.png" % name))
        print("  %-12s %s -> %s.png" % (name, tuple(t.shape), name))

    # ---- the same two cues at ALL FOUR scales, for the multi-scale figure --
    print("  multi-scale:")
    with torch.no_grad():
        for i in range(4):
            blk_i = net.fuse[i]
            d_i = torch.abs(fa[i] - fb[i])
            c_i = blk_i.conc(torch.cat([fa[i], fb[i]], dim=1))
            f_i = blk_i.fuse(torch.cat([d_i, c_i], dim=1))
            for tag, ten, cm in [("enc1", fa[i], "Blues"), ("enc2", fb[i], "Blues"),
                                 ("diff", d_i, "Oranges"), ("conc", c_i, "Purples"),
                                 ("fused", f_i, "Reds")]:
                colour(ten[0].numpy(), cm, nearest=True).save(
                    os.path.join(OUT, "scale%d_%s.png" % (i + 1, tag)))
            print("    scale %d  %2d x %2d px" % (i + 1, d_i.shape[-2], d_i.shape[-1]))

    Image.fromarray(
        (matplotlib.colormaps["magma"](prob)[..., :3] * 255).astype(np.uint8)
    ).save(os.path.join(OUT, "flow_prob.png"))

    pred = prob > 0.5
    Image.fromarray(np.dstack([(pred * 255).astype(np.uint8)] * 3)).save(
        os.path.join(OUT, "flow_pred.png"))

    err = np.zeros(gt.shape + (3,), np.uint8)
    err[pred & gt] = (56, 176, 88)
    err[pred & ~gt] = (214, 69, 69)
    err[~pred & gt] = (58, 110, 214)
    Image.fromarray(err).save(os.path.join(OUT, "flow_err.png"))

    tp = int((pred & gt).sum()); fp = int((pred & ~gt).sum()); fn = int((~pred & gt).sum())
    print("tile F1 = %.4f" % (2 * tp / max(2 * tp + fp + fn, 1)))


if __name__ == "__main__":
    main()
