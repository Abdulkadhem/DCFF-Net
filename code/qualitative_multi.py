"""
Qualitative change-detection panels for the NEW in-domain datasets (EGY_BCD, DSIFN),
reusing the LEVIR renderer in make_qualitative.py. [T1 | T2 | GT | Prediction | Error map]
with large titles + bottom legend. Reads each dataset's TEST parquet directly from the
local HF cache (no train/val download). CPU-only.

    python qualitative_multi.py --dataset egy   --ckpt ../results/multidataset/egy_dcff_best.pt
    python qualitative_multi.py --dataset dsifn --ckpt ../results/multidataset/dsifn_dcff_best.pt
"""
import argparse, io, numpy as np, torch
import pandas as pd
from PIL import Image
from models import DCFFNet
from paths import find_parquet
import make_qualitative as mq

# dataset -> ((registry name, split), (colA, colB, colLabel), native size)
SPEC = {
    "egy":   (("egy", "test"),   ("imageA", "imageB", "label"), 256),
    "dsifn": (("dsifn", "test"), ("t1_image", "t2_image", "change_mask"), 512),
}
DEV = mq.DEV


class QualDS:
    """test-only reader with make_qualitative's ds interface (label_np + __getitem__)."""
    def __init__(self, dataset):
        (name, split), cols, native = SPEC[dataset]
        self.cA, self.cB, self.cL = cols
        self.native, self.crop = native, 256
        self.df = pd.read_parquet(find_parquet(name, split)[0]).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def _img(self, cell, mode="RGB"):
        b = cell["bytes"] if isinstance(cell, dict) else cell
        return Image.open(io.BytesIO(b)).convert(mode)

    def _cc(self, a):
        """No cropping: the adopted DSIFN model is trained/evaluated at native 512, so we
        feed the model the native image and only down-sample the *rendered* tiles later."""
        return a

    def label_np(self, i):                             # HxW uint8 0/255 (cropped)
        return self._cc(np.array(self._img(self.df.iloc[i][self.cL], "L"), np.uint8))

    def __getitem__(self, i):
        r = self.df.iloc[i]
        a = self._cc(np.array(self._img(r[self.cA]), np.uint8))
        b = self._cc(np.array(self._img(r[self.cB]), np.uint8))
        y = (self._cc(np.array(self._img(r[self.cL], "L"), np.uint8)) > 127).astype(np.float32)
        xa = torch.from_numpy(((a / 255.0 - mq.IMAGENET_MEAN) / mq.IMAGENET_STD).transpose(2, 0, 1).astype(np.float32))
        xb = torch.from_numpy(((b / 255.0 - mq.IMAGENET_MEAN) / mq.IMAGENET_STD).transpose(2, 0, 1).astype(np.float32))
        return xa, xb, torch.from_numpy(y)[None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["egy", "dsifn"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()

    ds = QualDS(args.dataset)
    model = DCFFNet(backbone="resnet18", pretrained=False, use_cbam=False, use_aspp=True).to(DEV).eval()
    model.load_state_dict(torch.load(args.ckpt, map_location=DEV))
    print(f"[LOG] {args.dataset} | N={len(ds)}", flush=True)

    stride = max(1, len(ds) // 200)
    ratios = [(i, float((ds.label_np(i) > 127).mean())) for i in range(0, len(ds), stride)]
    pool = [i for i, r in ratios if 0.05 < r < 0.55]
    sel = [pool[int(round(k))] for k in np.linspace(0, len(pool) - 1, args.n)]
    print("[LOG] picks", sel, flush=True)

    GREEN, RED, BLUE = (40, 170, 70), (210, 60, 60), (60, 90, 200)

    def tiles_resized(idx, m, d):
        """run the model at the dataset's native resolution, then down-sample the rendered
        tiles to the 256-px canvas cell so every qualitative figure has identical geometry."""
        tiles = mq.tiles_error(idx, m, d)
        out = []
        for t in tiles:
            im = Image.fromarray(t)
            if im.size != (256, 256):
                im = im.resize((256, 256), Image.BILINEAR if t.ndim == 3 else Image.NEAREST)
            out.append(np.array(im))
        return out

    out = f"../paper/figures/fig_qualitative_{args.dataset}.png"
    mq.render(out, sel,
              ["T1 (before)", "T2 (after)", "Ground truth", "Prediction", "Error map"],
              tiles_resized,
              [(GREEN, "True positive"), (RED, "False positive"), (BLUE, "False negative")],
              model, ds)
    print("[ALL_DONE]", flush=True)


if __name__ == "__main__":
    main()
