"""
Generic HuggingFace change-detection loader for in-domain multi-dataset training.
One registry, one Dataset class. Handles the ericyu-style layout (imageA/imageB/label,
256x256) and the torchange DSIFN layout (t1_image/t2_image/change_mask, 512 -> crop 256).

    from data_cd import make_cd
    ds = make_cd("egy", "train")          # ericyu/EGY_BCD
    ds = make_cd("dsifn", "test", augment=False)
"""
import io, numpy as np, torch
from torch.utils.data import Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)

# name -> (repo_id, (colA, colB, colLabel), native_size)
REGISTRY = {
    "levir": ("ericyu/LEVIRCD_Cropped_256", ("imageA", "imageB", "label"), 256),
    "egy":   ("ericyu/EGY_BCD",             ("imageA", "imageB", "label"), 256),
    "clcd":  ("ericyu/CLCD_Cropped_256",    ("imageA", "imageB", "label"), 256),
    "sysu":  ("ericyu/SYSU_CD",             ("imageA", "imageB", "label"), 256),
    "gvlm":  ("ericyu/GVLM_Cropped_256",    ("imageA", "imageB", "label"), 256),
    "dsifn": ("EVER-Z/torchange_dsifn-cd",  ("t1_image", "t2_image", "change_mask"), 512),
}

SPLIT_ALIASES = {"val": ["validation", "val"], "validation": ["validation", "val"],
                 "train": ["train"], "test": ["test"]}


def _norm(img):
    x = img.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1))


def _strong_aug(a, b, y, rng, Image):
    """scale-jitter crop (resized back) + joint photometric jitter, applied IDENTICALLY
    to T1/T2 so change semantics are preserved. Targets small-dataset overfitting."""
    H, W = y.shape
    s = float(rng.uniform(0.70, 1.0))
    ch, cw = max(32, int(H * s)), max(32, int(W * s))
    top = int(rng.integers(0, H - ch + 1)); left = int(rng.integers(0, W - cw + 1))
    a = np.array(Image.fromarray(a[top:top + ch, left:left + cw]).resize((W, H), Image.BILINEAR))
    b = np.array(Image.fromarray(b[top:top + ch, left:left + cw]).resize((W, H), Image.BILINEAR))
    ym = (y[top:top + ch, left:left + cw] * 255).astype(np.uint8)
    y = (np.array(Image.fromarray(ym).resize((W, H), Image.NEAREST)) > 127).astype(np.float32)
    br = float(rng.uniform(0.88, 1.12)); ct = float(rng.uniform(0.88, 1.12))   # joint jitter
    a = np.clip((a.astype(np.float32) - 128) * ct + 128 * br, 0, 255).astype(np.uint8)
    b = np.clip((b.astype(np.float32) - 128) * ct + 128 * br, 0, 255).astype(np.uint8)
    return a, b, y


class HFChangeDetection(Dataset):
    def __init__(self, name, split="train", augment=None, crop=256, strong=False):
        from datasets import load_dataset
        from PIL import Image
        self.Image = Image
        repo, cols, native = REGISTRY[name]
        self.cA, self.cB, self.cL = cols
        self.crop = crop
        self.need_crop = native != crop           # DSIFN 512 -> 256
        ds = load_dataset(repo, verification_mode="no_checks")
        key = next((k for k in SPLIT_ALIASES[split] if k in ds), split)
        self.ds = ds[key]
        self.augment = (split == "train") if augment is None else augment
        self.strong = strong
        self.rng = np.random.default_rng()

    def __len__(self):
        return len(self.ds)

    def _arr(self, cell, mode="RGB"):
        if hasattr(cell, "convert"):                    # datasets already decoded -> PIL
            im = cell
        else:                                           # raw bytes / {"bytes": ...}
            b = cell["bytes"] if isinstance(cell, dict) else cell
            im = self.Image.open(io.BytesIO(b))
        return np.array(im.convert(mode), np.uint8)

    def __getitem__(self, i):
        ex = self.ds[i]
        a = self._arr(ex[self.cA]); b = self._arr(ex[self.cB])
        y = (self._arr(ex[self.cL], "L") > 127).astype(np.float32)
        H, W = y.shape; c = self.crop
        if c > H or c > W:            # FixRes-style: fine-tune/eval at a HIGHER resolution
            I = self.Image
            a = np.array(I.fromarray(a).resize((c, c), I.BILINEAR))
            b = np.array(I.fromarray(b).resize((c, c), I.BILINEAR))
            y = (np.array(I.fromarray((y * 255).astype(np.uint8)).resize((c, c), I.NEAREST)) > 127).astype(np.float32)
        elif self.need_crop:
            if self.augment:
                top = np.random.randint(0, H - c + 1); left = np.random.randint(0, W - c + 1)
            else:
                top = (H - c) // 2; left = (W - c) // 2
            a = a[top:top+c, left:left+c]; b = b[top:top+c, left:left+c]; y = y[top:top+c, left:left+c]
        if self.augment:
            if self.strong:
                a, b, y = _strong_aug(a, b, y, self.rng, self.Image)
            if np.random.rand() < 0.5: a, b, y = a[:, ::-1], b[:, ::-1], y[:, ::-1]
            if np.random.rand() < 0.5: a, b, y = a[::-1], b[::-1], y[::-1]
            k = np.random.randint(4)
            if k: a, b, y = np.rot90(a, k), np.rot90(b, k), np.rot90(y, k)
            a, b, y = map(np.ascontiguousarray, (a, b, y))
        return _norm(a), _norm(b), torch.from_numpy(y)[None]


def make_cd(name, split="train", augment=None, crop=256, strong=False):
    return HFChangeDetection(name, split, augment=augment, crop=crop, strong=strong)


if __name__ == "__main__":
    import sys
    nm = sys.argv[1] if len(sys.argv) > 1 else "egy"
    for sp in ["train", "val", "test"]:
        try:
            d = make_cd(nm, sp, augment=False)
            xa, xb, y = d[0]
            print(f"{nm} {sp}: N={len(d)} xa={tuple(xa.shape)} pos={float(y.mean()):.3f}")
        except Exception as e:
            print(f"{nm} {sp}: {e}")
