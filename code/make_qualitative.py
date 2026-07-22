"""
Qualitative change-detection panels on LEVIR-CD test using DCFF-Net (final model).
Generates THREE figures (6 examples each), with LARGE column titles + a bottom color legend.
PIL only (no matplotlib dependency on the GPU box).

  1) qualitative.png          [T1 | T2 | Ground Truth | Our Prediction | Error map]
       legend: True Positive (green) / False Positive (red) / False Negative (blue)
  2) qualitative_heatmap.png  [T1 | T2 | Ground Truth | Confidence P(change) | Our Prediction]
       legend: low -> medium -> high confidence (heatmap colorbar)
  3) qualitative_overlay.png  [T1 | T2 | GT on T2 | Prediction on T2 | Error map]
       legend: Ground truth (yellow) / Prediction (cyan) / TP-FP-FN

    python code/make_qualitative.py --ckpt weights/dcff_final_best.pt --outdir paper/figures
"""
import argparse, os, io, numpy as np, torch
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from models import DCFFNet
from paths import find_parquet, ROOT

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ParquetCD:
    """Reads a LEVIRCD_Cropped_256 split parquet directly from the local HF cache
    (no HuggingFace Hub contact -> instant, immune to network throttling)."""
    def __init__(self, split="test"):
        self.df = pd.read_parquet(find_parquet("levir", split)[0]).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def _img(self, cell):  # HF image column cell -> PIL
        b = cell["bytes"] if isinstance(cell, dict) else cell
        return Image.open(io.BytesIO(b)).convert("RGB")

    def label_np(self, i):  # HxW uint8 0/255
        row = self.df.iloc[i]
        m = Image.open(io.BytesIO(row["label"]["bytes"] if isinstance(row["label"], dict) else row["label"]))
        return np.array(m.convert("L"), np.uint8)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        a = np.array(self._img(row["imageA"]), np.uint8)
        b = np.array(self._img(row["imageB"]), np.uint8)
        y = (self.label_np(i) > 127).astype(np.float32)
        xa = torch.from_numpy(((a / 255.0 - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1).astype(np.float32))
        xb = torch.from_numpy(((b / 255.0 - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1).astype(np.float32))
        return xa, xb, torch.from_numpy(y)[None]


def get_font(size, bold=True):
    names = (["DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"] if bold
             else ["DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"])
    for p in names:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    try:
        return ImageFont.load_default(size=size)   # Pillow >= 10.1
    except Exception:
        return ImageFont.load_default()


def denorm(t):  # 3xHxW normalized -> HxWx3 uint8
    x = t.numpy().transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def mask_rgb(m):  # HxW {0,1} -> white change on black
    return np.stack([m * 255] * 3, -1).astype(np.uint8)


def error_rgb(pred, gt):  # TP green, FP red, FN blue, TN white
    h, w = gt.shape
    out = np.full((h, w, 3), 255, np.uint8)
    tp = (pred == 1) & (gt == 1); fp = (pred == 1) & (gt == 0); fn = (pred == 0) & (gt == 1)
    out[tp] = [40, 170, 70]; out[fp] = [210, 60, 60]; out[fn] = [60, 90, 200]
    return out


def heat_rgb(prob):  # HxW in [0,1] -> jet-like heatmap (blue->cyan->green->yellow->red)
    p = np.clip(prob, 0, 1)
    r = np.clip(1.5 - np.abs(4 * p - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * p - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * p - 1), 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def heat_color(p):  # scalar prob -> (r,g,b) tuple for legend swatch
    return tuple(int(v) for v in heat_rgb(np.array([[float(p)]]))[0, 0])


def overlay(img, mask, color, alpha=0.55):  # img HxWx3 uint8, mask HxW {0,1}
    out = img.astype(np.float32).copy()
    m = mask.astype(bool)
    for c in range(3):
        out[..., c][m] = (1 - alpha) * out[..., c][m] + alpha * color[c]
    return np.clip(out, 0, 255).astype(np.uint8)


# ---- per-example tile builders (each returns list of 5 HxWx3 uint8) --------
def tiles_error(idx, model, ds):
    xa, xb, y = ds[idx]
    gt = y[0].numpy().astype(np.uint8)
    prob = torch.sigmoid(model(xa[None].to(DEV), xb[None].to(DEV)))[0, 0].cpu().numpy()
    pred = (prob > 0.5).astype(np.uint8)
    return [denorm(xa), denorm(xb), mask_rgb(gt), mask_rgb(pred), error_rgb(pred, gt)]


def tiles_heat(idx, model, ds):
    xa, xb, y = ds[idx]
    gt = y[0].numpy().astype(np.uint8)
    prob = torch.sigmoid(model(xa[None].to(DEV), xb[None].to(DEV)))[0, 0].cpu().numpy()
    pred = (prob > 0.5).astype(np.uint8)
    return [denorm(xa), denorm(xb), mask_rgb(gt), heat_rgb(prob), mask_rgb(pred)]


def tiles_overlay(idx, model, ds):
    xa, xb, y = ds[idx]
    gt = y[0].numpy().astype(np.uint8)
    t2 = denorm(xb)
    prob = torch.sigmoid(model(xa[None].to(DEV), xb[None].to(DEV)))[0, 0].cpu().numpy()
    pred = (prob > 0.5).astype(np.uint8)
    gt_ov = overlay(t2, gt, [255, 215, 0])     # yellow
    pr_ov = overlay(t2, pred, [0, 220, 220])   # cyan
    return [denorm(xa), t2, gt_ov, pr_ov, error_rgb(pred, gt)]


def render(path, picks, cols, tile_fn, legend, model, ds):
    S, G = 256, 8
    ncol = len(cols)
    Hd, Hf = 46, 70                       # header (titles) + footer (legend) heights
    W = ncol * S + (ncol + 1) * G
    H = Hd + len(picks) * (S + G) + G + Hf
    canvas = Image.new("RGB", (W, H), (250, 250, 250))
    dr = ImageDraw.Draw(canvas)
    # unified typography across ALL qualitative figures: DejaVu Sans Bold, 24 pt titles / 22 pt legend
    tfont = get_font(24, bold=True)
    lfont = get_font(22, bold=True)

    # column titles: centred on the column, then clamped so nothing overflows the canvas
    for c, name in enumerate(cols):
        cx = G + c * (S + G) + S // 2
        bb = dr.textbbox((0, 0), name, font=tfont)
        tw = bb[2] - bb[0]
        x = min(max(cx - tw // 2, G), W - G - tw)
        dr.text((x, 11), name, fill=(20, 20, 20), font=tfont)

    # image rows
    with torch.no_grad():
        for r, idx in enumerate(picks):
            for c, t in enumerate(tile_fn(idx, model, ds)):
                x0 = G + c * (S + G); y0 = Hd + r * (S + G)
                canvas.paste(Image.fromarray(t), (x0, y0))

    # bottom legend (centered row of colored swatches + labels)
    sq, pad, gap = 30, 12, 46
    widths = []
    for _, txt in legend:
        bb = dr.textbbox((0, 0), txt, font=lfont)
        widths.append(sq + pad + (bb[2] - bb[0]))
    total = sum(widths) + gap * (len(legend) - 1)
    x = (W - total) // 2
    ly = H - Hf + 18
    for (col, txt), wd in zip(legend, widths):
        dr.rectangle([x, ly, x + sq, ly + sq], fill=tuple(col), outline=(60, 60, 60))
        dr.text((x + sq + pad, ly + 2), txt, fill=(20, 20, 20), font=lfont)
        x += wd + gap

    canvas.save(path)
    print("[SAVED]", path, canvas.size, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--outdir", default=os.path.join(ROOT, "paper", "figures"))
    args = ap.parse_args()

    ds = ParquetCD("test")
    print("[LOG] test split loaded:", len(ds), "examples", flush=True)
    model = DCFFNet(backbone=args.backbone, pretrained=False, use_cbam=False, use_aspp=True).to(DEV).eval()
    model.load_state_dict(torch.load(args.ckpt, map_location=DEV))

    # candidate pool with a visible amount of change, then spread 18 distinct examples
    ratios = [(i, float((ds.label_np(i) > 127).mean())) for i in range(0, len(ds), 3)]
    pool = [i for i, r in ratios if 0.06 < r < 0.45]
    sel = [pool[int(round(k))] for k in np.linspace(0, len(pool) - 1, 18)]
    p_err, p_heat, p_ov = sel[0:6], sel[6:12], sel[12:18]
    print("[LOG] pool", len(pool), "| err", p_err, "| heat", p_heat, "| overlay", p_ov, flush=True)

    GREEN, RED, BLUE = (40, 170, 70), (210, 60, 60), (60, 90, 200)
    YELLOW, CYAN = (255, 215, 0), (0, 220, 220)

    render(f"{args.outdir}/qualitative.png", p_err,
           ["T1 (before)", "T2 (after)", "Ground truth", "Prediction", "Error map"],
           tiles_error,
           [(GREEN, "True positive"), (RED, "False positive"), (BLUE, "False negative")],
           model, ds)

    render(f"{args.outdir}/qualitative_heatmap.png", p_heat,
           ["T1 (before)", "T2 (after)", "Ground truth", "Confidence", "Prediction"],
           tiles_heat,
           [(heat_color(0.05), "low P(change)"), (heat_color(0.5), "medium P(change)"),
            (heat_color(0.95), "high P(change)")],
           model, ds)

    render(f"{args.outdir}/qualitative_overlay.png", p_ov,
           ["T1 (before)", "T2 (after)", "GT overlay", "Pred. overlay", "Error map"],
           tiles_overlay,
           [(YELLOW, "Ground truth"), (CYAN, "Prediction"),
            (GREEN, "True positive"), (RED, "False positive"), (BLUE, "False negative")],
           model, ds)

    print("[ALL_DONE]", flush=True)


if __name__ == "__main__":
    main()
