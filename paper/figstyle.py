"""
Shared drawing primitives for the DCFF-Net block diagrams (Figs. 1 and 2).

Feature tensors are drawn as pseudo-3D slabs whose front face is the spatial
extent and whose depth is the channel axis, so that the usual CNN reading of
"the block shrinks as the stride grows" is preserved.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle, FancyArrowPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
import os

plt.rcParams["font.family"] = "DejaVu Sans"

# ------------------------------------------------------------------ palette
C_T1    = "#6aa3f5"   # features from T1
C_T2    = "#3767c4"   # features from T2
C_DIFF  = "#f0883e"   # difference cue
C_CONC  = "#9a6fd4"   # concatenation cue
C_FUSED = "#e05c5c"   # fused feature
C_WC    = "#f7d060"   # 1x1 convolution
C_ASPP  = "#4fae6d"   # ASPP
C_DEC   = "#7b68c9"   # decoder
C_CONF  = "#e8607a"   # conformal layer
C_OUT   = "#f2c14e"

BG_ENC  = "#eaf2ff"
BG_FUSE = "#fdf5e6"
BG_ASPP = "#eaf7ee"
BG_DEC  = "#f0ecff"
BG_CONF = "#fdecef"
BG_IN   = "#fafafa"

INK  = "#1c1c1c"
GREY = "#6b7280"


def shade(hexcol, f):
    h = hexcol.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    c = lambda v: max(0, min(255, int(v * f)))
    return "#%02x%02x%02x" % (c(r), c(g), c(b))


def slab(ax, cx, cy, w, h, d, color, lw=1.0, z=4):
    """
    A feature tensor. (cx, cy) is the centre of the front face, (w, h) its
    spatial extent and d the apparent channel depth.
    """
    x, y = cx - w / 2.0, cy - h / 2.0
    dx, dy = d, d * 0.72
    ax.add_patch(Polygon([(x, y + h), (x + dx, y + h + dy),
                          (x + w + dx, y + h + dy), (x + w, y + h)],
                         closed=True, fc=shade(color, 1.24), ec=INK, lw=lw, zorder=z))
    ax.add_patch(Polygon([(x + w, y), (x + w + dx, y + dy),
                          (x + w + dx, y + h + dy), (x + w, y + h)],
                         closed=True, fc=shade(color, 0.74), ec=INK, lw=lw, zorder=z))
    ax.add_patch(Rectangle((x, y), w, h, fc=color, ec=INK, lw=lw, zorder=z + 1))
    return x + w + dx          # right-most drawn x, handy for arrows


def stack(ax, cx, cy, w, h, d, color, n=3, gap=1.1, lw=0.9, z=4):
    """A short stack of slabs, used for the encoder pyramid."""
    for i in range(n - 1, -1, -1):
        slab(ax, cx + i * gap, cy - i * gap * 0.55, w, h, d, color, lw=lw, z=z + i)
    return cx + w / 2.0 + d + (n - 1) * gap


def panel(ax, x, y, w, h, fc, ec, title=None, tsize=16, tcol=None, ls="-",
          sub=None, ssize=12):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=1.0,rounding_size=2.6",
                                fc=fc, ec=ec, lw=1.8, ls=ls, zorder=1))
    if title:
        ax.text(x + w / 2, y + h - 2.0, title, ha="center", va="top",
                fontsize=tsize, fontweight="bold", color=tcol or ec, zorder=6)
    if sub:
        ax.text(x + w / 2, y + h - 2.0 - tsize * 0.46, sub, ha="center", va="top",
                fontsize=ssize, style="italic", fontweight="bold",
                color="#5b6472", zorder=6)


def rbox(ax, cx, cy, w, h, fc, ec, txt, fs=12.5, fw="bold", tc=INK, z=7, lw=1.6):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.35,rounding_size=1.3",
                                fc=fc, ec=ec, lw=lw, zorder=z))
    ax.text(cx, cy, txt, ha="center", va="center", fontsize=fs,
            fontweight=fw, color=tc, zorder=z + 1)


def arrow(ax, p, q, color=INK, lw=1.6, ls="-", rad=0.0, z=6, ms=13):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms,
                                 lw=lw, color=color, linestyle=ls, zorder=z,
                                 connectionstyle="arc3,rad=%.3f" % rad,
                                 shrinkA=1.5, shrinkB=1.5))


def label(ax, cx, cy, txt, fs=12, color=INK, weight="bold", va="bottom",
          ha="center", style="normal", z=8):
    ax.text(cx, cy, txt, ha=ha, va=va, fontsize=fs, color=color,
            fontweight=weight, style=style, zorder=z)


def photo(ax, path, cx, cy, zoom, ec=INK, lw=1.6):
    if not os.path.exists(path):
        return False
    ab = AnnotationBbox(OffsetImage(mpimg.imread(path), zoom=zoom), (cx, cy),
                        frameon=True, zorder=7,
                        bboxprops=dict(edgecolor=ec, lw=lw))
    ax.add_artist(ab)
    return True


def legend_row(ax, items, x0, y, dx, cube_w=4.6, cube_h=4.2, fs=13, d=1.8):
    """items = [(colour, label), ...] laid out left to right from x0."""
    for n, (col, lab) in enumerate(items):
        cx = x0 + n * dx
        slab(ax, cx, y, cube_w, cube_h, d, col, lw=0.9)
        ax.text(cx + cube_w / 2 + d + 2.2, y, lab, ha="left", va="center",
                fontsize=fs, fontweight="bold", zorder=8)
