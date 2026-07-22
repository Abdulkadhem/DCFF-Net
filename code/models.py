"""
DCFF-Net: Dual-Cue Feature Fusion Network for change detection (pure CNN).
Also includes a FC-Siam-Diff baseline for fair comparison.

Design (see docs/04_method_design.md):
  Siamese ImageNet-pretrained ResNet encoder -> per-scale dual-cue fusion
  (|F1-F2| difference cue  +  concat cue) -> CBAM -> ASPP (deepest) ->
  U-Net decoder with deep supervision -> change logits.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


# ---------------- building blocks ----------------
class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)


class CBAM(nn.Module):
    """Convolutional Block Attention Module (channel + spatial), pure CNN."""
    def __init__(self, c, r=16):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(c, max(c // r, 8)), nn.ReLU(inplace=True), nn.Linear(max(c // r, 8), c))
        self.spatial = nn.Conv2d(2, 1, 7, padding=3, bias=False)
    def forward(self, x):
        b, c, h, w = x.shape
        avg = self.mlp(F.adaptive_avg_pool2d(x, 1).view(b, c))
        mx = self.mlp(F.adaptive_max_pool2d(x, 1).view(b, c))
        ca = torch.sigmoid(avg + mx).view(b, c, 1, 1)
        x = x * ca
        sa = torch.sigmoid(self.spatial(torch.cat(
            [x.mean(1, keepdim=True), x.max(1, keepdim=True)[0]], dim=1)))
        return x * sa


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling — long-range context with CNN only.
    Rates kept small (safe at low spatial resolution, e.g. 8x8)."""
    def __init__(self, cin, cout, rates=(1, 2, 4)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv2d(cin, cout, 3, padding=r, dilation=r, bias=False),
                          nn.BatchNorm2d(cout), nn.ReLU(inplace=True)) for r in rates])
        self.gpool = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                   nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))
        self.project = nn.Sequential(nn.Conv2d(cout * (len(rates) + 1), cout, 1, bias=False),
                                     nn.BatchNorm2d(cout), nn.ReLU(inplace=True))
    def forward(self, x):
        feats = [b(x) for b in self.branches]
        g = F.interpolate(self.gpool(x), size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.project(torch.cat(feats + [g], dim=1))


class DualCueFusion(nn.Module):
    """Combine difference cue |F1-F2| and concatenation cue [F1,F2] -> CBAM.
    mode in {dual, diff, conc} for ablation; use_cbam toggles attention."""
    def __init__(self, cin, cout, mode="dual", use_cbam=True):
        super().__init__()
        self.mode = mode
        self.conc = (nn.Sequential(nn.Conv2d(2 * cin, cin, 1, bias=False), nn.BatchNorm2d(cin),
                                   nn.ReLU(inplace=True)) if mode in ("dual", "conc") else None)
        in_ch = 2 * cin if mode == "dual" else cin
        self.fuse = ConvBlock(in_ch, cout)
        self.att = CBAM(cout) if use_cbam else nn.Identity()
    def forward(self, f1, f2):
        if self.mode == "diff":
            x = torch.abs(f1 - f2)
        elif self.mode == "conc":
            x = self.conc(torch.cat([f1, f2], dim=1))
        else:
            x = torch.cat([torch.abs(f1 - f2), self.conc(torch.cat([f1, f2], dim=1))], dim=1)
        return self.att(self.fuse(x))


# ---------------- Siamese ResNet encoder ----------------
def _load_rs_weights(net, backbone):
    """Overwrite an ImageNet-initialised torchvision ResNet with remote-sensing
    self-supervised weights (torchgeo, Sentinel-2 RGB MoCo). Key names match
    (conv1/bn1/layer1..4) so a non-strict load transfers the encoder."""
    import torchgeo.models as tgm
    if backbone == "resnet18":
        src = tgm.resnet18(weights=tgm.ResNet18_Weights.SENTINEL2_RGB_MOCO)
    elif backbone == "resnet50":
        src = tgm.resnet50(weights=tgm.ResNet50_Weights.SENTINEL2_RGB_MOCO)
    else:
        raise ValueError(f"no RS weights for {backbone}")
    sd = {k: v for k, v in src.state_dict().items() if not k.startswith("fc.")}
    tgt = net.state_dict()
    ok = {k: v for k, v in sd.items() if k in tgt and tgt[k].shape == v.shape}
    net.load_state_dict(ok, strict=False)
    print(f"[RS-pretrain] transferred {len(ok)}/{len(tgt)} tensors from torchgeo", flush=True)
    return net


class ResNetEncoder(nn.Module):
    def __init__(self, backbone="resnet18", pretrained=True, rs_pretrain=False):
        super().__init__()
        weights = "IMAGENET1K_V1" if pretrained else None
        net = getattr(torchvision.models, backbone)(weights=weights)
        if rs_pretrain:
            net = _load_rs_weights(net, backbone)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)  # /4
        self.layer1, self.layer2, self.layer3, self.layer4 = net.layer1, net.layer2, net.layer3, net.layer4
        self.chs = [64, 128, 256, 512] if backbone in ("resnet18", "resnet34") else [256, 512, 1024, 2048]
    def forward(self, x):
        x = self.stem(x)
        c1 = self.layer1(x)   # /4
        c2 = self.layer2(c1)  # /8
        c3 = self.layer3(c2)  # /16
        c4 = self.layer4(c3)  # /32
        return [c1, c2, c3, c4]


# ---------------- DCFF-Net ----------------
class DCFFNet(nn.Module):
    def __init__(self, backbone="resnet18", pretrained=True, dec=64, deep_sup=True,
                 fusion="dual", use_cbam=True, use_aspp=True, rs_pretrain=False):
        super().__init__()
        self.enc = ResNetEncoder(backbone, pretrained, rs_pretrain=rs_pretrain)
        chs = self.enc.chs
        self.fuse = nn.ModuleList([DualCueFusion(c, dec, mode=fusion, use_cbam=use_cbam) for c in chs])
        self.aspp = ASPP(dec, dec) if use_aspp else ConvBlock(dec, dec)   # ablation: replace context
        # decoder: from deepest to shallowest
        self.up3 = ConvBlock(dec + dec, dec)
        self.up2 = ConvBlock(dec + dec, dec)
        self.up1 = ConvBlock(dec + dec, dec)
        self.head = nn.Conv2d(dec, 1, 1)
        self.deep_sup = deep_sup
        if deep_sup:
            self.aux = nn.ModuleList([nn.Conv2d(dec, 1, 1) for _ in range(3)])

    def _up(self, x, ref): return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, xa, xb):
        fa, fb = self.enc(xa), self.enc(xb)
        p = [self.fuse[i](fa[i], fb[i]) for i in range(4)]   # fused per scale
        d4 = self.aspp(p[3])
        d3 = self.up3(torch.cat([self._up(d4, p[2]), p[2]], 1))
        d2 = self.up2(torch.cat([self._up(d3, p[1]), p[1]], 1))
        d1 = self.up1(torch.cat([self._up(d2, p[0]), p[0]], 1))
        out = F.interpolate(self.head(d1), size=xa.shape[-2:], mode="bilinear", align_corners=False)
        if self.deep_sup and self.training:
            aux = [F.interpolate(self.aux[i](d), size=xa.shape[-2:], mode="bilinear", align_corners=False)
                   for i, d in enumerate([d3, d2, d1])]
            return out, aux
        return out


# ---------------- baseline: FC-Siam-Diff ----------------
class FCSiamDiff(nn.Module):
    """Classic Siamese U-Net with difference skip fusion (baseline)."""
    def __init__(self, base=16):
        super().__init__()
        def enc(ci, co): return nn.Sequential(ConvBlock(ci, co), nn.MaxPool2d(2))
        self.e1 = ConvBlock(3, base); self.p1 = nn.MaxPool2d(2)
        self.e2 = ConvBlock(base, base*2); self.p2 = nn.MaxPool2d(2)
        self.e3 = ConvBlock(base*2, base*4); self.p3 = nn.MaxPool2d(2)
        self.e4 = ConvBlock(base*4, base*8)
        self.d3 = ConvBlock(base*8 + base*4, base*4)
        self.d2 = ConvBlock(base*4 + base*2, base*2)
        self.d1 = ConvBlock(base*2 + base, base)
        self.head = nn.Conv2d(base, 1, 1)
    def _feats(self, x):
        s1 = self.e1(x); s2 = self.e2(self.p1(s1)); s3 = self.e3(self.p2(s2)); s4 = self.e4(self.p3(s3))
        return s1, s2, s3, s4
    def forward(self, xa, xb):
        a = self._feats(xa); b = self._feats(xb)
        d = [torch.abs(a[i] - b[i]) for i in range(4)]
        x = F.interpolate(d[3], size=d[2].shape[-2:], mode="bilinear", align_corners=False)
        x = self.d3(torch.cat([x, d[2]], 1))
        x = F.interpolate(x, size=d[1].shape[-2:], mode="bilinear", align_corners=False)
        x = self.d2(torch.cat([x, d[1]], 1))
        x = F.interpolate(x, size=d[0].shape[-2:], mode="bilinear", align_corners=False)
        x = self.d1(torch.cat([x, d[0]], 1))
        return self.head(x)


def build_model(name="dcff", **kw):
    if name == "dcff": return DCFFNet(**kw)
    if name == "fcsiam": return FCSiamDiff()
    raise ValueError(name)


if __name__ == "__main__":
    m = DCFFNet(pretrained=False)
    xa = torch.randn(2, 3, 256, 256); xb = torch.randn(2, 3, 256, 256)
    m.train(); out, aux = m(xa, xb)
    print("train out:", out.shape, "aux:", [a.shape for a in aux])
    m.eval(); print("eval out:", m(xa, xb).shape)
    print("params(M):", sum(p.numel() for p in m.parameters())/1e6)
