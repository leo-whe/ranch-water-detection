"""Shared dense-prediction model on the (SSL) ViT-L backbone: multi-level ViT features + full-resolution CNN detail branch."""
import torch, torch.nn as nn, torch.nn.functional as Fn
from ssl_backbone import build_backbone, forward_tokens, normalize
LAYERS = (5, 11, 17, 23)

def cbr(i, o, s=1): return nn.Sequential(nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

class ViTSeg(nn.Module):
    def __init__(self, bands, init, vit_scale=1):
        super().__init__()
        self.bands = bands; self.vit_scale = vit_scale   # vit_scale=2: ViT sees 2x-upsampled input (4.8 m tokens) for small ponds
        self.bb = build_backbone(bands, pretrained=True, drop_path=0.1)
        if init != "dinov3sat":
            ck = torch.load(init, map_location="cpu"); sd = ck["teacher_bb"]
            assert ck["in_chans"] == bands, "ssl checkpoint band count mismatch"
            print(self.bb.load_state_dict(sd, strict=True))
        D = self.bb.embed_dim
        self.neck = nn.Sequential(nn.Conv2d(4 * D, 384, 1, bias=False), nn.BatchNorm2d(384), nn.ReLU(inplace=True), cbr(384, 256))
        # detail branch on raw pixels: 1/1, 1/2, 1/4, 1/8
        self.s1 = nn.Sequential(cbr(bands, 32), cbr(32, 32)); self.s2 = cbr(32, 64, 2); self.s4 = cbr(64, 96, 2); self.s8 = cbr(96, 128, 2)
        self.u8 = cbr(256 + 128, 192); self.u4 = cbr(192 + 96, 128); self.u2 = cbr(128 + 64, 64); self.u1 = cbr(64 + 32, 32)
        self.head = nn.Conv2d(32, 1, 1)
    def forward(self, x):
        xv = x if self.vit_scale == 1 else Fn.interpolate(x, scale_factor=self.vit_scale, mode="bilinear", align_corners=False)
        _, _, inter = forward_tokens(self.bb, xv, return_layers=LAYERS)
        f = self.neck(torch.cat(inter, 1))                                       # 1/16
        d1 = self.s1(x); d2 = self.s2(d1); d4 = self.s4(d2); d8 = self.s8(d4)
        up = lambda a, b: Fn.interpolate(a, size=b.shape[-2:], mode="bilinear", align_corners=False)
        f = self.u8(torch.cat([up(f, d8), d8], 1)); f = self.u4(torch.cat([up(f, d4), d4], 1))
        f = self.u2(torch.cat([up(f, d2), d2], 1)); f = self.u1(torch.cat([up(f, d1), d1], 1))
        return self.head(f)

def prep(x_uint8, bands):
    """uint8 (B,4,H,W) tensor -> normalised float for the ViT (RGB or RGBN)."""
    return normalize(x_uint8[:, :bands].float() / 255.0, bands)

