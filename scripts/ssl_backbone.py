"""DINOv3-sat ViT-L backbone helpers for self-supervised (DINO + iBOT + Gram-anchor) training and downstream use.

* build_backbone(in_chans=4): DINOv3 ViT-L/16 pretrained (SSL) on SAT-493M Maxar imagery; for 4-band NAIP the patch
  embedding is widened to RGB+NIR with the NIR filters zero-initialised, so the model starts exactly as the RGB model.
* forward_tokens(m, x, mask): forward with optional iBOT patch masking (masked patches -> learnable mask token).
"""
import torch, torch.nn as nn, timm

NAME = "vit_large_patch16_dinov3.sat493m"
MEAN = {3: (0.430, 0.411, 0.296), 4: (0.430, 0.411, 0.296, 0.55)}   # DINOv3-sat stats; NIR from NAIP county stats
STD = {3: (0.213, 0.156, 0.143), 4: (0.213, 0.156, 0.143, 0.10)}


def build_backbone(in_chans=3, pretrained=True, drop_path=0.0):
    m = timm.create_model(NAME, pretrained=pretrained, num_classes=0, dynamic_img_size=True, drop_path_rate=drop_path)
    if in_chans != 3:
        old = m.patch_embed.proj
        new = nn.Conv2d(in_chans, old.out_channels, old.kernel_size, old.stride, bias=old.bias is not None)
        with torch.no_grad():
            new.weight.zero_(); new.weight[:, :3] = old.weight
            if old.bias is not None: new.bias.copy_(old.bias)
        m.patch_embed.proj = new
    m.mask_token = nn.Parameter(torch.zeros(1, 1, 1, m.embed_dim))
    return m


def normalize(x01, in_chans):
    """x01: float tensor in [0,1], (B,C,H,W)."""
    mean = torch.tensor(MEAN[in_chans], device=x01.device, dtype=x01.dtype).view(1, -1, 1, 1)
    std = torch.tensor(STD[in_chans], device=x01.device, dtype=x01.dtype).view(1, -1, 1, 1)
    return (x01 - mean) / std


def forward_tokens(m, x, mask=None, return_layers=None):
    """Returns (cls (B,D), patch tokens (B,N,D) after final norm[, list of intermediate patch maps]).
    mask: bool (B, h, w) of patches to replace by the mask token (iBOT)."""
    x = m.patch_embed(x)                                   # (B, h, w, D) with dynamic_img_size
    if mask is not None:
        x = torch.where(mask[..., None], m.mask_token.to(x.dtype), x)
    B, h, w, D = x.shape
    x, rope = m._pos_embed(x)
    x = m.norm_pre(x)
    inter = []
    for i, blk in enumerate(m.blocks):
        x = blk(x, rope=rope)
        if return_layers is not None and i in return_layers:
            inter.append(x[:, m.num_prefix_tokens:].transpose(1, 2).reshape(B, D, h, w))
    x = m.norm(x)
    cls, patches = x[:, 0], x[:, m.num_prefix_tokens:]
    return (cls, patches, inter) if return_layers is not None else (cls, patches)
