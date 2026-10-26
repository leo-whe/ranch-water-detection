"""Pond segmentation with a self-supervised ViT-L backbone (DINOv3-sat, optionally domain-SSL on NAIP) + light decoder.

Decoder: multi-level ViT features (blocks 6/12/18/24, stride 16) fused and progressively upsampled with skips from a
small full-resolution CNN detail branch (ponds can be smaller than one 9.6 m ViT patch).
Same weak labels / sampler / loss / schedule length as the U-Net baseline (scripts/04_train.py) for a fair comparison.
"""
import os, sys, argparse, random, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn, rasterio
from rasterio.windows import Window
import geopandas as gpd, rasterio.features as RF, pandas as pd
from shapely.geometry import box
sys.path.insert(0, os.path.dirname(__file__))
from common import tile_split
from ssl_backbone import build_backbone, forward_tokens, normalize

ap = argparse.ArgumentParser()
ap.add_argument("--init", default="dinov3sat", help="dinov3sat | path to ssl ckpt.pt")
ap.add_argument("--bands", type=int, default=4); ap.add_argument("--iters", type=int, default=8000)
ap.add_argument("--bs", type=int, default=8); ap.add_argument("--size", type=int, default=512)
ap.add_argument("--frac", type=float, default=1.0, help="fraction of training tiles (label-efficiency study)")
ap.add_argument("--out", required=True); ap.add_argument("--gpu", default="0"); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--vit_scale", type=int, default=1, help="2 = upsample input 2x for the ViT (better small ponds)")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True); dev = f"cuda:{args.gpu}"
torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)

from ssl_models import ViTSeg, prep, LAYERS

# -------------------------------------------------------------------------------------------- data
ROADS = gpd.read_file("zip://data/vector/tl_2023_48193_roads.zip").to_crs(26914).geometry
BLDG = gpd.read_file("data/vector/ms_buildings.gpkg").to_crs(26914).geometry
def lbl_path(t): return t.replace("data/naip", "data/labels")

class Patches(torch.utils.data.IterableDataset):
    """Identical sampling to 04_train.py: 45% positive-centred, 25% hard-negative-centred, 30% random."""
    def __init__(self, tiles, train=True, n=10**9, seed=0):
        self.tiles, self.train, self.n, self.seed = tiles, train, n, seed; self.pos, self.neg = {}, {}
        for t in tiles:
            with rasterio.open(lbl_path(t)) as s:
                l = s.read(1, out_shape=(s.height // 8, s.width // 8), resampling=rasterio.enums.Resampling.nearest)
            with rasterio.open(t) as s:
                a = s.read(out_shape=(4, s.height // 8, s.width // 8)).astype(np.float32); tr = s.transform * s.transform.scale(8, 8); tb = box(*s.bounds)
            wl = (a[3] < 0.8 * a[0]) & (a[3] < 0.8 * a[1]) & (a[:3].max(0) > 0)
            gs = [g for g in pd.concat([ROADS[ROADS.intersects(tb)].buffer(6), BLDG[BLDG.intersects(tb)]]) if not g.is_empty]
            if gs: wl |= RF.rasterize([(g, 1) for g in gs], out_shape=wl.shape, transform=tr).astype(bool)
            r, c = np.nonzero(l == 1); self.pos[t] = np.stack([r * 8, c * 8], 1)
            r, c = np.nonzero((l == 0) & wl); self.neg[t] = np.stack([r * 8, c * 8], 1)
    def __iter__(self):
        wi = torch.utils.data.get_worker_info(); rng = np.random.default_rng((wi.id if wi else 0) + (random.randint(0, 10**6) if self.train else self.seed))
        S = args.size
        for _ in range(self.n):
            t = self.tiles[rng.integers(len(self.tiles))]
            with rasterio.open(t) as s, rasterio.open(lbl_path(t)) as sl:
                u = rng.random(); idx = self.pos[t] if u < 0.45 else self.neg[t] if u < 0.7 else []
                if len(idx):
                    r, c = idx[rng.integers(len(idx))]
                    r0 = int(np.clip(r - rng.integers(S), 0, s.height - S)); c0 = int(np.clip(c - rng.integers(S), 0, s.width - S))
                else:
                    r0 = int(rng.integers(0, s.height - S)); c0 = int(rng.integers(0, s.width - S))
                w = Window(c0, r0, S, S); x = s.read(window=w); y = sl.read(1, window=w)
            if self.train:
                k = rng.integers(4); x = np.rot90(x, k, (1, 2)); y = np.rot90(y, k)
                if rng.random() < 0.5: x = x[:, :, ::-1]; y = y[:, ::-1]
                x = np.clip(x.astype(np.float32) * rng.uniform(0.8, 1.2, (4, 1, 1)) + rng.uniform(-15, 15, (4, 1, 1)), 0, 255)
            yield torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(torch.uint8), torch.from_numpy(np.ascontiguousarray(y)).long()

def loss_fn(logit, y):
    valid = (y != 255).float(); t = (y == 1).float(); logit = logit[:, 0]
    bce = (Fn.binary_cross_entropy_with_logits(logit, t, reduction="none", pos_weight=torch.tensor(3.0, device=logit.device)) * valid).sum() / valid.sum().clamp(min=1)
    p = torch.sigmoid(logit) * valid
    return bce + 1 - (2 * (p * t).sum() + 1) / (p.sum() + (t * valid).sum() + 1)

if __name__ == "__main__":
    sp = tile_split(); train_tiles = sp["train"]
    if args.frac < 1:
        rs = np.random.default_rng(123); train_tiles = sorted(rs.choice(train_tiles, max(1, int(round(args.frac * len(train_tiles)))), replace=False).tolist())
    json.dump(dict(vars(args), train_tiles=train_tiles), open(f"{args.out}/config.json", "w"), indent=1)
    tr = torch.utils.data.DataLoader(Patches(train_tiles), batch_size=args.bs, num_workers=10, persistent_workers=True)
    va = list(iter(Patches(sp["val"], train=False, n=400, seed=0)))
    model = ViTSeg(args.bands, args.init, args.vit_scale).to(dev)
    def lid(n):
        if n.startswith("bb.blocks."): return int(n.split(".")[2]) + 1
        return 0 if n.startswith("bb.") and not n.startswith("bb.norm") else 25
    groups = {}
    for n, p in model.named_parameters():
        lr = 3e-4 if not n.startswith("bb.") else 3e-5 * 0.9 ** (25 - lid(n))
        if n == "bb.patch_embed.proj.weight": lr = 1.5e-4                       # lets NIR filters grow if they start at zero
        groups.setdefault(lr, []).append(p)
    opt = torch.optim.AdamW([dict(params=v, lr=k) for k, v in groups.items()], weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in opt.param_groups], total_steps=args.iters, pct_start=0.05)
    best, it_ = -1, iter(tr); log = open(f"{args.out}/log.txt", "w")
    for step in range(1, args.iters + 1):
        model.train(); x, y = next(it_); x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16): out = model(prep(x, args.bands))
        loss = loss_fn(out.float(), y); opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if step % 400 == 0:
            model.eval(); I = U = 0
            with torch.no_grad():
                for j in range(0, len(va), 16):
                    xb = torch.stack([v[0] for v in va[j:j + 16]]).to(dev); yb = torch.stack([v[1] for v in va[j:j + 16]]).to(dev)
                    with torch.autocast("cuda", dtype=torch.bfloat16): p = model(prep(xb, args.bands))[:, 0].float() > 0
                    v = yb != 255; t = yb == 1; I += (p & t & v).sum().item(); U += ((p | t) & v).sum().item()
            iou = I / max(U, 1); msg = f"step {step} loss {loss.item():.4f} val_IoU {iou:.4f}"; print(msg, flush=True); log.write(msg + "\n"); log.flush()
            if iou > best: best = iou; torch.save(model.state_dict(), f"{args.out}/best.pt")
    print("best", best, flush=True); log.write(f"best {best}\n")
