"""Train a U-Net (4-band NAIP RGBN, or RGB-only) for stock pond / water tank segmentation on weak labels."""
import os, sys, argparse, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn, rasterio
from rasterio.windows import Window
import segmentation_models_pytorch as smp, geopandas as gpd, rasterio.features as RF
from shapely.geometry import box
sys.path.insert(0, os.path.dirname(__file__)); from common import tile_split, norm

ap = argparse.ArgumentParser()
ap.add_argument("--bands", type=int, default=4); ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--iters", type=int, default=500); ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--size", type=int, default=512); ap.add_argument("--encoder", default="efficientnet-b4")
ap.add_argument("--out", default="runs/unet_rgbn"); ap.add_argument("--gpu", default="0")
ap.add_argument("--init", default=None, help="checkpoint to fine-tune from")
ap.add_argument("--frac", type=float, default=1.0, help="fraction of training tiles (same selection as ssl_seg_train.py)")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True); dev = f"cuda:{args.gpu}"
torch.manual_seed(0); random.seed(0); np.random.seed(0)

import pandas as pd
ROADS = gpd.read_file("zip://data/vector/tl_2023_48193_roads.zip").to_crs(26914).geometry
BLDG = gpd.read_file("data/vector/ms_buildings.gpkg").to_crs(26914).geometry
def lbl_path(t): return t.replace("data/naip", "data/labels")

class Patches(torch.utils.data.IterableDataset):
    """Random patches: 45% centred on positive pixels, 25% on hard negatives (water-like pixels labelled 0:
    roads, tilled fields, roofs), 30% uniformly random."""
    def __init__(self, tiles, train=True, n=10**9):
        self.tiles, self.train, self.n = tiles, train, n
        self.pos, self.neg = {}, {}
        for t in tiles:  # sparse positive / hard-negative index (subsampled /8)
            with rasterio.open(lbl_path(t)) as s:
                l = s.read(1, out_shape=(s.height // 8, s.width // 8), resampling=rasterio.enums.Resampling.nearest)
            with rasterio.open(t) as s:
                a = s.read(out_shape=(4, s.height // 8, s.width // 8)).astype(np.float32)
            wl = (a[3] < 0.8 * a[0]) & (a[3] < 0.8 * a[1]) & (a[:3].max(0) > 0)
            with rasterio.open(t) as s:  # asphalt roads and roofs are dark but often not 'water-like' -> add explicitly
                tr = s.transform * s.transform.scale(8, 8); tb = box(*s.bounds)
            gs = [g for g in pd.concat([ROADS[ROADS.intersects(tb)].buffer(6), BLDG[BLDG.intersects(tb)]]) if not g.is_empty]
            if gs: wl |= RF.rasterize([(g, 1) for g in gs], out_shape=wl.shape, transform=tr).astype(bool)
            r, c = np.nonzero(l == 1); self.pos[t] = np.stack([r * 8, c * 8], 1)
            r, c = np.nonzero((l == 0) & wl); self.neg[t] = np.stack([r * 8, c * 8], 1)
    def __iter__(self):
        wi = torch.utils.data.get_worker_info(); rng = np.random.default_rng((wi.id if wi else 0) + random.randint(0, 10**6))
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
                x = x.astype(np.float32) * rng.uniform(0.8, 1.2, (4, 1, 1)) + rng.uniform(-15, 15, (4, 1, 1))  # radiometric jitter
                x = np.clip(x, 0, 255)
            x = norm(x)[: args.bands]
            yield torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y)).long()

def loss_fn(logit, y):
    valid = (y != 255).float(); t = (y == 1).float(); logit = logit[:, 0]
    bce = (Fn.binary_cross_entropy_with_logits(logit, t, reduction="none", pos_weight=torch.tensor(3.0, device=logit.device)) * valid).sum() / valid.sum().clamp(min=1)
    p = torch.sigmoid(logit) * valid
    dice = 1 - (2 * (p * t).sum() + 1) / (p.sum() + (t * valid).sum() + 1)
    return bce + dice

sp = tile_split()
train_tiles = sp["train"]
if args.frac < 1:
    rs = np.random.default_rng(123); train_tiles = sorted(rs.choice(train_tiles, max(1, int(round(args.frac * len(train_tiles)))), replace=False).tolist())
tr = torch.utils.data.DataLoader(Patches(train_tiles), batch_size=args.bs, num_workers=12, persistent_workers=True)
va_ds = Patches(sp["val"], train=False, n=400); va_ds_items = list(iter(va_ds))  # fixed val set
model = smp.Unet(args.encoder, encoder_weights="imagenet", in_channels=args.bands, classes=1).to(dev)
if args.init: model.load_state_dict(torch.load(args.init, map_location="cpu"))
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=args.epochs * args.iters, pct_start=0.05)
scaler = torch.amp.GradScaler()
best = -1; it = iter(tr)
for ep in range(args.epochs):
    model.train(); tl = 0
    for i in range(args.iters):
        x, y = next(it); x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16): out = model(x)
        loss = loss_fn(out.float(), y); opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); tl += loss.item()
    model.eval(); I = U = 0
    with torch.no_grad():
        for j in range(0, len(va_ds_items), 16):
            xb = torch.stack([v[0] for v in va_ds_items[j:j + 16]]).to(dev); yb = torch.stack([v[1] for v in va_ds_items[j:j + 16]]).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16): p = model(xb)[:, 0].float() > 0
            v = yb != 255; t = yb == 1; I += (p & t & v).sum().item(); U += ((p | t) & v).sum().item()
    iou = I / max(U, 1)
    print(f"ep {ep} loss {tl / args.iters:.4f} val_IoU {iou:.4f}", flush=True)
    if iou > best: best = iou; torch.save(model.state_dict(), f"{args.out}/best.pt")
    torch.save(model.state_dict(), f"{args.out}/last.pt")
print("best", best)
