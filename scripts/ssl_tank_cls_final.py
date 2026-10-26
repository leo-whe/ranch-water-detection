"""Train the SSL ViT-L tank-type classifier (water / hay_ring / bale / other) on all z20 labels and score detections.
args: --init dinov3sat|ckpt  --labels csv  --score in.gpkg --out out.gpkg  [--cv]  (--cv: 5-fold CV report on the label set)"""
import os, sys, json, argparse, numpy as np, pandas as pd, geopandas as gpd, torch, torch.nn as nn, torch.nn.functional as Fn
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(__file__)); from webtiles import chip; from ssl_backbone import build_backbone, forward_tokens, normalize
ap = argparse.ArgumentParser(); ap.add_argument("--init", default="dinov3sat"); ap.add_argument("--labels", default="data/manual/tank_type_labels.csv")
ap.add_argument("--score", default=None); ap.add_argument("--out", default=None); ap.add_argument("--cv", action="store_true")
ap.add_argument("--save", default="runs/tank_type_vit.pt"); ap.add_argument("--load", default=None); ap.add_argument("--gpu", default="0"); ap.add_argument("--epochs", type=int, default=30)
args = ap.parse_args(); dev = f"cuda:{args.gpu}"; CLS = ["water", "hay_ring", "bale", "other"]
def get(lon, lat): return chip("google", lon, lat, 20, 128)
lab = pd.read_csv(args.labels); lab = lab[lab.cls.isin(CLS)].reset_index(drop=True)
if args.load: lab = lab.iloc[:1]   # scoring only
with ThreadPoolExecutor(16) as ex: X = torch.tensor(np.stack(list(ex.map(lambda r: get(r.lon, r.lat), lab.itertuples())))).permute(0, 3, 1, 2).float() / 255
y = lab.cls.map({c: i for i, c in enumerate(CLS)}).values; print(lab.cls.value_counts().to_dict(), flush=True)
class VitCls(nn.Module):
    def __init__(self):
        super().__init__(); self.bb = build_backbone(3, pretrained=True, drop_path=0.1)
        if args.init != "dinov3sat": self.bb.load_state_dict(torch.load(args.init, map_location="cpu")["teacher_bb"])
        self.head = nn.Linear(2 * self.bb.embed_dim, 4)
    def forward(self, xb):
        c, p = forward_tokens(self.bb, normalize(Fn.interpolate(xb, size=224, mode="bilinear"), 3)); return self.head(torch.cat([c, p.mean(1)], 1))
def aug(xb):
    xb = torch.rot90(xb, np.random.randint(4), (2, 3)); xb = xb.flip(3) if np.random.rand() < .5 else xb
    xb = torch.roll(xb, tuple(int(v) for v in np.random.randint(-6, 7, 2)), (2, 3))
    return (xb * torch.empty(len(xb), 1, 1, 1, device=xb.device).uniform_(.85, 1.15)).clamp(0, 1)
def fit(idx):
    m = VitCls().to(dev); groups = {}
    for n, p in m.named_parameters():
        lr = 1e-3 if n.startswith("head") else (2e-5 * 0.85 ** (24 - int(n.split(".")[2])) if n.startswith("bb.blocks.") else (2e-5 if n.startswith("bb.norm") else 2e-5 * 0.85 ** 25))
        groups.setdefault(lr, []).append(p)
    opt = torch.optim.AdamW([dict(params=v, lr=k) for k, v in groups.items()], weight_decay=0.05)
    w = torch.tensor(1 / np.bincount(y[idx], minlength=4).clip(1), dtype=torch.float32, device=dev); w = w / w.sum() * 4
    steps = args.epochs * max(1, len(idx) // 32); sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in opt.param_groups], total_steps=steps, pct_start=0.1)
    for s in range(steps):
        m.train(); bi = np.random.choice(idx, 32)
        with torch.autocast("cuda", dtype=torch.bfloat16): out = m(aug(X[bi].to(dev)))
        loss = Fn.cross_entropy(out.float(), torch.tensor(y[bi], device=dev), weight=w); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    return m.eval()
@torch.no_grad()
def predict(m, Xs):
    P = []
    for b in range(0, len(Xs), 64):
        xb = Xs[b:b + 64].to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16): P.append((sum(torch.softmax(m(torch.rot90(xb, k, (2, 3))).float(), 1) for k in range(4)) / 4).cpu().numpy())
    return np.concatenate(P)
torch.manual_seed(0); np.random.seed(0)
if args.cv:
    rng = np.random.default_rng(0); folds = np.zeros(len(y), int)
    for c in range(4): ix = np.where(y == c)[0]; folds[rng.permutation(ix)] = np.arange(len(ix)) % 5
    oof = np.zeros((len(y), 4))
    for k in range(5): oof[folds == k] = predict(fit(np.where(folds != k)[0]), X[folds == k])
    pred = oof.argmax(1); wat, pw = y == 0, pred == 0; P = (wat & pw).sum() / pw.sum(); R = (wat & pw).sum() / wat.sum()
    print(f"CV acc {(pred == y).mean():.3f} water P {P:.3f} R {R:.3f} F1 {2 * P * R / (P + R):.3f}", flush=True)
if args.load: m = VitCls().to(dev); m.load_state_dict(torch.load(args.load, map_location='cpu')); m.eval()
else: m = fit(np.arange(len(y))); torch.save(m.state_dict(), args.save)
if args.score:
    d = gpd.read_file(args.score); Ps = []
    for b in range(0, len(d), 2000):
        sub = d.iloc[b:b + 2000]
        with ThreadPoolExecutor(24) as ex: Xs = torch.tensor(np.stack(list(ex.map(lambda r: get(r.lon, r.lat), sub.itertuples())))).permute(0, 3, 1, 2).float() / 255
        Ps.append(predict(m, Xs))
    Pm = np.concatenate(Ps); d[[f"p_{c}" for c in CLS]] = Pm; d["tank_type"] = [CLS[i] for i in Pm.argmax(1)]
    d.to_file(args.out); print(d.tank_type.value_counts().to_dict())
