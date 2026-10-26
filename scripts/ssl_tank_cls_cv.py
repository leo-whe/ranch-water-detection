"""Round-tank type classification (water / hay_ring / bale / other) on Google z20 chips:
compare backbones under identical 5-fold CV and label budgets.
  resnet18_in   : ImageNet-supervised ResNet18, fine-tuned (current production model)
  dinov3_lin    : DINOv3-sat ViT-L frozen, linear probe on [CLS ; mean patch]
  dinov3_ft     : DINOv3-sat ViT-L fine-tuned (layer-wise lr decay)
  ssl_lin/ssl_ft: same, from our domain-SSL checkpoint (--ssl runs/ssl/google/ckpt.pt)
"""
import os, sys, json, argparse, numpy as np, pandas as pd, torch, timm, torch.nn as nn, torch.nn.functional as Fn
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
from webtiles import chip
from ssl_backbone import build_backbone, forward_tokens, normalize
ap = argparse.ArgumentParser(); ap.add_argument("--methods", default="resnet18_in,dinov3_lin,dinov3_ft")
ap.add_argument("--ssl", default=None); ap.add_argument("--gpu", default="0"); ap.add_argument("--fracs", default="0.25,0.5,1.0")
ap.add_argument("--out", default="outputs/eval/ssl_tank_cls_cv.json")
args = ap.parse_args(); dev = f"cuda:{args.gpu}"
CLS = ["water", "hay_ring", "bale", "other"]
lab = pd.read_csv("data/manual/tank_type_labels.csv"); lab = lab[lab.cls.isin(CLS)].reset_index(drop=True)
cache = "data/tank_chips/z20_label_chips.npy"
if os.path.exists(cache): X = np.load(cache)
else:
    with ThreadPoolExecutor(16) as ex: X = np.stack(list(ex.map(lambda r: chip("google", r.lon, r.lat, 20, 128), lab.itertuples())))
    np.save(cache, X)
y = lab.cls.map({c: i for i, c in enumerate(CLS)}).values
rng = np.random.default_rng(0); folds = np.zeros(len(y), int)
for c in range(len(CLS)): ix = np.where(y == c)[0]; folds[rng.permutation(ix)] = np.arange(len(ix)) % 5     # stratified
Xt = torch.tensor(X).permute(0, 3, 1, 2).float() / 255.0                                                     # (N,3,128,128) [0,1]

def aug(xb):
    k = np.random.randint(4); xb = torch.rot90(xb, k, (2, 3))
    if np.random.rand() < .5: xb = xb.flip(3)
    sh = np.random.randint(-6, 7, 2); xb = torch.roll(xb, tuple(int(v) for v in sh), (2, 3))
    return (xb * torch.empty(len(xb), 1, 1, 1, device=xb.device).uniform_(.85, 1.15)).clamp(0, 1)

def run_cnn(tr, te, epochs=30):
    m = timm.create_model("resnet18", pretrained=True, num_classes=4).to(dev); opt = torch.optim.AdamW(m.parameters(), 3e-4, weight_decay=1e-3)
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    f = lambda xb: m(Fn.interpolate((xb - mean) / std, size=224, mode="bilinear"))
    return train_eval(m, f, tr, te, opt, epochs)

class VitCls(nn.Module):
    def __init__(self, init):
        super().__init__(); self.bb = build_backbone(3, pretrained=True, drop_path=0.1)
        if init: self.bb.load_state_dict(torch.load(init, map_location="cpu")["teacher_bb"])
        self.head = nn.Linear(2 * self.bb.embed_dim, 4)
    def feats(self, xb):
        c, p = forward_tokens(self.bb, normalize(Fn.interpolate(xb, size=224, mode="bilinear"), 3)); return torch.cat([c, p.mean(1)], 1)
    def forward(self, xb): return self.head(self.feats(xb))

def train_eval(m, f, tr, te, opt, epochs):
    w = torch.tensor(1 / np.bincount(y[tr], minlength=4).clip(1), dtype=torch.float32, device=dev); w = w / w.sum() * 4
    steps = epochs * max(1, len(tr) // 32); sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in opt.param_groups], total_steps=steps, pct_start=0.1)
    for s in range(steps):
        m.train(); bi = np.random.choice(tr, 32)
        with torch.autocast("cuda", dtype=torch.bfloat16): out = f(aug(Xt[bi].to(dev)))
        loss = Fn.cross_entropy(out.float(), torch.tensor(y[bi], device=dev), weight=w); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    m.eval(); P = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for b in range(0, len(te), 64):
            xb = Xt[te[b:b + 64]].to(dev); P.append(sum(torch.softmax(f(torch.rot90(xb, k, (2, 3))).float(), 1) for k in range(4)).cpu().numpy() / 4)
    return np.concatenate(P)

def run_vit_ft(tr, te, init, epochs=30):
    m = VitCls(init).to(dev); groups = {}
    for n, p in m.named_parameters():
        if n.startswith("bb.blocks."): lr = 2e-5 * 0.85 ** (24 - int(n.split(".")[2]))
        elif n.startswith("bb."): lr = 2e-5 * 0.85 ** 25 if not n.startswith("bb.norm") else 2e-5
        else: lr = 1e-3
        groups.setdefault(lr, []).append(p)
    opt = torch.optim.AdamW([dict(params=v, lr=k) for k, v in groups.items()], weight_decay=0.05)
    return train_eval(m, m, tr, te, opt, epochs)

_feat_cache = {}
def run_vit_lin(tr, te, init):
    key = init or "dinov3"
    if key not in _feat_cache:
        m = VitCls(init).to(dev).eval(); Fs = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for k in range(4):   # 4 rotations -> average features
                Fs.append(torch.cat([m.feats(torch.rot90(Xt[b:b + 64].to(dev), k, (2, 3))).float().cpu() for b in range(0, len(Xt), 64)]))
        _feat_cache[key] = torch.stack(Fs).mean(0); del m; torch.cuda.empty_cache()
    Fm = _feat_cache[key]; mu, sd = Fm[tr].mean(0), Fm[tr].std(0) + 1e-6; Z = ((Fm - mu) / sd).to(dev)
    lin = nn.Linear(Z.shape[1], 4).to(dev); opt = torch.optim.AdamW(lin.parameters(), 1e-3, weight_decay=1e-2)
    w = torch.tensor(1 / np.bincount(y[tr], minlength=4).clip(1), dtype=torch.float32, device=dev); w = w / w.sum() * 4
    yt = torch.tensor(y, device=dev)
    for _ in range(500): loss = Fn.cross_entropy(lin(Z[tr]), yt[tr], weight=w); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): return torch.softmax(lin(Z[te]), 1).cpu().numpy()

def metrics(P, idx):
    pred = P.argmax(1); t = y[idx]; wat, pw = t == 0, pred == 0
    prec = (wat & pw).sum() / max(pw.sum(), 1); rec = (wat & pw).sum() / max(wat.sum(), 1)
    return dict(acc=float((pred == t).mean()), water_precision=float(prec), water_recall=float(rec), water_f1=float(2 * prec * rec / max(prec + rec, 1e-9)),
                macro_f1=float(np.mean([2 * ((t == c) & (pred == c)).sum() / max((t == c).sum() + (pred == c).sum(), 1) for c in range(4)])))

res = json.load(open(args.out)) if os.path.exists(args.out) else {}
for meth in args.methods.split(","):
    for frac in [float(f) for f in args.fracs.split(",")]:
        key = f"{meth}@{frac}"; torch.manual_seed(0); np.random.seed(0); oof = np.zeros((len(y), 4))
        for k in range(5):
            tr_all = np.where(folds != k)[0]; te = np.where(folds == k)[0]
            r = np.random.default_rng(k); tr = np.sort(r.choice(tr_all, int(round(frac * len(tr_all))), replace=False)) if frac < 1 else tr_all
            if meth == "resnet18_in": oof[te] = run_cnn(tr, te)
            elif meth == "dinov3_lin": oof[te] = run_vit_lin(tr, te, None)
            elif meth == "dinov3_ft": oof[te] = run_vit_ft(tr, te, None)
            elif meth == "ssl_lin": oof[te] = run_vit_lin(tr, te, args.ssl)
            elif meth == "ssl_ft": oof[te] = run_vit_ft(tr, te, args.ssl)
        res[key] = metrics(oof, np.arange(len(y))); print(key, {k: round(v, 3) for k, v in res[key].items()}, flush=True)
        json.dump(res, open(args.out, "w"), indent=1)
