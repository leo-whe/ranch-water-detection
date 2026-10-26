"""Second stage: classify YOLO round-object candidates on Google z20 chips into
water (tank/trough) / hay_ring / bale / other. Trained on manually reviewed training-area candidates.
args: labels.csv (lon,lat,cls)  dets_to_score.gpkg  out.gpkg"""
import os, sys, numpy as np, pandas as pd, geopandas as gpd, torch, timm, torch.nn.functional as Fn
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(__file__)); from webtiles import chip
torch.manual_seed(0); np.random.seed(0); dev = "cuda:1"; CLS = ["water", "hay_ring", "bale", "other"]; P = 128
def get(lon, lat): return chip("google", lon, lat, 20, P)
def norm(a): return (a.astype(np.float32) / 255 - 0.45) / 0.2
lab = pd.read_csv(sys.argv[1]); lab = lab[lab.cls.isin(CLS)].reset_index(drop=True)
with ThreadPoolExecutor(16) as ex: X = np.stack(list(ex.map(lambda r: get(r.lon, r.lat), lab.itertuples())))
X = norm(X).transpose(0, 3, 1, 2); y = lab.cls.map({c: i for i, c in enumerate(CLS)}).values
print(lab.cls.value_counts().to_dict())
def aug(x):
    x = np.rot90(x, np.random.randint(4), (1, 2))
    if np.random.rand() < .5: x = x[:, :, ::-1]
    x = np.roll(x, tuple(np.random.randint(-6, 7, 2)), (1, 2))
    return np.ascontiguousarray(x * np.random.uniform(.85, 1.15) + np.random.normal(0, .08, (3, 1, 1)))
def fit(idx, epochs=30):
    m = timm.create_model("resnet18", pretrained=True, num_classes=len(CLS)).to(dev); opt = torch.optim.AdamW(m.parameters(), 3e-4, weight_decay=1e-3)
    w = torch.tensor(1 / np.bincount(y[idx], minlength=len(CLS)).clip(1), dtype=torch.float32, device=dev); w = w / w.sum() * len(CLS)
    for ep in range(epochs):
        m.train(); perm = np.random.permutation(idx)
        for b in range(0, len(perm), 32):
            bi = perm[b:b + 32]; xb = torch.tensor(np.stack([aug(X[i]) for i in bi])).float().to(dev)
            loss = Fn.cross_entropy(m(Fn.interpolate(xb, size=224, mode="bilinear")), torch.tensor(y[bi]).to(dev), weight=w)
            opt.zero_grad(); loss.backward(); opt.step()
    return m
@torch.no_grad()
def predict(m, Xs):
    m.eval(); out = []
    for b in range(0, len(Xs), 128):
        xb = Fn.interpolate(torch.tensor(Xs[b:b + 128]).float().to(dev), size=224, mode="bilinear")
        out.append(sum(torch.softmax(m(torch.rot90(xb, k, (2, 3))), 1) for k in range(4)).cpu().numpy() / 4)
    return np.concatenate(out)
# 5-fold CV
folds = np.arange(len(y)) % 5; np.random.shuffle(folds); oof = np.zeros((len(y), len(CLS)))
for k in range(5): oof[folds == k] = predict(fit(np.where(folds != k)[0]), X[folds == k])
pred = oof.argmax(1); cm = pd.crosstab(pd.Series([CLS[i] for i in y], name="true"), pd.Series([CLS[i] for i in pred], name="pred"))
print("CV accuracy", round((pred == y).mean(), 3)); print(cm)
wat = y == 0; pw = pred == 0
print(f"CV water: precision {(wat & pw).sum() / max(pw.sum(), 1):.3f} recall {(wat & pw).sum() / max(wat.sum(), 1):.3f}")
lab[[f"p_{c}" for c in CLS]] = oof; lab.to_csv("outputs/eval/tank_type_classifier_cv.csv", index=False)
cm.to_csv("outputs/eval/tank_type_classifier_confusion.csv")
m = fit(np.arange(len(y))); torch.save(m.state_dict(), "runs/tank_type_cls.pt")
if len(sys.argv) > 3:
    d = gpd.read_file(sys.argv[2]); Ps = []
    for b in range(0, len(d), 2000):
        sub = d.iloc[b:b + 2000]
        with ThreadPoolExecutor(24) as ex: Xs = np.stack(list(ex.map(lambda r: get(r.lon, r.lat), sub.itertuples())))
        Ps.append(predict(m, norm(Xs).transpose(0, 3, 1, 2))); print(b, flush=True)
    Pm = np.concatenate(Ps); d[[f"p_{c}" for c in CLS]] = Pm; d["type"] = [CLS[i] for i in Pm.argmax(1)]
    d.to_file(sys.argv[3]); print(d.type.value_counts().to_dict())
