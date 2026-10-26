"""Second stage for round tanks: tiny CNN that re-scores LoG candidates using 4-band NAIP chips.
Trained on the manually reviewed candidates (+ random unlabeled candidates as presumed negatives)."""
import os, sys, glob, numpy as np, pandas as pd, geopandas as gpd, rasterio, torch, timm, torch.nn.functional as Fn
from rasterio.windows import from_bounds
from sklearn.metrics import roc_auc_score, average_precision_score
from shapely.geometry import box
sys.path.insert(0, os.path.dirname(__file__)); from common import norm
torch.manual_seed(0); np.random.seed(0); dev = "cuda:2"; P = 48; EPOCHS = 15
TB = {t: box(*rasterio.open(t).bounds) for t in glob.glob("data/naip/2022/*.tif")}; OPEN = {}
def chip(x, y):
    from shapely.geometry import Point
    t = next((t for t, b in TB.items() if b.contains(Point(x, y))), None)
    if t is None: return np.zeros((4, P, P), np.float32)  # candidate outside imagery footprint
    if t not in OPEN: OPEN[t] = rasterio.open(t)
    s = OPEN[t]
    a = s.read(window=from_bounds(x - P * 0.3, y - P * 0.3, x + P * 0.3, y + P * 0.3, s.transform), out_shape=(4, P, P), boundless=True)
    return norm(a)
def aug(x):
    k = np.random.randint(4); x = np.rot90(x, k, (1, 2))
    if np.random.rand() < .5: x = x[:, :, ::-1]
    sh = np.random.randint(-3, 4, 2); x = np.roll(x, tuple(sh), (1, 2))
    return np.ascontiguousarray(x * np.random.uniform(.85, 1.15) + np.random.normal(0, .1, (4, 1, 1)))
def model(): return timm.create_model("resnet18", pretrained=True, in_chans=4, num_classes=1).to(dev)
def fit(X, y, w, epochs=40):
    m = model(); opt = torch.optim.AdamW(m.parameters(), 3e-4, weight_decay=1e-3)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    for ep in range(epochs):
        m.train()
        for _ in range(30):
            idx = np.concatenate([np.random.choice(pos, 16), np.random.choice(neg, 48)])
            xb = torch.tensor(np.stack([aug(X[i]) for i in idx])).float().to(dev)
            xb = Fn.interpolate(xb, scale_factor=2, mode="bilinear")
            yb = torch.tensor(y[idx]).float().to(dev); wb = torch.tensor(w[idx]).float().to(dev)
            loss = (Fn.binary_cross_entropy_with_logits(m(xb)[:, 0], yb, reduction="none") * wb).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if os.environ.get("DBG"): print("  ep", ep, "loss", round(loss.item(), 4), flush=True)
    return m
@torch.no_grad()
def predict(m, X, bs=512):
    m.eval(); out = []
    for i in range(0, len(X), bs):
        xb = Fn.interpolate(torch.tensor(X[i:i + bs]).float().to(dev), scale_factor=2, mode="bilinear")
        p = sum(torch.sigmoid(m(torch.rot90(xb, k, (2, 3)))[:, 0]) for k in range(4)) / 4
        out.append(p.cpu().numpy())
    return np.concatenate(out)
def score_tile(m, t, sub):
    """Score all candidates of one tile from a single full-tile read (much faster than per-point window reads)."""
    s = rasterio.open(t); a = norm(s.read()); H, W = a.shape[1:]; h = P // 2
    a = np.pad(a, ((0, 0), (h, h), (h, h)))
    rc = [s.index(x, y) for x, y in zip(sub.x, sub.y)]
    X = np.stack([a[:, r:r + P, c:c + P] if (0 <= r < H and 0 <= c < W) else np.zeros((4, P, P), np.float32) for r, c in rc])
    return predict(m, X)
if __name__ == "__main__":
    lab = pd.read_csv("data/manual/tank_candidate_review.csv", index_col=0); lab = lab[lab.label != "skip"]
    lab["is_tank"] = (lab.label != "no").astype(int); lab["w"] = np.where(lab.label == "probable", 0.6, 1.0)
    cand = gpd.read_file("data/vector/tank_candidates_2022.gpkg")
    unl = cand.sample(1500, random_state=0)
    XL = np.stack([chip(r.x, r.y) for r in lab.itertuples()]); XU = np.stack([chip(r.x, r.y) for r in unl.itertuples()])
    yL, wL = lab.is_tank.values, lab.w.values; yU, wU = np.zeros(len(XU)), np.full(len(XU), 0.5)
    folds = np.zeros(len(lab), int); rng = np.random.default_rng(0)
    for c in (0, 1): ix = np.where(yL == c)[0]; folds[rng.permutation(ix)] = np.arange(len(ix)) % 5
    oof = np.zeros(len(lab))
    for k in range(5):
        tr = folds != k
        mk = fit(np.concatenate([XL[tr], XU]), np.concatenate([yL[tr], yU]), np.concatenate([wL[tr], wU]), epochs=EPOCHS)
        oof[~tr] = predict(mk, XL[~tr]); print(f"fold {k}: n={(~tr).sum()} pos={yL[~tr].sum()} score_std={oof[~tr].std():.3f}", flush=True)
    auc, ap = roc_auc_score(yL, oof), average_precision_score(yL, oof)
    print(f"CV on reviewed candidates: ROC-AUC {auc:.3f}  AP {ap:.3f}  (prevalence {yL.mean():.3f})", flush=True)
    lab["cv_score"] = oof; lab.to_csv("outputs/eval/tank_classifier_cv.csv")
    m = fit(np.concatenate([XL, XU]), np.concatenate([yL, yU]), np.concatenate([wL, wU]), epochs=EPOCHS)
    torch.save(m.state_dict(), "runs/tank_cls.pt")
    cand["tank_score"] = np.nan
    for t, sub in cand.groupby("tile"):
        cand.loc[sub.index, "tank_score"] = score_tile(m, f"data/naip/2022/{t}", sub)
    cand.to_file("outputs/vector/tank_candidates_scored_2022.gpkg")
    print(cand.tank_score.describe(), flush=True); print((cand.tank_score > 0.5).sum(), "candidates > 0.5")
