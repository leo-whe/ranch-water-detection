"""PCA visualisation of self-supervised patch features on NAIP windows: DINOv3-sat (RGB) vs domain-SSL (RGB+NIR).
Also reports how separable pond water is in each feature space (linear probe on patch features vs weak labels).
args: ssl_ckpt out.png [gpu]"""
import os, sys, glob, numpy as np, torch, rasterio, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.font_manager as fm
fm.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"); plt.rcParams["font.family"] = "Noto Sans CJK SC"
from rasterio.windows import Window
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..')); from ssl_backbone import build_backbone, forward_tokens, normalize; from common import tile_split
ck, out = sys.argv[1], sys.argv[2]; dev = f"cuda:{sys.argv[3] if len(sys.argv) > 3 else 0}"
base = build_backbone(3).to(dev).eval(); ssl = build_backbone(4).to(dev).eval()
ssl.load_state_dict(torch.load(ck, map_location="cpu")["teacher_bb"])
rng = np.random.default_rng(5); wins = []
for t in tile_split()["test"]:
    s = rasterio.open(t); l = rasterio.open(t.replace("naip", "labels")).read(1, out_shape=(s.height // 16, s.width // 16), resampling=rasterio.enums.Resampling.nearest)
    r, c = np.nonzero(l == 1)
    for k in rng.choice(len(r), min(6, len(r)), replace=False):
        r0, c0 = int(np.clip(r[k] * 16 - 224, 0, s.height - 448)), int(np.clip(c[k] * 16 - 224, 0, s.width - 448))
        a = s.read(window=Window(c0, r0, 448, 448)); y = rasterio.open(t.replace("naip", "labels")).read(1, window=Window(c0, r0, 448, 448))
        wins.append((a, y))
def feats(m, a, C):
    x = normalize(torch.from_numpy(a[:C]).float().div(255)[None].to(dev), C)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16): _, p = forward_tokens(m, x)
    return p[0].float().cpu().numpy()                                       # (28*28, 1024)
def pca_rgb(F):
    Fc = F - F.mean(0); U, S, Vt = np.linalg.svd(Fc, full_matrices=False); Z = Fc @ Vt[:3].T
    Z = (Z - np.percentile(Z, 2, 0)) / (np.percentile(Z, 98, 0) - np.percentile(Z, 2, 0) + 1e-6); return np.clip(Z, 0, 1)
FB, FS, Y = [], [], []
for a, y in wins:
    FB.append(feats(base, a, 3)); FS.append(feats(ssl, a, 4))
    yy = y.reshape(28, 16, 28, 16).transpose(0, 2, 1, 3).reshape(784, 256); Y.append(np.where((yy == 255).mean(1) > 0.5, -1, (yy == 1).mean(1) > 0.3).astype(int))
# patch-level linear separability of water (logistic regression, half windows train / half test)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
res = {}
for nm, Fl in (("dinov3_sat_rgb", FB), ("domain_ssl_rgbn", FS)):
    h = len(Fl) // 2; Xtr, Ytr = np.concatenate(Fl[:h]), np.concatenate(Y[:h]); Xte, Yte = np.concatenate(Fl[h:]), np.concatenate(Y[h:])
    k1, k2 = Ytr >= 0, Yte >= 0; clf = LogisticRegression(max_iter=2000, C=0.1).fit(Xtr[k1], Ytr[k1])
    res[nm] = average_precision_score(Yte[k2], clf.predict_proba(Xte[k2])[:, 1])
print("patch-level water AP (linear probe):", {k: round(v, 3) for k, v in res.items()})
n = 4; fig, ax = plt.subplots(n, 4, figsize=(10, 2.6 * n))
for i in range(n):
    a, y = wins[i * 3]
    ax[i, 0].imshow(a[:3].transpose(1, 2, 0)); ax[i, 1].imshow(a[[3, 0, 1]].transpose(1, 2, 0))
    ax[i, 2].imshow(pca_rgb(FB[i * 3]).reshape(28, 28, 3), interpolation="nearest"); ax[i, 3].imshow(pca_rgb(FS[i * 3]).reshape(28, 28, 3), interpolation="nearest")
for a_, t_ in zip(ax[0], ["NAIP true colour", "NIR false colour", "DINOv3-sat feature PCA", "Domain SSL (RGB+NIR) feature PCA"]): a_.set_title(t_, fontsize=9)
for a_ in ax.flat: a_.axis("off")
plt.tight_layout(); plt.savefig(out, dpi=130); open(out.replace(".png", ".txt"), "w").write(str(res))
