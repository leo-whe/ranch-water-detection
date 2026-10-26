"""Deploy the dense SSL tank detector over a z19 tile range: overlapping 384-px windows -> stitched probability map per
block -> connected components -> tank points (lon, lat, conf=max prob, diam_m). args: weights tile_range.json out.gpkg [shard] [gpu]"""
import os, sys, json, math, numpy as np, pandas as pd, geopandas as gpd, torch
from PIL import Image
from scipy import ndimage as ndi
sys.path.insert(0, os.path.dirname(__file__)); from ssl_models import ViTSeg, prep; from webtiles import gsd
W, R_, OUT = sys.argv[1], json.load(open(sys.argv[2])), sys.argv[3]
SK, SN = map(int, (sys.argv[4] if len(sys.argv) > 4 else "0/1").split("/")); dev = f"cuda:{sys.argv[5] if len(sys.argv) > 5 else 0}"
Z = R_["z"]; S, STRIDE, BLK = 384, 320, 8
m = ViTSeg(3, "dinov3sat"); m.load_state_dict(torch.load(W, map_location="cpu")); m.to(dev).eval()
def px2ll(gx, gy):
    n = 2 ** Z * 256; return gx / n * 360 - 180, math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * gy / n))))
rows = []; blocks = [(bx, by) for by in range(R_["y0"], R_["y1"] + 1, BLK) for bx in range(R_["x0"], R_["x1"] + 1, BLK)][SK::SN]
for bi, (bx, by) in enumerate(blocks):
    # block of BLK x BLK tiles plus a 1-tile margin on each side
    M = np.zeros(((BLK + 2) * 256, (BLK + 2) * 256, 3), np.uint8); have = False
    for i in range(BLK + 2):
        for j in range(BLK + 2):
            fn = f"data/webtiles/google/{Z}/{bx - 1 + i}_{by - 1 + j}.jpg"
            if os.path.exists(fn): M[j * 256:(j + 1) * 256, i * 256:(i + 1) * 256] = np.asarray(Image.open(fn).convert("RGB")); have = True
    if not have: continue
    H = M.shape[0]; prob = np.zeros((H, H), np.float32)
    starts = sorted(set(list(range(0, H - S + 1, STRIDE)) + [H - S]))
    wins = [(r, c) for r in starts for c in starts if M[r:r + S, c:c + S].mean() > 5]
    with torch.no_grad():
        for k in range(0, len(wins), 32):
            ws = wins[k:k + 32]; x = torch.from_numpy(np.stack([M[r:r + S, c:c + S].transpose(2, 0, 1) for r, c in ws])).float().to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16): p = torch.sigmoid(m(prep(x, 3)).float())[:, 0]
            p = p.cpu().numpy()
            for (r, c), pi in zip(ws, p):
                e = 32  # discard window borders (except at block edge, covered by the 1-tile margin)
                prob[r + e:r + S - e, c + e:c + S - e] = np.maximum(prob[r + e:r + S - e, c + e:c + S - e], pi[e:S - e, e:S - e])
    lab, n = ndi.label(prob > 0.3)
    if n:
        idx = np.arange(1, n + 1); mx = ndi.maximum(prob, lab, idx); area = ndi.sum(np.ones_like(prob), lab, idx); com = ndi.center_of_mass(prob, lab, idx)
        for k in range(n):
            cy, cx = com[k]
            if not (256 <= cy < (BLK + 1) * 256 and 256 <= cx < (BLK + 1) * 256): continue   # centre must be inside the core block
            if area[k] < 12: continue
            gx, gy = (bx - 1) * 256 + cx, (by - 1) * 256 + cy; lon, lat = px2ll(gx, gy)
            rows.append(dict(lon=lon, lat=lat, conf=float(mx[k]), diam_m=2 * math.sqrt(area[k] / math.pi) * gsd(lat, Z), gx=gx, gy=gy))
    if bi % 50 == 0: print(bi, len(blocks), len(rows), flush=True)
    if bi % 500 == 0 and rows: pd.DataFrame(rows).to_csv(OUT + ".partial.csv", index=False)
g = gpd.GeoDataFrame(pd.DataFrame(rows), geometry=gpd.points_from_xy([r["lon"] for r in rows], [r["lat"] for r in rows]), crs=4326)
if not rows: sys.exit(0)
g.to_file(OUT); print(len(g), "detections;", int((g.conf > 0.5).sum()), "conf>0.5")
