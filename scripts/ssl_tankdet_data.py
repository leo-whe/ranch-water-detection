"""Training crops for the SSL dense round-tank detector (Google z19 ~0.25 m, 384x384).
Positives  : z20-verified water facilities (data/manual/tank_type_labels.csv, cls=water)
Negatives  : z20-verified bales / hay-ring feeders / other look-alikes + active-learning rejects + random background
Mask       : 1 = tank disk, 0 = background, 255 = ignore (disk+margin around any *unverified* current-system tank
             detection, so unlabeled tanks inside a crop are not taught as background). Training area only.
"""
import os, sys, json, numpy as np, pandas as pd, geopandas as gpd, rasterio
from shapely.geometry import box
sys.path.insert(0, os.path.dirname(__file__)); from webtiles import tile, fxy, gsd; from common import tile_split
from PIL import Image
Z, C = 19, 384; rng = np.random.default_rng(0); OUT = "data/tankdet"; os.makedirs(OUT, exist_ok=True)
test = gpd.GeoSeries([box(*rasterio.open(t).bounds) for t in tile_split()["test"]], crs=26914).to_crs(4326).union_all().buffer(0.003)
def gxy(lon, lat): fx, fy = fxy(lon, lat, Z); return fx * 256, fy * 256
def mosaic(gx0, gy0):
    tx0, ty0 = gx0 // 256, gy0 // 256; M = Image.new("RGB", (3 * 256, 3 * 256))
    for i in range(3):
        for j in range(3): M.paste(tile("google", tx0 + i, ty0 + j, Z), (i * 256, j * 256))
    ox, oy = gx0 - tx0 * 256, gy0 - ty0 * 256; return np.asarray(M.crop((ox, oy, ox + C, oy + C)))
lab = pd.read_csv(os.environ.get("LABELS", "data/manual/tank_type_labels.csv"))
pts = gpd.GeoDataFrame(lab, geometry=gpd.points_from_xy(lab.lon, lab.lat), crs=4326); pts = pts[~pts.within(test)].reset_index(drop=True)
# radius (z19 px) of verified water objects: from YOLO box json where available, else 9 px (~4.6 m diameter)
B = np.array(json.load(open("data/manual/tank_boxes_global_z19.json")))
P = np.array([gxy(r.lon, r.lat) for r in pts.itertuples()])
rad = []
for (x, y) in P:
    d = np.hypot(B[:, 0] - x, B[:, 1] - y); rad.append(float(B[d.argmin(), 2]) if d.min() < 6 else 9.0)
pts["gx"], pts["gy"], pts["r"] = P[:, 0], P[:, 1], np.clip(rad, 4, 30)
water = pts[pts.cls == "water"]; negs = pts[pts.cls != "water"]
# unverified current-system detections -> ignore regions
cand = gpd.read_file("outputs/vector/round_tanks_county_v3_typed.gpkg"); cand = cand[(cand.tank_type == "water") & (cand.p_water > 0.5)]
CG = np.array([gxy(r.lon, r.lat) for r in cand.itertuples()]); CR = (cand.diam_m.values / gsd(31.7, Z) / 2)
def make(gx0, gy0, name, split):
    img = mosaic(int(gx0), int(gy0)); m = np.zeros((C, C), np.uint8); yy, xx = np.mgrid[0:C, 0:C]
    near = (np.abs(CG[:, 0] - gx0 - C / 2) < C) & (np.abs(CG[:, 1] - gy0 - C / 2) < C)
    for (x, y), r in zip(CG[near], CR[near]): m[(xx - (x - gx0)) ** 2 + (yy - (y - gy0)) ** 2 <= (r + 6) ** 2] = 255
    sel = (np.abs(water.gx - gx0 - C / 2) < C) & (np.abs(water.gy - gy0 - C / 2) < C)
    for r_ in water[sel].itertuples():
        d2 = (xx - (r_.gx - gx0)) ** 2 + (yy - (r_.gy - gy0)) ** 2
        m[d2 <= (r_.r + 6) ** 2] = 255; m[d2 <= r_.r ** 2] = 1          # 6-px ignore ring around each verified tank
    seln = (np.abs(negs.gx - gx0 - C / 2) < C) & (np.abs(negs.gy - gy0 - C / 2) < C)
    for r_ in negs[seln].itertuples():                                 # verified look-alikes are explicit background
        m[(xx - (r_.gx - gx0)) ** 2 + (yy - (r_.gy - gy0)) ** 2 <= (r_.r + 6) ** 2] = 0
    np.save(f"{OUT}/{split}_{name}.npy", np.concatenate([img.transpose(2, 0, 1), m[None]], 0))
# spatial-ish val split: 15% of verified objects
idx = rng.permutation(len(pts)); val_ids = set(idx[: int(0.15 * len(pts))])
n = 0
for i, r in pts.iterrows():
    split = "val" if i in val_ids else "train"
    for a in range(4 if (r.cls == "water" and split == "train") else 1):
        ox, oy = rng.integers(48, C - 48, 2); make(r.gx - ox, r.gy - oy, f"{r.cls}{i}_{a}", split); n += 1
R = json.load(open("data/tank_chips/county_tile_range.json")); k = 0
while k < 400:   # random background in the training area
    x, y = rng.integers(R["x0"], R["x1"]) * 256, rng.integers(R["y0"], R["y1"]) * 256
    n_ = 2 ** Z * 256; lon = x / n_ * 360 - 180; lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * y / n_))))
    if test.contains(gpd.points_from_xy([lon], [lat])[0]): continue
    make(x, y, f"bg{k}", "val" if k % 7 == 0 else "train"); k += 1; n += 1
print(n, "crops;", len(water), "water objects;", len(negs), "look-alike negatives")
