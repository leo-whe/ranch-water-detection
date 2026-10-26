"""YOLO dataset for round tanks on Google z19 (~0.25 m) imagery.
Positives: manual boxes (data/manual/tank_boxes_manual.csv, chip coords in 128-px chips centred on reviewed candidates).
Negatives: reviewed-and-rejected candidates + random background crops in the training area."""
import os, sys, math, random, numpy as np, pandas as pd, geopandas as gpd, rasterio, json
from PIL import Image
from shapely.geometry import box
sys.path.insert(0, os.path.dirname(__file__)); from webtiles import tile, fxy; from common import tile_split
random.seed(0); np.random.seed(0); Z = 19; C = 384; ROOT = "data/yolo_tanks"
def mosaic(gx0, gy0, size):
    tx0, ty0, tx1, ty1 = gx0 // 256, gy0 // 256, (gx0 + size) // 256, (gy0 + size) // 256
    M = Image.new("RGB", ((tx1 - tx0 + 1) * 256, (ty1 - ty0 + 1) * 256))
    for tx in range(tx0, tx1 + 1):
        for ty in range(ty0, ty1 + 1): M.paste(tile("google", tx, ty, Z), ((tx - tx0) * 256, (ty - ty0) * 256))
    return M.crop((gx0 - tx0 * 256, gy0 - ty0 * 256, gx0 - tx0 * 256 + size, gy0 - ty0 * 256 + size))
pos = pd.read_csv("data/manual/tank_pos_for_boxes.csv"); mb = pd.read_csv("data/manual/tank_boxes_manual.csv")
boxes = []  # global z19 pixel coords
for r in mb.itertuples():
    p = pos.loc[r.pid]; fx, fy = fxy(p.lon, p.lat, Z); gx0, gy0 = int(fx * 256 - 64), int(fy * 256 - 64)
    boxes.append((gx0 + r.cx, gy0 + r.cy, r.r, r.pid))
B = np.array(boxes, float); json.dump(B.tolist(), open("data/manual/tank_boxes_global_z19.json", "w"))
pids = sorted(mb.pid.unique()); random.shuffle(pids); val_p = set(pids[: int(0.18 * len(pids))])
for s in ("train", "val"): os.makedirs(f"{ROOT}/images/{s}", exist_ok=True); os.makedirs(f"{ROOT}/labels/{s}", exist_ok=True)
def write(img, gx0, gy0, split, name):
    img.save(f"{ROOT}/images/{split}/{name}.jpg", quality=92); L = []
    for cx, cy, r, _ in B:
        x, y = cx - gx0, cy - gy0; rr = r + 1.5  # small margin around the tank rim
        if 0 <= x < C and 0 <= y < C:
            L.append(f"0 {x / C:.5f} {y / C:.5f} {2 * rr / C:.5f} {2 * rr / C:.5f}")
    open(f"{ROOT}/labels/{split}/{name}.txt", "w").write("\n".join(L))
n = 0
for k, (cx, cy, r, pid) in enumerate(B):
    split = "val" if pid in val_p else "train"
    for a in range(3 if split == "train" else 1):
        ox, oy = np.random.randint(40, C - 40, 2); gx0, gy0 = int(cx - ox), int(cy - oy)
        write(mosaic(gx0, gy0, C), gx0, gy0, split, f"pos{k}_{a}"); n += 1
# negatives: rejected reviewed candidates (shed/trailer/shrub look-alikes)
rej = pd.read_csv("data/manual/tank_bright_review_google.csv"); rej = rej[rej.label == "no"]
for i, r in enumerate(rej.itertuples()):
    fx, fy = fxy(r.lon, r.lat, Z); gx0, gy0 = int(fx * 256 - C / 2 + np.random.randint(-60, 60)), int(fy * 256 - C / 2 + np.random.randint(-60, 60))
    if np.any((np.abs(B[:, 0] - gx0 - C / 2) < C) & (np.abs(B[:, 1] - gy0 - C / 2) < C)): continue
    write(mosaic(gx0, gy0, C), gx0, gy0, "val" if i % 6 == 0 else "train", f"neg{i}"); n += 1
# random background in the training area
tr = gpd.GeoSeries([box(*rasterio.open(t).bounds) for t in tile_split()["train"]], crs=26914).to_crs(4326)
for i in range(250):
    b = tr.iloc[np.random.randint(len(tr))].bounds; lon, lat = np.random.uniform(b[0], b[2]), np.random.uniform(b[1], b[3])
    fx, fy = fxy(lon, lat, Z); gx0, gy0 = int(fx * 256), int(fy * 256)
    write(mosaic(gx0, gy0, C), gx0, gy0, "val" if i % 6 == 0 else "train", f"bg{i}"); n += 1
open(f"{ROOT}/data.yaml", "w").write(f"path: {os.path.abspath(ROOT)}\ntrain: images/train\nval: images/val\nnames:\n  0: round_tank\n")
print(n, "images;", len(B), "boxes;", len(val_p), "val positives groups")
