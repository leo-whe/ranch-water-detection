"""Active learning: add reviewed detections (train area) to the YOLO dataset.
args: dets.gpkg  yes_ids(comma)  no_ids(comma)  tag"""
import os, sys, json, numpy as np, geopandas as gpd
sys.path.insert(0, os.path.dirname(__file__))
import importlib.util
spec = importlib.util.spec_from_file_location("bd", os.path.join(os.path.dirname(__file__), "15_build_yolo_dataset.py"))
g = gpd.read_file(sys.argv[1]).sort_values("conf", ascending=False).reset_index(drop=True)
yes = [int(v) for v in sys.argv[2].split(",") if v]; no = [int(v) for v in sys.argv[3].split(",") if v]; tag = sys.argv[4]
from webtiles import gsd, tile, fxy
from PIL import Image
Z, C, ROOT = 19, 384, "data/yolo_tanks"
src = open(os.path.join(os.path.dirname(__file__), "15_build_yolo_dataset.py")).read()
exec(src[src.index("def mosaic"):src.index("pos = pd.read_csv")])  # reuse mosaic()
B = np.array(json.load(open("data/manual/tank_boxes_global_z19.json")), float)
new = np.array([[r.gx, r.gy, r.diam_m / gsd(r.lat, Z) / 2, -1] for r in g.loc[yes].itertuples()], float).reshape(-1, 4)
B = np.unique(np.round(np.vstack([B, new]), 2), axis=0); json.dump(B.tolist(), open("data/manual/tank_boxes_global_z19.json", "w"))
def write(img, gx0, gy0, split, name):
    img.save(f"{ROOT}/images/{split}/{name}.jpg", quality=92); L = []
    for cx, cy, r, _ in B:
        x, y = cx - gx0, cy - gy0; rr = r + 1.5
        if 0 <= x < C and 0 <= y < C: L.append(f"0 {x / C:.5f} {y / C:.5f} {2 * rr / C:.5f} {2 * rr / C:.5f}")
    open(f"{ROOT}/labels/{split}/{name}.txt", "w").write("\n".join(L))
rng = np.random.default_rng(1)
for k, (cx, cy, r, _) in enumerate(new):
    for a in range(2):
        ox, oy = rng.integers(40, C - 40, 2); write(mosaic(int(cx - ox), int(cy - oy), C), int(cx - ox), int(cy - oy), "val" if k % 7 == 0 and a == 0 else "train", f"{tag}_pos{k}_{a}")
for k, r in enumerate(g.loc[no].itertuples()):
    ox, oy = rng.integers(60, C - 60, 2); gx0, gy0 = int(r.gx - ox), int(r.gy - oy)
    write(mosaic(gx0, gy0, C), gx0, gy0, "val" if k % 7 == 0 else "train", f"{tag}_neg{k}")
print(len(new), "new positives,", len(no), "hard negatives; total boxes", len(B))
