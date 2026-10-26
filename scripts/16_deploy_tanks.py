"""Run the round-tank YOLO over the test-area Google z19 tiles (384-px windows, 64-px overlap) -> points/boxes in lon/lat."""
import os, sys, json, math, numpy as np, pandas as pd, geopandas as gpd, torch, torchvision
from PIL import Image
from ultralytics import YOLO
sys.path.insert(0, os.path.dirname(__file__)); from webtiles import tile, gsd
R = json.load(open(sys.argv[2] if len(sys.argv) > 2 else "data/tank_chips/test_tile_range.json")); Z = R["z"]
OUT = sys.argv[3] if len(sys.argv) > 3 else "outputs/vector/round_tanks_test_area_google.gpkg"
SK, SN = map(int, (sys.argv[4] if len(sys.argv) > 4 else "0/1").split("/")); GPU = int(sys.argv[5]) if len(sys.argv) > 5 else 0  # row sharding for parallel runs
W_, STRIDE, BLK = 384, 320, 6  # window, stride, tiles per mosaic block
model = YOLO(sys.argv[1] if len(sys.argv) > 1 else "runs/yolo/tank_v1/weights/best.pt")
dets = []
def px2ll(gx, gy):
    n = 2 ** Z * 256; lon = gx / n * 360 - 180; lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * gy / n)))); return lon, lat
gx_min, gy_min = R["x0"] * 256, R["y0"] * 256; gx_max, gy_max = (R["x1"] + 1) * 256, (R["y1"] + 1) * 256
blk = BLK * 256
for by in list(range(gy_min, gy_max, blk - 64))[SK::SN]:
    for bx in range(gx_min, gx_max, blk - 64):
        tx0, ty0 = bx // 256, by // 256
        M = Image.new("RGB", (BLK * 256 + 256, BLK * 256 + 256))
        for i in range(BLK + 1):
            for j in range(BLK + 1):
                fn = f"data/webtiles/google/{Z}/{tx0 + i}_{ty0 + j}.jpg"
                if os.path.exists(fn): M.paste(Image.open(fn).convert("RGB"), (i * 256, j * 256))
        A = np.array(M)[by - ty0 * 256: by - ty0 * 256 + blk, bx - tx0 * 256: bx - tx0 * 256 + blk]
        wins, offs = [], []
        def starts(n): return sorted(set(list(range(0, n - W_ + 1, STRIDE)) + [n - W_]))  # always include the last window: no gaps between blocks
        for oy in starts(A.shape[0]):
            for ox in starts(A.shape[1]):
                w = A[oy:oy + W_, ox:ox + W_]
                if w.mean() < 5: continue
                wins.append(w[:, :, ::-1].copy()); offs.append((bx + ox, by + oy))
        if not wins: continue
        for res, (ox, oy) in zip(model.predict(wins, imgsz=640, conf=0.15, device=GPU, verbose=False, batch=64), offs):
            for b, c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
                dets.append([b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy, float(c)])
    print(by, len(dets), flush=True)
if not dets: sys.exit(0)
D = np.array(dets); keep = torchvision.ops.nms(torch.tensor(D[:, :4]), torch.tensor(D[:, 4]), 0.3).numpy(); D = D[keep]
rows = []
for x0, y0, x1, y1, c in D:
    lon, lat = px2ll((x0 + x1) / 2, (y0 + y1) / 2); rows.append(dict(lon=lon, lat=lat, conf=c, diam_m=((x1 - x0) + (y1 - y0)) / 2 * gsd(lat, Z), gx=(x0 + x1) / 2, gy=(y0 + y1) / 2))
g = gpd.GeoDataFrame(pd.DataFrame(rows), geometry=gpd.points_from_xy([r["lon"] for r in rows], [r["lat"] for r in rows]), crs=4326)
os.makedirs("outputs/vector", exist_ok=True); g.to_file(OUT)
print(len(g), "detections;", (g.conf > 0.5).sum(), "conf>0.5")
