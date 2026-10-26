"""Montages of Google z19 chips (24 m) around NAIP tank candidates (train/val area only) for fast yes/no review."""
import os, sys, json, numpy as np, geopandas as gpd, rasterio
from PIL import Image, ImageDraw
from shapely.geometry import box
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(__file__)); from webtiles import chip; from common import tile_split
OUT = "data/tank_chips/review"; os.makedirs(OUT, exist_ok=True)
c = gpd.read_file("outputs/vector/tank_candidates_scored_2022.gpkg")
test = gpd.GeoSeries([box(*rasterio.open(t).bounds) for t in tile_split()["test"]], crs=26914).union_all()
c = c[(c.tank_score > 0.3) & ~c.within(test.buffer(200))].copy()
ll = c.to_crs(4326); c["lon"], c["lat"] = ll.geometry.x, ll.geometry.y
c = c.sample(frac=1, random_state=0).reset_index(drop=True)
c[["x", "y", "lon", "lat", "tank_score", "rad_m", "pol"]].to_csv(f"{OUT}/candidates.csv", index_label="cid")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 640
def one(r): return chip("google", r.lon, r.lat, 19, 96)
for b in range(0, N, 64):
    sub = c.iloc[b:b + 64]
    with ThreadPoolExecutor(16) as ex: ims = list(ex.map(one, sub.itertuples()))
    G = Image.new("RGB", (8 * 130, 8 * 130), (255, 255, 255))
    for i, a in enumerate(ims):
        im = Image.fromarray(a).resize((128, 128)); d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 26, 12], fill=(0, 0, 0)); d.text((1, 0), str(b + i), fill=(255, 255, 0))
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)): d.line([(64 + dx * 34, 64 + dy * 34), (64 + dx * 44, 64 + dy * 44)], fill=(255, 0, 255), width=2)
        G.paste(im, ((i % 8) * 130, (i // 8) * 130))
    G.save(f"{OUT}/rev_{b:04d}.jpg")
print(len(c), "candidates (train/val area, score>0.3)")
