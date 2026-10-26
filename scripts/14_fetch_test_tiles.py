"""Download Google z19 tiles covering a set of NAIP quads (default: the held-out test quads) for tank-detector deployment."""
import os, sys, json, math, rasterio, geopandas as gpd
from shapely.geometry import box
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(__file__)); from webtiles import tile, fxy; from common import tile_split
Z = 19
QUADS = sys.argv[1].split(",") if len(sys.argv) > 1 else None   # e.g. 3109806,3109807 (default: the test quads)
OUTR = sys.argv[2] if len(sys.argv) > 2 else "data/tank_chips/test_tile_range.json"
tl = [t for t in tile_split()["train"] + tile_split()["val"] if os.path.basename(t).split("_")[2] in QUADS] if QUADS else tile_split()["test"]
test = gpd.GeoSeries([box(*rasterio.open(t).bounds) for t in tl], crs=26914).union_all()
W, S, E, N = gpd.GeoSeries([test], crs=26914).to_crs(4326).total_bounds
x0, y0 = map(int, fxy(W, N, Z)); x1, y1 = map(int, fxy(E, S, Z))
xy = [(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]
json.dump(dict(z=Z, x0=x0, y0=y0, x1=x1, y1=y1), open(OUTR, "w"))
print(len(xy), "tiles", flush=True)
with ThreadPoolExecutor(24) as ex:
    for i, _ in enumerate(ex.map(lambda t: tile("google", t[0], t[1], Z), xy)):
        if i % 10000 == 0: print(i, flush=True)
print("done")
