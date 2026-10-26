"""Download Google z19 tiles intersecting Hamilton County (for county-wide round-tank detection)."""
import os, sys, json, math, geopandas as gpd
from shapely.geometry import box
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(__file__)); from webtiles import tile, fxy
Z = 19
cty = gpd.read_file("data/vector/hamilton_county.geojson").to_crs(4326).geometry.iloc[0].buffer(0.001)
W, S, E, N = cty.bounds; x0, y0 = map(int, fxy(W, N, Z)); x1, y1 = map(int, fxy(E, S, Z))
def t2ll(x, y):
    n = 2 ** Z; lon = x / n * 360 - 180; lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n)))); return lon, lat
from shapely.prepared import prep
P = prep(cty); xy = []
for x in range(x0, x1 + 1):
    for y in range(y0, y1 + 1):
        a, b = t2ll(x, y); c, d = t2ll(x + 1, y + 1)
        if P.intersects(box(a, d, c, b)): xy.append((x, y))
json.dump(dict(z=Z, x0=x0, y0=y0, x1=x1, y1=y1, n_tiles=len(xy)), open("data/tank_chips/county_tile_range.json", "w"))
print(len(xy), "tiles", flush=True)
with ThreadPoolExecutor(24) as ex:
    for i, _ in enumerate(ex.map(lambda t: tile("google", t[0], t[1], Z), xy)):
        if i % 50000 == 0: print(i, flush=True)
print("done")
