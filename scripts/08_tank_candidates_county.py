"""County-wide round-tank candidates (tree-masked LoG blobs), excluding building footprints and road corridors."""
import os, sys, glob, numpy as np, pandas as pd, geopandas as gpd, rasterio
from rasterio.windows import Window
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(__file__)); from tank_blobs import blobs_filtered
C, PAD = 2048, 16
def run(t):
    s = rasterio.open(t); rows = []
    for r0 in range(0, s.height, C):
        for c0 in range(0, s.width, C):
            w = Window(c0 - PAD, r0 - PAD, C + 2 * PAD, C + 2 * PAD); a = s.read(window=w, boundless=True)
            if a[:3].max() == 0: continue
            tr = s.window_transform(w)
            for (y, x, rad, pol, con, nd) in blobs_filtered(a):
                if not (PAD <= y < C + PAD and PAD <= x < C + PAD): continue
                X, Y = tr * (x + 0.5, y + 0.5)
                rows.append(dict(tile=os.path.basename(t), x=X, y=Y, rad_m=rad * 0.6, pol=pol, contrast=con, ndvi=nd))
    print(t, len(rows), flush=True); return rows
if __name__ == "__main__":
    tiles = sorted(glob.glob("data/naip/2022/*.tif"))
    with Pool(16) as p: rows = sum(p.map(run, tiles), [])
    g = gpd.GeoDataFrame(pd.DataFrame(rows), geometry=gpd.points_from_xy([r["x"] for r in rows], [r["y"] for r in rows]), crs=26914)
    B = gpd.read_file("data/vector/ms_buildings.gpkg").to_crs(26914); R = gpd.read_file("zip://data/vector/tl_2023_48193_roads.zip").to_crs(26914)
    g = g[~g.index.isin(gpd.sjoin(g, B.assign(geometry=B.buffer(3)), predicate="within").index)]
    g = g[~g.index.isin(gpd.sjoin(g, R.assign(geometry=R.buffer(12)), predicate="within").index)]
    g = g.drop_duplicates(subset=["x", "y"]).reset_index(drop=True)  # tile overlaps
    g.to_file("data/vector/tank_candidates_2022.gpkg"); print(len(g))
