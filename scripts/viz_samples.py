"""Montage of NAIP chips (RGB | CIR) around random NWI ponds, with NWI outline, for visual QA."""
import sys, glob, numpy as np, geopandas as gpd, rasterio, rasterio.features, rasterio.windows, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from shapely.geometry import box
tiles = [t for t in glob.glob("data/naip/2022/*.tif")]
nwi = gpd.read_file("data/vector/nwi_hamilton.gpkg").to_crs(26914)
ponds = nwi[nwi.WETLAND_TYPE == "Freshwater Pond"]
rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
rows = []
for t in rng.permutation(tiles):
    s = rasterio.open(t); tb = box(*s.bounds)
    sub = ponds[ponds.centroid.within(tb.buffer(-100))]
    for _, p in sub.sample(min(3, len(sub)), random_state=0).iterrows():
        c = p.geometry.centroid; half = 75  # 150 m chip
        w = rasterio.windows.from_bounds(c.x-half, c.y-half, c.x+half, c.y+half, s.transform)
        a = s.read(window=w, boundless=True)
        m = rasterio.features.rasterize([(g, 1) for g in ponds.geometry.boundary if g.intersects(box(c.x-half,c.y-half,c.x+half,c.y+half))],
                                        out_shape=a.shape[1:], transform=s.window_transform(w))
        rows.append((a, m, p.ATTRIBUTE, p.ACRES))
    if len(rows) >= 16: break
fig, ax = plt.subplots(4, 8, figsize=(24, 12))
for i, (a, m, attr, ac) in enumerate(rows[:16]):
    rgb = a[:3].transpose(1, 2, 0).copy(); rgb[m > 0] = [255, 0, 255]
    cir = a[[3, 0, 1]].transpose(1, 2, 0)
    ax.flat[2*i].imshow(rgb); ax.flat[2*i].set_title(f"{attr} {ac:.2f}ac", fontsize=8)
    ax.flat[2*i+1].imshow(cir); ax.flat[2*i+1].set_title("CIR", fontsize=8)
for a_ in ax.flat: a_.axis("off")
plt.tight_layout(); plt.savefig(f"viz/nwi_samples_{sys.argv[1] if len(sys.argv)>1 else 0}.jpg", dpi=60)
