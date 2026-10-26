"""Overlay candidate spectral water rules on chips around NWI ponds (and random background)."""
import sys, glob, numpy as np, geopandas as gpd, rasterio, rasterio.windows, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from shapely.geometry import box
def rules(a):
    a = a.astype(np.float32); r, g, b, n = a
    ndwi = (g - n) / (g + n + 1e-6)
    return {"ndwi>0.05": ndwi > 0.05,
            "n<0.8r&n<0.8g": (n < 0.8 * r) & (n < 0.8 * g),
            "ndwi>0&n<b": (ndwi > 0) & (n < b)}
tiles = sorted(glob.glob("data/naip/2022/*.tif"))
nwi = gpd.read_file("data/vector/nwi_hamilton.gpkg").to_crs(26914)
ponds = nwi[nwi.WETLAND_TYPE == "Freshwater Pond"]
rng = np.random.default_rng(int(sys.argv[1]))
chips = []
for t in rng.permutation(tiles):
    s = rasterio.open(t)
    sub = ponds[ponds.centroid.within(box(*s.bounds).buffer(-100))]
    for _, p in sub.sample(min(2, len(sub)), random_state=1).iterrows():
        c = p.geometry.centroid
        chips.append(s.read(window=rasterio.windows.from_bounds(c.x-75, c.y-75, c.x+75, c.y+75, s.transform), boundless=True))
    if len(chips) >= 8: break
K = list(rules(chips[0]).keys())
fig, ax = plt.subplots(len(chips), 2 + len(K), figsize=(3 * (2 + len(K)), 3 * len(chips)))
for i, a in enumerate(chips):
    ax[i, 0].imshow(a[:3].transpose(1, 2, 0)); ax[i, 1].imshow(a[[3, 0, 1]].transpose(1, 2, 0))
    for j, (k, m) in enumerate(rules(a).items()):
        o = a[:3].transpose(1, 2, 0).copy(); o[m] = [255, 0, 0]
        ax[i, 2 + j].imshow(o); ax[i, 2 + j].set_title(k, fontsize=8)
for x in ax.flat: x.axis("off")
plt.tight_layout(); plt.savefig(f"viz/rules_{sys.argv[1]}.jpg", dpi=55)
