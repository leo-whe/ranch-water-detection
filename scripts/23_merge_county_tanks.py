"""Merge row-sharded county detections (blocks overlap by 64 px, so re-run NMS across shards) and clip to the county."""
import sys, glob, numpy as np, pandas as pd, geopandas as gpd, torch, torchvision
SRC = sys.argv[1] if len(sys.argv) > 1 else "outputs/vector/county_shards"; DST = sys.argv[2] if len(sys.argv) > 2 else "outputs/vector/round_tanks_county_v3_raw.gpkg"
g = pd.concat([gpd.read_file(f) for f in sorted(glob.glob(f"{SRC}/s*.gpkg"))], ignore_index=True)
half = (g.diam_m / 0.254 / 2).values  # box half-size in z19 px (approx)
B = np.stack([g.gx - half, g.gy - half, g.gx + half, g.gy + half], 1)
keep = torchvision.ops.nms(torch.tensor(B, dtype=torch.float64), torch.tensor(g.conf.values, dtype=torch.float64), 0.3).numpy()
g = gpd.GeoDataFrame(g.iloc[keep].reset_index(drop=True), crs=4326)
cty = gpd.read_file("data/vector/hamilton_county.geojson").to_crs(4326).geometry.iloc[0]
g = g[g.within(cty)].reset_index(drop=True)
g.to_file(DST); print(len(g), "candidates in county;", int((g.conf >= 0.2).sum()), "with conf>=0.2")
