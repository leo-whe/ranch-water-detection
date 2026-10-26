"""Multi-year tracking: pond footprint inventory = union of water detections over all years;
per footprint, water-surface area in each NAIP year -> wet/dry status table."""
import os, glob, numpy as np, pandas as pd, geopandas as gpd
YEARS = ["2018", "2020", "2022"]; TAG = os.environ.get("TAG", "")
d = {y: gpd.read_file(f"outputs/vector/water_sources{TAG}_{y}.gpkg") for y in YEARS}
for y in YEARS:  # ponds/tanks only: skip creeks/rivers, pools & road/roof artefacts
    d[y] = d[y][(d[y].area_m2 >= 25) & ~d[y].near_building & ~d[y].likely_false_positive.astype(bool) & (d[y].src_class != "creek_or_river")]
allw = pd.concat([g[["geometry"]] for g in d.values()])
fp = gpd.GeoDataFrame(geometry=list(allw.buffer(1).union_all().geoms), crs=26914)  # merge pieces within ~2 m
fp["chain_like"] = (fp.area / fp.convex_hull.area < 0.25) & (fp.area > 5000)  # strings of creek pools, not one pond
fp["fp_area_m2"] = fp.area
for y in YEARS:
    j = gpd.overlay(fp.reset_index().rename(columns={"index": "fid"}), d[y][["geometry"]], how="intersection")
    fp[f"water_m2_{y}"] = j.groupby("fid").geometry.apply(lambda s: s.area.sum()).reindex(fp.index).fillna(0).values
W = fp[[f"water_m2_{y}" for y in YEARS]].values; mx = W.max(1)
for y in YEARS: fp[f"wet_{y}"] = fp[f"water_m2_{y}"] >= np.maximum(0.1 * mx, 10)
fp["n_years_wet"] = fp[[f"wet_{y}" for y in YEARS]].sum(axis=1)
fp["trend"] = np.select([fp.wet_2018 & ~fp.wet_2022, ~fp.wet_2018 & fp.wet_2022, fp.n_years_wet == 3], ["dried_by_2022", "new_or_refilled", "persistent"], "intermittent")
c = fp.geometry.centroid.to_crs(4326); fp["lon"], fp["lat"] = c.x, c.y
fp = fp.sort_values("fp_area_m2", ascending=False).reset_index(drop=True); fp["pond_id"] = np.arange(len(fp))
fp.to_file(f"outputs/vector/pond_inventory_multiyear{TAG}.gpkg")
fp.drop(columns="geometry").to_csv(f"outputs/pond_inventory_multiyear{TAG}.csv", index=False)
print(len(fp), "footprints"); print(fp.trend.value_counts()); print({y: int(fp[f"wet_{y}"].sum()) for y in YEARS})
print({y: round(fp[f"water_m2_{y}"].sum() / 1e6, 3) for y in YEARS}, "km2 open water in ponds")
