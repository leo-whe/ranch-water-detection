"""Probability rasters -> water-source instances (polygons + boxes) with shape attributes and a type guess."""
import os, sys, glob, argparse, numpy as np, rasterio, rasterio.features as F, geopandas as gpd, pandas as pd
from scipy import ndimage as ndi
from shapely.geometry import shape
from multiprocessing import Pool
ap = argparse.ArgumentParser(); ap.add_argument("--year", default="2022"); ap.add_argument("--thr", type=float, default=0.5)
ap.add_argument("--min_area", type=float, default=8.0)  # m^2
ap.add_argument("--prob_dir", default="outputs/prob"); ap.add_argument("--tag", default="")
args = ap.parse_args()
BLDG = gpd.read_file("data/vector/ms_buildings.gpkg").to_crs(26914)

def run(p):
    s = rasterio.open(p); pr = s.read(1) / 255.0; m = pr > args.thr
    lab, n = ndi.label(m)
    if n == 0: return []
    idx = np.arange(1, n + 1)
    mean_p = ndi.mean(pr, lab, idx); max_p = ndi.maximum(pr, lab, idx)
    rows = []
    for geom, v in F.shapes(lab.astype(np.int32), mask=m, transform=s.transform):
        g = shape(geom); v = int(v); A = g.area
        if A < args.min_area: continue
        P = g.length; circ = 4 * np.pi * A / P ** 2 if P else 0
        rows.append(dict(geometry=g, tile=os.path.basename(p), area_m2=A, perim_m=P, circularity=circ,
                         mean_prob=float(mean_p[v - 1]), max_prob=float(max_p[v - 1])))
    return rows

if __name__ == "__main__":
    files = sorted(glob.glob(f"{args.prob_dir}/{args.year}/*.tif"))
    with Pool(24) as pool: rows = sum(pool.map(run, files), [])
    g = gpd.GeoDataFrame(rows, crs=26914)
    # merge pieces split across tile seams (NAIP tiles overlap slightly)
    g = g.dissolve().explode(index_parts=False).reset_index(drop=True)[["geometry"]].sjoin(g, how="left", predicate="intersects")
    g = g.groupby(level=0).agg(dict(geometry="first", area_m2="max", mean_prob="max", max_prob="max", tile="first"))
    g = gpd.GeoDataFrame(g, geometry="geometry", crs=26914)
    g["area_m2"] = g.area; g["perim_m"] = g.length; g["circularity"] = 4 * np.pi * g.area / g.length ** 2
    # attribute: inside/next to a building footprint -> likely swimming pool / roof artefact
    near_b = gpd.sjoin(g, BLDG.assign(geometry=BLDG.buffer(15)), how="left", predicate="intersects").index_right.notna()
    g["near_building"] = near_b.groupby(level=0).any()
    # post-filter flag: open-water-looking pixels on asphalt roads / dark roofs are the dominant false positives
    roads = gpd.read_file("zip://data/vector/tl_2023_48193_roads.zip").to_crs(26914)
    for name, mask in (("frac_road", roads.buffer(10)), ("frac_bldg", BLDG.buffer(1))):
        m = gpd.GeoDataFrame(geometry=list(mask.union_all().geoms) if mask.union_all().geom_type == "MultiPolygon" else [mask.union_all()], crs=26914)
        ov = gpd.overlay(g[["geometry"]].reset_index(), m, how="intersection")
        g[name] = (ov.groupby("index").geometry.apply(lambda s: s.area.sum()).reindex(g.index).fillna(0) / g.area).clip(0, 1)
    g["likely_false_positive"] = (g.frac_road > 0.3) | (g.frac_bldg > 0.3)
    g["possible_pool"] = g.near_building & (g.area <= 200)  # backyard swimming pools / stock tanks next to barns
    eq_d = 2 * np.sqrt(g.area / np.pi)
    import shapely
    g["roundness"] = g.area / (np.pi * shapely.minimum_bounding_radius(g.geometry.values) ** 2)  # 1 = disk; robust to pixel stair-steps
    nwi = gpd.read_file("data/vector/nwi_hamilton.gpkg").to_crs(26914)
    riv = nwi[nwi.WETLAND_TYPE == "Riverine"]
    f_riv = gpd.overlay(g[["geometry"]].reset_index(), riv[["geometry"]].assign(geometry=riv.buffer(5)), how="intersection")
    g["frac_riverine"] = (f_riv.groupby("index").geometry.apply(lambda s: s.area.sum()).reindex(g.index).fillna(0) / g.area).clip(0, 1)
    g["near_nwi_pond"] = g.index.isin(gpd.sjoin(g, nwi[nwi.WETLAND_TYPE.isin(["Freshwater Pond", "Lake"])].assign(
        geometry=lambda d: d.buffer(60)), predicate="intersects").index)
    # coarse class: small (<=200 m2, ~16 m diam) & round open water = possible man-made round tank or tiny dug pond; riverine = creek/river pool; else pond
    g["src_class"] = np.select([g.frac_riverine > 0.5, (g.area <= 200) & (g.roundness >= 0.70)],
                               ["creek_or_river", "small_round_water"], "pond")
    g["equiv_diam_m"] = eq_d
    c = g.geometry.centroid.to_crs(4326); g["lon"] = c.x; g["lat"] = c.y
    b = g.geometry.bounds; g[["xmin", "ymin", "xmax", "ymax"]] = b.values
    g = g.sort_values("area_m2", ascending=False).reset_index(drop=True); g["id"] = np.arange(len(g))
    os.makedirs("outputs/vector", exist_ok=True)
    g.to_file(f"outputs/vector/water_sources{args.tag}_{args.year}.gpkg", driver="GPKG")
    g.to_crs(4326).to_file(f"outputs/vector/water_sources{args.tag}_{args.year}.geojson", driver="GeoJSON")
    print(len(g)); print(g.src_class.value_counts()); print("flagged likely FP:", int(g.likely_false_positive.sum())); print(g.area_m2.describe())
