"""Fetch USFWS NWI wetland polygons (incl. 'Freshwater Pond') over Hamilton County, TX."""
import json, itertools, time, requests, geopandas as gpd
from shapely.geometry import shape
from concurrent.futures import ThreadPoolExecutor

URL = "https://fwspublicservices.wim.usgs.gov/wetlandsmapservice/rest/services/Wetlands/MapServer/0/query"
county = gpd.read_file("data/vector/hamilton_county.geojson")
minx, miny, maxx, maxy = county.total_bounds
step = 0.04
cells = [(x, y, min(x + step, maxx), min(y + step, maxy))
         for x in [minx + i * step for i in range(int((maxx - minx) / step) + 1)]
         for y in [miny + j * step for j in range(int((maxy - miny) / step) + 1)]]
cells = [c for c in cells if county.geometry.iloc[0].intersects(
    gpd.GeoSeries.from_xy([c[0], c[2]], [c[1], c[3]]).union_all().envelope)]

def fetch(c):
    feats, off = [], 0
    while True:
        p = dict(geometry=",".join(map(str, c)), geometryType="esriGeometryEnvelope", inSR=4326,
                 outSR=4326, outFields="ATTRIBUTE,WETLAND_TYPE,ACRES", where="1=1",
                 resultOffset=off, resultRecordCount=500, f="geojson")
        for attempt in range(6):
            try:
                r = requests.get(URL, params=p, timeout=300); d = r.json(); break
            except Exception as e:
                time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError(f"failed {c}")
        fs = d.get("features", [])
        feats += fs
        if len(fs) < 500: return feats
        off += 500

with ThreadPoolExecutor(16) as ex:
    res = list(ex.map(fetch, cells))
feats = list(itertools.chain(*res))
gdf = gpd.GeoDataFrame([f["properties"] for f in feats],
                       geometry=[shape(f["geometry"]) for f in feats], crs=4326)
gdf = gdf.drop_duplicates(subset=["ATTRIBUTE", "ACRES"] + []).reset_index(drop=True) if False else gdf
gdf["wkb"] = gdf.geometry.apply(lambda g: g.wkb)
gdf = gdf.drop_duplicates("wkb").drop(columns="wkb")
gdf.to_file("data/vector/nwi_hamilton.gpkg", driver="GPKG")
print(len(cells), "cells;", len(gdf), "features"); print(gdf.WETLAND_TYPE.value_counts())
