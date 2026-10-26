"""Download NAIP 2022 (0.6 m, RGBN) COG tiles covering Hamilton County, TX from Microsoft Planetary Computer."""
import os, json, requests, geopandas as gpd, planetary_computer as pc
from pystac_client import Client
from concurrent.futures import ThreadPoolExecutor

YEAR = os.environ.get("NAIP_YEAR", "2022")
out = f"data/naip/{YEAR}"; os.makedirs(out, exist_ok=True)
county = gpd.read_file("data/vector/hamilton_county.geojson")
cat = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
items = list(cat.search(collections=["naip"], intersects=county.geometry.iloc[0].__geo_interface__,
                        query={"naip:year": {"eq": YEAR}}).items())
print(len(items), "items")
json.dump([{"id": i.id, "bbox": i.bbox, "datetime": str(i.datetime)} for i in items],
          open(f"{out}/items.json", "w"), indent=1)

def dl(it):
    fn = f"{out}/{it.id}.tif"
    if os.path.exists(fn) and os.path.getsize(fn) > 1e6: return
    with requests.get(it.assets["image"].href, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(fn + ".part", "wb") as f:
            for ch in r.iter_content(1 << 22): f.write(ch)
    os.rename(fn + ".part", fn)

with ThreadPoolExecutor(12) as ex: list(ex.map(dl, items))
print("done")
