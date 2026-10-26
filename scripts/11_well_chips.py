"""Google z19 (~0.25 m) chips around TWDB stock wells + Hough-circle proposals for manual box labelling."""
import os, sys, json, numpy as np, pandas as pd, geopandas as gpd, cv2
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(__file__)); from webtiles import chip
OUT = "data/tank_chips/wells"; os.makedirs(OUT, exist_ok=True)
SIZE = 512
def circles(img):
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (5, 5), 1.2)
    c = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1, minDist=10, param1=70, param2=17, minRadius=5, maxRadius=30)
    return [] if c is None else [tuple(map(int, v)) for v in c[0]]
def run(r):
    fn = f"{OUT}/w{r.Index:04d}.jpg"
    img = chip("google", r.lon, r.lat, 19, SIZE)
    cv2.imwrite(fn, cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    return dict(id=int(r.Index), file=fn, lon=r.lon, lat=r.lat, circles=circles(img))
if __name__ == "__main__":
    w = gpd.read_file("data/vector/twdb_wells_hamilton.gpkg"); w = w[w.use == "Stock"].reset_index(drop=True)
    w["lon"], w["lat"] = w.geometry.x, w.geometry.y
    with ThreadPoolExecutor(16) as ex: res = list(ex.map(run, w.itertuples()))
    json.dump(res, open(f"{OUT}/index.json", "w")); print(len(res), "chips;", sum(len(r["circles"]) for r in res), "circle proposals")
