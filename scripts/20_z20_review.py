"""Render numbered Google z20 (~0.13 m) chips (20 m wide) of detections for 4-way review:
W = water tank / trough (water visible, or metal/poly storage tank), H = hay-ring feeder site, B = round hay bale, O = other.
args: dets.gpkg  out_prefix  n  seed  [min_conf]"""
import os, sys, numpy as np, geopandas as gpd
from PIL import Image, ImageDraw
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(__file__)); from webtiles import chip
g = gpd.read_file(sys.argv[1]); out = sys.argv[2]; n = int(sys.argv[3]); seed = int(sys.argv[4]); mc = float(sys.argv[5]) if len(sys.argv) > 5 else 0.2
s = g[g.conf >= mc].sample(min(n, int((g.conf >= mc).sum())), random_state=seed).reset_index(drop=True)
s.to_file(out + ".gpkg")
with ThreadPoolExecutor(16) as ex: ims = list(ex.map(lambda r: chip("google", r.lon, r.lat, 20, 160), s.itertuples()))
for k in range(0, len(ims), 24):
    G = Image.new("RGB", (6 * 214, 4 * 214), (255, 255, 255))
    for i, a in enumerate(ims[k:k + 24]):
        im = Image.fromarray(a).resize((212, 212)); d = ImageDraw.Draw(im); d.rectangle([0, 0, 30, 13], fill=(0, 0, 0)); d.text((3, 1), str(k + i), fill=(255, 255, 0))
        G.paste(im, ((i % 6) * 214, (i // 6) * 214))
    G.save(f"{out}_{k:03d}.jpg")
print(len(s))
