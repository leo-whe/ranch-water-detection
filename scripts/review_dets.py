"""Montage of Google z19 chips (32 m) around detections, box drawn; for manual precision review."""
import sys, os, math, numpy as np, geopandas as gpd
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(__file__)); from webtiles import chip, gsd
def montage(g, out, start=0, cols=8):
    G = Image.new("RGB", (cols * 130, ((len(g) + cols - 1) // cols) * 130), (255, 255, 255))
    for i, r in enumerate(g.itertuples()):
        im = Image.fromarray(chip("google", r.lon, r.lat, 19, 128)).resize((128, 128)); d = ImageDraw.Draw(im)
        h = r.diam_m / gsd(r.lat, 19) / 2 + 3
        d.rectangle([64 - h, 64 - h, 64 + h, 64 + h], outline=(255, 0, 255), width=1)
        d.rectangle([0, 0, 52, 11], fill=(0, 0, 0)); d.text((1, 0), f"{start + i} {r.conf:.2f}", fill=(255, 255, 0))
        G.paste(im, ((i % cols) * 130, (i // cols) * 130))
    G.save(out)
