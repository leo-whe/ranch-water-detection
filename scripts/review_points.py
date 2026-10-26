"""Numbered review grid for point candidates: NAIP RGB | NAIP CIR | Esri, 40 m window, tick marks around the candidate."""
import sys, os, numpy as np, rasterio, rasterio.windows as W
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(__file__)); from esri import esri_chip; from review_chips import TILES
from shapely.geometry import Point
from concurrent.futures import ThreadPoolExecutor
def pchip(x, y, rad, half=20, px=128):
    t = next(t for t, tb in TILES.items() if tb.contains(Point(x, y))); s = rasterio.open(t)
    a = s.read(window=W.from_bounds(x - half, y - half, x + half, y + half, s.transform), boundless=True)
    try: e = esri_chip(x - half, y - half, x + half, y + half, res=0.3)
    except Exception: e = np.zeros((8, 8, 3), np.uint8)
    ims = [Image.fromarray(np.ascontiguousarray(z)).resize((px, px), Image.BICUBIC) for z in (a[:3].transpose(1, 2, 0), a[[3, 0, 1]].transpose(1, 2, 0), e)]
    k = px / (2 * half); c = px / 2; r = (rad + 2.5) * k
    for im in ims:
        d = ImageDraw.Draw(im)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            d.line([(c + dx * r, c + dy * r), (c + dx * (r + 8), c + dy * (r + 8))], fill=(255, 0, 255), width=2)
    return ims
def grid(df, out, cols=4, px=128, start=0):
    with ThreadPoolExecutor(16) as ex: chips = list(ex.map(lambda r: pchip(r.x, r.y, r.rad_m, px=px), df.itertuples()))
    rows = (len(chips) + cols - 1) // cols; cw = 3 * px + 10
    G = Image.new("RGB", (cols * cw, rows * (px + 4)), (255, 255, 255))
    for i, ims in enumerate(chips):
        x0, y0 = (i % cols) * cw, (i // cols) * (px + 4)
        for j, im in enumerate(ims): G.paste(im, (x0 + j * px, y0))
        d = ImageDraw.Draw(G); d.rectangle([x0, y0, x0 + 22, y0 + 12], fill=(0, 0, 0)); d.text((x0 + 2, y0), str(start + i), fill=(255, 255, 0))
    G.save(out)
