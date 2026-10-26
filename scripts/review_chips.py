"""Render numbered review montages for a GeoDataFrame of instances: NAIP RGB (+outline) | NAIP CIR | Esri World Imagery."""
import os, sys, glob, numpy as np, rasterio, rasterio.windows as W, rasterio.features as F, geopandas as gpd
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(__file__)); from esri import esri_chip
from shapely.geometry import box
from concurrent.futures import ThreadPoolExecutor
TILES = {t: box(*rasterio.open(t).bounds) for t in glob.glob("data/naip/2022/*.tif")}
def chip(geom, half=None, px=150, with_esri=True):
    c = geom.centroid; b = geom.bounds
    half = half or max(40, 0.8 * max(b[2] - b[0], b[3] - b[1]))
    t = next(t for t, tb in TILES.items() if tb.contains(c))
    s = rasterio.open(t); w = W.from_bounds(c.x - half, c.y - half, c.x + half, c.y + half, s.transform)
    a = s.read(window=w, boundless=True)
    o = F.rasterize([(geom.boundary, 1)], out_shape=a.shape[1:], transform=s.window_transform(w))
    rgb = a[:3].transpose(1, 2, 0).copy(); rgb[o > 0] = [255, 0, 255]
    ims = [rgb, a[[3, 0, 1]].transpose(1, 2, 0)]
    if with_esri:
        try: ims.append(esri_chip(c.x - half, c.y - half, c.x + half, c.y + half, res=max(half * 2 / 300, 0.3)))
        except Exception: ims.append(np.zeros((10, 10, 3), np.uint8))
    return [Image.fromarray(np.ascontiguousarray(x)).resize((px, px)) for x in ims]
def montage(gdf, out, cols=4, px=150, labels=None, half=None):
    gdf = gdf.reset_index(drop=True)
    with ThreadPoolExecutor(16) as ex: chips = list(ex.map(lambda g: chip(g, half, px), gdf.geometry))
    n = len(chips); rows = (n + cols - 1) // cols; cw = 3 * px + 12
    G = Image.new("RGB", (cols * cw, rows * (px + 4)), (255, 255, 255))
    for i, ims in enumerate(chips):
        x0, y0 = (i % cols) * cw, (i // cols) * (px + 4)
        for j, im in enumerate(ims): G.paste(im, (x0 + j * px, y0))
        txt = f"{i}" + (f" {labels[i]}" if labels is not None else "")
        d = ImageDraw.Draw(G); d.rectangle([x0, y0, x0 + 8 * len(txt) + 4, y0 + 12], fill=(0, 0, 0)); d.text((x0 + 2, y0), txt, fill=(255, 255, 0))
    G.save(out)

def chip2(center, outlines, half=60, px=150):
    """NAIP RGB with several outline layers [(geoms, rgb)], NAIP CIR, Esri."""
    t = next(t for t, tb in TILES.items() if tb.contains(center)); s = rasterio.open(t)
    w = W.from_bounds(center.x - half, center.y - half, center.x + half, center.y + half, s.transform)
    a = s.read(window=w, boundless=True); rgb = a[:3].transpose(1, 2, 0).copy()
    for geoms, col in outlines:
        gs = [g.boundary for g in geoms if g is not None and not g.is_empty]
        if gs:
            o = F.rasterize([(g, 1) for g in gs], out_shape=a.shape[1:], transform=s.window_transform(w)); rgb[o > 0] = col
    try: e = esri_chip(center.x - half, center.y - half, center.x + half, center.y + half, res=max(half * 2 / 300, 0.3))
    except Exception: e = np.zeros((8, 8, 3), np.uint8)
    return [Image.fromarray(np.ascontiguousarray(x)).resize((px, px)) for x in (rgb, a[[3, 0, 1]].transpose(1, 2, 0), e)]

def montage2(items, out, cols=4, px=150):
    """items: list of (center_point, outlines, label)."""
    with ThreadPoolExecutor(16) as ex: chips = list(ex.map(lambda it: chip2(it[0], it[1], px=px), items))
    rows = (len(chips) + cols - 1) // cols; cw = 3 * px + 12
    G = Image.new("RGB", (cols * cw, rows * (px + 4)), (255, 255, 255))
    for i, ims in enumerate(chips):
        x0, y0 = (i % cols) * cw, (i // cols) * (px + 4)
        for j, im in enumerate(ims): G.paste(im, (x0 + j * px, y0))
        txt = f"{i} {items[i][2]}"; d = ImageDraw.Draw(G); d.rectangle([x0, y0, x0 + 7 * len(txt) + 4, y0 + 12], fill=(0, 0, 0)); d.text((x0 + 2, y0), txt, fill=(255, 255, 0))
    G.save(out)
