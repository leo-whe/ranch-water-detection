"""Weak/pseudo labels (v2) for pond water per NAIP tile.

Water-like pixels: NIR < 0.8*R and NIR < 0.8*G. Each connected component is then classified:
label 1   : near an NWI pond/lake (NWI is often offset 20-50 m, so buffered 60 m), or a compact, clearly
            watery standalone body (unmapped pond); never on road corridors / buildings, never low-NDWI
label 0   : background, plus water-like components that are NOT ponds: on road corridors or roofs,
            elongated (roads, ditches), or low-NDWI dark soil (tilled fields) away from NWI
label 255 : ignore - compact watery components we cannot vouch for (possible unmapped pond or shadow),
            riverine water, ambiguous low-NDWI pixels next to NWI ponds, 1px border ring, nodata
v1 left every non-NWI water-like component as 'ignore', so the model never learned that roads,
fields and roofs are negatives; v2 makes those explicit.
"""
import os, sys, glob, numpy as np, geopandas as gpd, rasterio, rasterio.features as F
from shapely.geometry import box
from scipy import ndimage as ndi
from skimage import measure
from multiprocessing import Pool

YEAR = os.environ.get("NAIP_YEAR", "2022")
OUT = f"data/labels/{YEAR}"; os.makedirs(OUT, exist_ok=True)
nwi = gpd.read_file("data/vector/nwi_hamilton.gpkg").to_crs(26914)
POND = nwi[nwi.WETLAND_TYPE.isin(["Freshwater Pond", "Lake"])].geometry
RIV = nwi[nwi.WETLAND_TYPE == "Riverine"].geometry
ROADS = gpd.read_file("zip://data/vector/tl_2023_48193_roads.zip").to_crs(26914).geometry
BLDG = gpd.read_file("data/vector/ms_buildings.gpkg").to_crs(26914).geometry

def water_rule(a):
    r, g, b, n = a.astype(np.float32)
    return (n < 0.8 * r) & (n < 0.8 * g) & (a[:3].max(0) > 0)

def raster(geoms, s, shape):
    geoms = [g for g in geoms if not g.is_empty]
    if not geoms: return np.zeros(shape, bool)
    return F.rasterize([(g, 1) for g in geoms], out_shape=shape, transform=s.transform).astype(bool)

def make(path):
    out = f"{OUT}/{os.path.basename(path)}"
    if os.path.exists(out): return out
    s = rasterio.open(path); a = s.read(); shape = a.shape[1:]; tb = box(*s.bounds).buffer(100)
    w = ndi.binary_opening(water_rule(a), structure=np.ones((3, 3)))
    w = ndi.binary_fill_holes(w)
    near_pond = raster(POND[POND.intersects(tb)].buffer(60), s, shape)
    riv = raster(RIV[RIV.intersects(tb)].buffer(5), s, shape)
    neg = raster(ROADS[ROADS.intersects(tb)].buffer(15), s, shape) | raster(BLDG[BLDG.intersects(tb)].buffer(3), s, shape)
    lab, n = ndi.label(w)
    lbl = np.zeros(shape, np.uint8)
    if n:
        idx = np.arange(1, n + 1)
        area = ndi.sum(np.ones(shape), lab, idx)
        f_near = ndi.mean(near_pond, lab, idx); f_riv = ndi.mean(riv, lab, idx); f_neg = ndi.mean(neg, lab, idx)
        sol = np.ones(n); elong = np.ones(n)
        t = measure.regionprops_table(lab, properties=("label", "area", "solidity", "axis_major_length", "axis_minor_length"))
        big = t["area"] >= 30
        sol[t["label"][big] - 1] = t["solidity"][big]
        elong[t["label"][big] - 1] = t["axis_major_length"][big] / np.maximum(t["axis_minor_length"][big], 1)
        A = area * 0.36  # m^2
        r_, g_, b_, n_ = a.astype(np.float32)
        m_ndwi = ndi.mean((g_ - n_) / (g_ + n_ + 1e-6), lab, idx); m_nr = ndi.mean(n_ / (r_ + 1), lab, idx)
        weak = (m_ndwi < 0.16) & (m_nr > 0.66)  # muddy edge / wet soil / dark tilled field: spectrally ambiguous
        near = f_near > 0; onneg = f_neg >= 0.2; riv = f_riv >= 0.5
        compact = (sol > 0.6) & (elong < 4)
        cls = np.full(n + 1, 0, np.uint8)
        pos = ~onneg & ~riv & ~weak & (A >= 10) & (near | ((A >= 50) & (sol > 0.8) & (elong < 3) & (m_ndwi > 0.25)))
        ign = ~pos & ~onneg & (riv | (weak & near) | (~weak & compact) | (~weak & near))
        cls[1:][ign] = 255; cls[1:][pos] = 1
        lbl = cls[lab]
        ring = ndi.binary_dilation(lbl == 1, iterations=1) & (lbl != 1)
        lbl[ring] = 255
    lbl[a[:3].max(0) == 0] = 255  # nodata
    prof = s.profile.copy(); prof.update(count=1, dtype="uint8", nodata=None, compress="deflate")
    with rasterio.open(out, "w", **prof) as d: d.write(lbl, 1)
    return out

def unpenalise_nwi_interior(path):
    """Algae/duckweed-covered ponds fail the spectral water rule; inside mapped NWI ponds (+10 m) don't call them background."""
    with rasterio.open(path, "r+") as s:
        l = s.read(1); tb = box(*s.bounds).buffer(100)
        m = raster(POND[POND.intersects(tb)].buffer(10), s, l.shape)
        l[(l == 0) & m] = 255; s.write(l, 1)
    return path

if __name__ == "__main__":
    tiles = sorted(glob.glob(f"data/naip/{YEAR}/*.tif"))
    with Pool(24) as p:
        for o in p.imap_unordered(make, tiles): print(o, flush=True)
        for o in p.imap_unordered(unpenalise_nwi_interior, sorted(glob.glob(f"{OUT}/*.tif"))): pass
