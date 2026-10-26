"""Fetch a north-up RGB chip around (lon,lat) from XYZ web-map imagery (Google satellite / Bing aerial), with on-disk tile cache.
Used only for the round-tank stage, where 0.6 m NAIP is too coarse to separate tanks from look-alikes."""
import io, math, os, time, hashlib, requests, numpy as np
from PIL import Image
CACHE = "data/webtiles"; H = {"User-Agent": "Mozilla/5.0 (research; ranch water mapping)"}
_S = requests.Session()
def _quadkey(x, y, z):
    q = ""
    for i in range(z, 0, -1):
        d = 0; m = 1 << (i - 1)
        if x & m: d += 1
        if y & m: d += 2
        q += str(d)
    return q
def url(src, x, y, z):
    if src == "google": return f"https://mt{(x + y) % 4}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
    if src == "bing": return f"https://ecn.t{(x + y) % 4}.tiles.virtualearth.net/tiles/a{_quadkey(x, y, z)}.jpeg?g=1"
def tile(src, x, y, z):
    fn = f"{CACHE}/{src}/{z}/{x}_{y}.jpg"
    if os.path.exists(fn): return Image.open(fn).convert("RGB")
    for k in range(4):
        try:
            r = _S.get(url(src, x, y, z), headers=H, timeout=30)
            if r.status_code == 200 and len(r.content) > 500:
                os.makedirs(os.path.dirname(fn), exist_ok=True); open(fn, "wb").write(r.content)
                return Image.open(io.BytesIO(r.content)).convert("RGB")
        except Exception: pass
        time.sleep(1 + k)
    return Image.new("RGB", (256, 256))
def fxy(lon, lat, z):
    n = 2 ** z; return (lon + 180) / 360 * n, (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n
def chip(src, lon, lat, z=19, size=256):
    """size x size px chip centred on lon/lat at zoom z (gsd ~ 0.254 m at z19 for lat 31.7)."""
    fx, fy = fxy(lon, lat, z); px, py = fx * 256, fy * 256
    x0, y0 = int(px - size / 2), int(py - size / 2)
    tx0, ty0, tx1, ty1 = x0 // 256, y0 // 256, (x0 + size) // 256, (y0 + size) // 256
    M = Image.new("RGB", ((tx1 - tx0 + 1) * 256, (ty1 - ty0 + 1) * 256))
    for tx in range(tx0, tx1 + 1):
        for ty in range(ty0, ty1 + 1): M.paste(tile(src, tx, ty, z), ((tx - tx0) * 256, (ty - ty0) * 256))
    ox, oy = x0 - tx0 * 256, y0 - ty0 * 256
    return np.array(M.crop((ox, oy, ox + size, oy + size)))
def gsd(lat, z): return 156543.03 * math.cos(math.radians(lat)) / 2 ** z
