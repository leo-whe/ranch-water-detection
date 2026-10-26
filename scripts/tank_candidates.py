"""Find small round water blobs (possible circular stock tanks/troughs) in NAIP tiles; save montage + csv."""
import sys, glob, numpy as np, rasterio, pandas as pd
from skimage import measure
from scipy import ndimage as ndi
from PIL import Image, ImageDraw
def water_rule(a):
    r, g, b, n = a.astype(np.float32)
    return (n < 0.8 * r) & (n < 0.8 * g)
def candidates(path):
    s = rasterio.open(path); a = s.read()
    w = water_rule(a)
    lab, _ = ndi.label(w)
    br = a[:3].astype(np.float32).mean(0)
    out = []
    for p in measure.regionprops(lab):
        if not (4 <= p.area <= 250): continue          # 1.4 - 90 m^2
        if p.eccentricity > 0.8 or p.solidity < 0.8: continue
        r0, c0, r1, c1 = p.bbox
        R0, C0, R1, C1 = max(r0-4, 0), max(c0-4, 0), r1+4, c1+4
        ring = ndi.binary_dilation(lab[R0:R1, C0:C1] == p.label, iterations=3) & ~ndi.binary_dilation(lab[R0:R1, C0:C1] == p.label, iterations=1)
        ring_br = br[R0:R1, C0:C1][ring].mean()
        bg = br[max(r0-15,0):r1+15, max(c0-15,0):c1+15].mean()
        x, y = s.xy(*map(int, p.centroid))
        out.append(dict(tile=path.split("/")[-1], row=int(p.centroid[0]), col=int(p.centroid[1]), x=x, y=y,
                        area_m2=p.area * 0.36, ecc=p.eccentricity, ring_br=ring_br, bg_br=bg, ring_contrast=ring_br - bg))
    return out
if __name__ == "__main__":
    tiles = sorted(glob.glob("data/naip/2022/*.tif"))[:int(sys.argv[1])]
    df = pd.DataFrame(sum([candidates(t) for t in tiles], []))
    df.to_csv("viz/tank_candidates.csv", index=False); print(len(df))
    top = df.sort_values("ring_contrast", ascending=False).head(64)
    tiles_open = {}
    grid = Image.new("RGB", (8 * 122, 8 * 122))
    for i, (_, d) in enumerate(top.iterrows()):
        s = tiles_open.setdefault(d.tile, rasterio.open("data/naip/2022/" + d.tile))
        win = rasterio.windows.Window(d.col - 30, d.row - 30, 60, 60)
        ch = s.read([1, 2, 3], window=win, boundless=True).transpose(1, 2, 0)
        im = Image.fromarray(ch).resize((120, 120), Image.NEAREST); ImageDraw.Draw(im).text((2, 2), str(i), fill=(255, 0, 0))
        grid.paste(im, ((i % 8) * 122, (i // 8) * 122))
    grid.save("viz/tank_candidates.jpg"); print(top.head(64)[["tile","area_m2","ring_contrast"]].to_string())
