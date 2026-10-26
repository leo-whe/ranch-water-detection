"""Candidate generator for small round man-made water tanks/troughs (diam ~2.5-9 m) in NAIP RGBN.
LoG blobs on brightness (bright & dark polarity), drop vegetation (NDVI) and tiny-contrast blobs."""
import numpy as np
from skimage.feature import blob_log
from scipy import ndimage as ndi
RES = 0.6
def blobs(a, thr=0.06):
    """a: uint8 (4,H,W) -> list of (row, col, radius_px, polarity, contrast, ndvi)."""
    a = a.astype(np.float32); r, g, b, n = a; B = a[:3].mean(0) / 255.0
    ndvi = (n - r) / (n + r + 1e-6)
    out = []
    for pol, img in (("bright", B), ("dark", 1 - B)):
        for y, x, s in blob_log(img, min_sigma=1.5, max_sigma=5.0, num_sigma=8, threshold=thr, overlap=0.3):
            y, x = int(y), int(x); rad = s * np.sqrt(2)
            R = int(np.ceil(rad)); y0, y1, x0, x1 = max(y - R, 0), y + R + 1, max(x - R, 0), x + R + 1
            inner = img[y0:y1, x0:x1].mean()
            Y0, Y1, X0, X1 = max(y - 3 * R, 0), y + 3 * R + 1, max(x - 3 * R, 0), x + 3 * R + 1
            outer = img[Y0:Y1, X0:X1].mean()
            nd = ndvi[y0:y1, x0:x1].mean()
            out.append((y, x, rad, pol, inner - outer, nd))
    return out

def blobs_filtered(a, thr=0.06, min_contrast=0.08):
    """blobs() minus anything touching tree canopy/shadow (NDVI>0.12 dilated 6 px) - trees are the dominant false positive."""
    af = a.astype(np.float32); ndvi = (af[3] - af[0]) / (af[3] + af[0] + 1e-6)
    tree = ndi.binary_dilation(ndi.binary_opening(ndvi > 0.12, np.ones((3, 3))), iterations=6)
    out = []
    for (y, x, rad, pol, con, nd) in blobs(a, thr):
        if abs(con) < min_contrast: continue
        R = int(np.ceil(rad)); 
        if tree[max(y - R, 0):y + R + 1, max(x - R, 0):x + R + 1].mean() > 0.1: continue
        out.append((y, x, rad, pol, con, nd))
    return out
