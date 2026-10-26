import glob, json, os, numpy as np, rasterio
from shapely.geometry import box

def tile_split(year="2022"):
    """Spatial split by NAIP quarter-quad. Test = a contiguous block in the county's south-west/centre
    ranch land; val = a few scattered tiles; train = rest."""
    tiles = sorted(glob.glob(f"data/naip/{year}/*.tif"))
    sp = json.load(open("data/split.json"))
    key = lambda t: "_".join(os.path.basename(t).split("_")[2:4])  # e.g. 3109814_se
    return {k: [t for t in tiles if key(t) in set(v)] for k, v in sp.items()}

def norm(a):
    """uint8 RGBN -> float32 roughly standardised."""
    a = a.astype(np.float32) / 255.0
    mean = np.array([0.45, 0.43, 0.38, 0.45], np.float32)[: a.shape[0], None, None]
    std = np.array([0.15, 0.13, 0.12, 0.15], np.float32)[: a.shape[0], None, None]
    return (a - mean) / std
