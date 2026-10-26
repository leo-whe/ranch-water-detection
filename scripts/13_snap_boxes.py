"""Turn reviewed candidate points into precise tank boxes on Google z19 imagery (Hough-circle snap near the point)."""
import os, sys, json, numpy as np, pandas as pd, cv2
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(__file__)); from webtiles import chip, gsd
b = pd.read_csv("data/tank_chips/review/bright_cands.csv").iloc[:384].copy()
SURE = [20,36,44,50,52,54,57, 64,69,74,79,85,86,97,99,115,116,117,118,127, 178,190, 194,201,202,204,211,222,234,244,249,252,255, 256,257,281,315,317, 325,327,337,340,341,351,362,373]
PROB = [13,14,15,31,32,46,47, 66,89,95,107,112,113,124,125, 131,146,153,159, 196,216,230,241,254, 258,266,272,294,301,314, 342,354,360,363,366,367,370]
b["label"] = "no"; b.loc[b.index.isin(SURE), "label"] = "tank"; b.loc[b.index.isin(PROB), "label"] = "probable"
def snap(r, S=128):
    img = chip("google", r.lon, r.lat, 19, S); g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (3, 3), 0.8)
    best = None
    for p2 in (22, 18, 14, 11):
        c = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1, minDist=6, param1=60, param2=p2, minRadius=4, maxRadius=30)
        if c is None: continue
        c = [v for v in c[0] if np.hypot(v[0] - S / 2, v[1] - S / 2) < 26]
        if c: best = c[0]; break
    if best is None: return S / 2, S / 2, max(6, r.rad_m / 0.254), False
    return float(best[0]), float(best[1]), float(best[2]), True
rows = []
for r in b[b.label != "no"].itertuples():
    cx, cy, rad, ok = snap(r); res = gsd(r.lat, 19)
    # pixel offset -> lon/lat via mercator pixel math at z19
    import math
    n = 2 ** 19 * 256; X = (r.lon + 180) / 360 * n + (cx - 64); Y = (1 - math.asinh(math.tan(math.radians(r.lat))) / math.pi) / 2 * n + (cy - 64)
    lon = X / n * 360 - 180; lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * Y / n))))
    rows.append(dict(cid=r.cid, label=r.label, lon=lon, lat=lat, radius_px=rad, radius_m=rad * res, snapped=ok))
t = pd.DataFrame(rows); t.to_csv("data/manual/tank_boxes_google_z19.csv", index=False)
b.to_csv("data/manual/tank_bright_review_google.csv", index=False)
print(t.label.value_counts().to_dict(), "snapped", t.snapped.mean().round(2)); print(t.radius_m.describe().round(2))
# verification montage
G = Image.new("RGB", (10 * 130, ((len(t) + 9) // 10) * 130), (255, 255, 255))
for i, r in enumerate(t.itertuples()):
    img = Image.fromarray(chip("google", r.lon, r.lat, 19, 96)).resize((128, 128)); d = ImageDraw.Draw(img); k = 128 / 96; R = r.radius_px * k
    d.rectangle([64 - R, 64 - R, 64 + R, 64 + R], outline=(255, 0, 255) if r.label == "tank" else (0, 200, 255), width=2); d.text((2, 0), str(i), fill=(255, 255, 0))
    G.paste(img, ((i % 10) * 130, (i // 10) * 130))
G.save("data/tank_chips/boxes_check.jpg")
