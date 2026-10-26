"""Compare pond models on the test area. args: name prob_dir
 1) weak-label metrics (07_eval_pseudo.py logic)
 2) recall on the 42 manually verified water-holding NWI ponds (auto rule: a predicted object >=25 m2 within 40 m)
 3) random sample of 60 predicted objects (>=25 m2) -> review montage for manual precision"""
import os, sys, json, subprocess, numpy as np, geopandas as gpd, rasterio, rasterio.features as F
from scipy import ndimage as ndi
from shapely.geometry import shape, box
sys.path.insert(0, os.path.dirname(__file__)); from common import tile_split
name, pdir = sys.argv[1], sys.argv[2]
r = subprocess.run([sys.executable, "scripts/07_eval_pseudo.py", pdir], capture_output=True, text=True).stdout
weak = json.loads(r[r.index("{"):])["model"]
objs = []
for t in tile_split()["test"]:
    s = rasterio.open(f"{pdir}/{os.path.basename(t)}"); m = s.read(1) > 127
    lab, n = ndi.label(m)
    for g, v in F.shapes(lab.astype(np.int32), mask=m, transform=s.transform):
        g = shape(g)
        if g.area >= 25: objs.append(g)
P = gpd.GeoDataFrame(geometry=objs, crs=26914)
rec = gpd.read_file("outputs/eval/manual_recall_sample.gpkg").to_crs(26914); rec = rec[rec.water_visible_2022.astype(bool)]
hit = rec.geometry.buffer(40).apply(lambda b: P.sindex.query(b, predicate="intersects").size > 0)
out = dict(name=name, weak_label=dict(obj_precision=weak["obj_precision"], obj_recall=weak["obj_recall"], obj_F1=weak["obj_F1"], pixel_IoU=weak["pixel_IoU"]),
           nwi_recall_auto=float(hit.mean()), n_objects_test_area=len(P))
smp = P.sample(min(60, len(P)), random_state=42).reset_index(drop=True); os.makedirs("outputs/eval/ssl", exist_ok=True)
smp.to_file(f"outputs/eval/ssl/{name}_precision_sample.gpkg")
from review_chips import montage2
items = [(g.centroid, [([g], (255, 0, 255))], f"{g.area:.0f}m2") for g in smp.geometry]
montage2(items[:30], f"outputs/eval/ssl/{name}_prec_a.jpg"); montage2(items[30:], f"outputs/eval/ssl/{name}_prec_b.jpg")
json.dump(out, open(f"outputs/eval/ssl/{name}.json", "w"), indent=1); print(json.dumps(out, indent=1))
