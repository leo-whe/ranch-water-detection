"""Test-area evaluation of a tank detector against the z20-verified reference (18 water facilities, same as the two-stage
YOLO system's 72 % recall), plus a random precision sample montage (z20) at a chosen threshold.
args: dets.gpkg name thr"""
import os, sys, json, numpy as np, geopandas as gpd
sys.path.insert(0, os.path.dirname(__file__))
d = gpd.read_file(sys.argv[1]).to_crs(26914); name = sys.argv[2]; thr = float(sys.argv[3])
ref = gpd.read_file("outputs/eval/tank_2stage_recall_reference.gpkg").to_crs(26914); R = ref[ref.z20 == "water"]
grid = {}
for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
    dd = d[d.conf >= t]; hit = R.geometry.apply(lambda p: dd.sindex.query(p.buffer(15), predicate="intersects").size > 0) if len(dd) else R.geometry.apply(lambda p: False)
    grid[str(t)] = dict(recall=round(float(hit.mean()), 3), n_dets_test_area=int(len(dd)))
print(json.dumps(grid, indent=1))
s = d[d.conf >= thr].to_crs(4326).sample(min(48, int((d.conf >= thr).sum())), random_state=0).reset_index(drop=True)
os.makedirs("outputs/eval/ssl", exist_ok=True); s.to_file(f"outputs/eval/ssl/{name}_prec_sample.gpkg")
json.dump(dict(recall_grid=grid, n_reference=len(R), precision_sample_thr=thr), open(f"outputs/eval/ssl/{name}.json", "w"), indent=1)
os.system(f"{sys.executable} scripts/20_z20_review.py outputs/eval/ssl/{name}_prec_sample.gpkg outputs/eval/ssl/{name}_z20 48 0 0 > /dev/null 2>&1")
print("precision sample:", len(s))
