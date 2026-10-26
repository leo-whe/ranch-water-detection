"""Tank detector evaluation on the test area: recall vs the independent reference set at several thresholds,
plus a random precision sample (montage) above a chosen threshold. args: dets.gpkg  out_tag  [thr]"""
import os, sys, json, numpy as np, pandas as pd, geopandas as gpd
sys.path.insert(0, os.path.dirname(__file__)); from review_dets import montage
d = gpd.read_file(sys.argv[1]).to_crs(26914); tag = sys.argv[2]; thr = float(sys.argv[3]) if len(sys.argv) > 3 else None
c = pd.read_csv("outputs/eval/tank_recall_reference_test_area.csv"); r = c[c.ref != "no"]
ref = gpd.GeoDataFrame(r, geometry=gpd.points_from_xy(r.lon, r.lat), crs=4326).to_crs(26914)
keep = []
for i, p in ref.geometry.items():
    if all(p.distance(ref.geometry[j]) > 15 for j in keep): keep.append(i)
ref = ref.loc[keep]
res = {}
for t in (0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2):
    dd = d[d.conf >= t]
    hit = ref.geometry.apply(lambda p: dd.sindex.query(p.buffer(15), predicate="intersects").size > 0) if len(dd) else ref.geometry.apply(lambda p: False)
    res[f"{t}"] = dict(recall_sure=round(float(hit[ref.ref == "tank"].mean()), 3), recall_all=round(float(hit.mean()), 3), n_detections=int(len(dd)))
print(json.dumps(res, indent=1)); json.dump(dict(reference=ref.ref.value_counts().to_dict(), recall=res), open(f"outputs/eval/tank_{tag}_recall.json", "w"), indent=1)
if thr is not None:
    s = d[d.conf >= thr].to_crs(4326).sample(min(64, int((d.conf >= thr).sum())), random_state=0).reset_index(drop=True)
    s.to_file(f"outputs/eval/tank_{tag}_precision_sample.gpkg"); montage(s, f"data/tank_chips/{tag}_prec.jpg")
    print("precision sample", len(s), "from", int((d.conf >= thr).sum()))
