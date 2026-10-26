"""Two-stage evaluation (dense detector + type classifier) on the test area using the manually labelled precision sample
of the dense detector. args: typed_dets.gpkg labelled_sample.gpkg"""
import sys, json, geopandas as gpd
d = gpd.read_file(sys.argv[1]).to_crs(26914); s = gpd.read_file(sys.argv[2]).to_crs(26914)
ref = gpd.read_file("outputs/eval/tank_2stage_recall_reference.gpkg").to_crs(26914); R = ref[ref.z20 == "water"]
j = gpd.sjoin_nearest(s[["geometry", "manual"]], d[["geometry", "p_water", "tank_type"]], max_distance=2); j = j[~j.index.duplicated()]
out = {}
for pw in (0.0, 0.5, 0.7, 0.8, 0.9):
    keep = (d.p_water >= pw) & ((d.tank_type == "water") | (pw == 0)); dd = d[keep]
    hit = R.geometry.apply(lambda p: dd.sindex.query(p.buffer(15), predicate="intersects").size > 0)
    q = j[(j.p_water >= pw) & ((j.tank_type == "water") | (pw == 0))]
    out[str(pw)] = dict(recall=round(float(hit.mean()), 3), n_test_dets=int(len(dd)), n_sample=int(len(q)),
                        precision_strict=round(float((q.manual == "water").mean()), 3), precision_lenient=round(float((q.manual != "no").mean()), 3))
print(json.dumps(out, indent=1))
