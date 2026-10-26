"""Collect all SSL-vs-baseline comparison results into outputs/eval/ssl/summary.json + a markdown table."""
import json, os, glob
E = "outputs/eval/ssl"; S = {}
# pond segmentation (test area)
pond = {}
for f in sorted(glob.glob(f"{E}/*.json")):
    d = json.load(open(f))
    if "weak_label" in d: pond[d["name"]] = dict(**{k: round(v, 3) for k, v in d["weak_label"].items()}, nwi_recall=round(d["nwi_recall_auto"], 3))
S["pond"] = pond
# tank-type classifier CV
cv = {}
for f in ["outputs/eval/ssl_tank_cls_cv.json"]:
    if os.path.exists(f): cv.update(json.load(open(f)))
S["tank_type_cv"] = {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in cv.items()}
# tank detection
det = {}
for f in sorted(glob.glob(f"{E}/tankdet_*_2stage.json")): det[os.path.basename(f)[8:-12]] = json.load(open(f))
S["tank_detection"] = det
json.dump(S, open(f"{E}/summary.json", "w"), indent=1)
lines = ["| pond model | pixel IoU | obj P | obj R | obj F1 | NWI recall |", "|---|---|---|---|---|---|"]
for k, v in pond.items(): lines.append(f"| {k} | {v['pixel_IoU']} | {v['obj_precision']} | {v['obj_recall']} | {v['obj_F1']} | {v['nwi_recall']} |")
lines += ["", "| tank classifier (CV) | acc | water P | water R | water F1 |", "|---|---|---|---|---|"]
for k, v in S["tank_type_cv"].items(): lines.append(f"| {k} | {v['acc']} | {v['water_precision']} | {v['water_recall']} | {v['water_f1']} |")
open(f"{E}/summary.md", "w").write("\n".join(lines)); print("\n".join(lines))
