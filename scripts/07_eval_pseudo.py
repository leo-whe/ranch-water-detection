"""Evaluate predictions on held-out test tiles against the weak labels (pixel IoU + object-level P/R),
and compare with a pure spectral-rule baseline."""
import os, sys, glob, numpy as np, rasterio, json
from scipy import ndimage as ndi
sys.path.insert(0, os.path.dirname(__file__)); from common import tile_split
pred_dir = sys.argv[1] if len(sys.argv) > 1 else "outputs/prob/2022"
def rule(a):
    r, g, b, n = a.astype(np.float32); w = (n < 0.8 * r) & (n < 0.8 * g)
    return ndi.binary_fill_holes(ndi.binary_opening(w, np.ones((3, 3))))
res = {k: dict(I=0, U=0, tp_obj=0, n_lbl=0, tp_pred=0, n_pred=0) for k in ["model", "rule_baseline"]}
MIN = int(25 / 0.36)  # objects >= 25 m^2
for t in tile_split()["test"]:
    y = rasterio.open(t.replace("naip", "labels")).read(1); v = y != 255; gt = y == 1
    preds = {"model": rasterio.open(f"{pred_dir}/{os.path.basename(t)}").read(1) > 127, "rule_baseline": rule(rasterio.open(t).read())}
    lg, ng = ndi.label(gt)
    gsz = ndi.sum(np.ones_like(lg), lg, np.arange(1, ng + 1))
    for k, p in preds.items():
        R = res[k]; R["I"] += int((p & gt & v).sum()); R["U"] += int(((p | gt) & v).sum())
        # GT object recall: GT object (>=25m2) hit if >=30% of its pixels predicted
        hit = ndi.mean(p, lg, np.arange(1, ng + 1)) >= 0.3
        R["tp_obj"] += int((hit & (gsz >= MIN)).sum()); R["n_lbl"] += int((gsz >= MIN).sum())
        # predicted object precision: pred object counts as TP if >=30% of it is on GT-positive, ignored if mostly on ignore
        lp, npd = ndi.label(p); idx = np.arange(1, npd + 1)
        psz = ndi.sum(np.ones_like(lp), lp, idx); on_gt = ndi.mean(gt, lp, idx); on_ign = ndi.mean(~v, lp, idx)
        keep = (psz >= MIN) & (on_ign < 0.5)
        R["tp_pred"] += int((keep & (on_gt >= 0.3)).sum()); R["n_pred"] += int(keep.sum())
out = {}
for k, R in res.items():
    P = R["tp_pred"] / max(R["n_pred"], 1); Rc = R["tp_obj"] / max(R["n_lbl"], 1)
    out[k] = dict(pixel_IoU=R["I"] / max(R["U"], 1), obj_precision=P, obj_recall=Rc, obj_F1=2 * P * Rc / max(P + Rc, 1e-9), **R)
print(json.dumps(out, indent=1)); os.makedirs("outputs/eval", exist_ok=True)
json.dump(out, open("outputs/eval/test_vs_weaklabels.json", "w"), indent=1)
