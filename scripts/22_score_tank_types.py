"""Score YOLO candidates with the saved tank-type classifier (runs/tank_type_cls.pt). args: in.gpkg out.gpkg [min_conf]"""
import os, sys, numpy as np, geopandas as gpd, torch, timm, torch.nn.functional as Fn
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(__file__)); from webtiles import chip
CLS = ["water", "hay_ring", "bale", "other"]; dev = os.environ.get("DEV", "cuda:1")
m = timm.create_model("resnet18", pretrained=False, num_classes=4); m.load_state_dict(torch.load("runs/tank_type_cls.pt", map_location="cpu")); m.to(dev).eval()
d = gpd.read_file(sys.argv[1]); mc = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2; d = d[d.conf >= mc].reset_index(drop=True)
P = []
with torch.no_grad():
    for b in range(0, len(d), 2000):
        sub = d.iloc[b:b + 2000]
        with ThreadPoolExecutor(24) as ex: X = np.stack(list(ex.map(lambda r: chip("google", r.lon, r.lat, 20, 128), sub.itertuples())))
        X = ((X.astype(np.float32) / 255 - 0.45) / 0.2).transpose(0, 3, 1, 2)
        for k in range(0, len(X), 128):
            xb = Fn.interpolate(torch.tensor(X[k:k + 128]).to(dev), size=224, mode="bilinear")
            P.append((sum(torch.softmax(m(torch.rot90(xb, r, (2, 3))), 1) for r in range(4)) / 4).cpu().numpy())
        print(b, flush=True)
P = np.concatenate(P); d[[f"p_{c}" for c in CLS]] = P; d["tank_type"] = [CLS[i] for i in P.argmax(1)]
d["water_source"] = (d.tank_type == "water") & (d.p_water >= 0.8)
d.to_file(sys.argv[2]); print(d.tank_type.value_counts().to_dict(), "water_source:", int(d.water_source.sum()))
