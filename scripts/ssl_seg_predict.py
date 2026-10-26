"""Sliding-window inference of an SSL ViTSeg pond model over NAIP tiles -> uint8 probability GeoTIFFs.
args: --ckpt --bands --year --tiles test|all --out --gpu (non-2022 years are radiometrically matched to 2022 like 05_predict.py)"""
import os, sys, glob, argparse, numpy as np, torch, rasterio
sys.path.insert(0, os.path.dirname(__file__)); from ssl_models import ViTSeg, prep; from common import tile_split
ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--bands", type=int, default=4)
ap.add_argument("--year", default="2022"); ap.add_argument("--tiles", default="test"); ap.add_argument("--out", required=True)
ap.add_argument("--gpu", default="0"); ap.add_argument("--shard", default="0/1"); ap.add_argument("--vit_scale", type=int, default=1)
ap.add_argument("--skip_dir", default=None, help="skip windows with no water evidence in this CNN prob dir + spectral rule")
args = ap.parse_args(); dev = f"cuda:{args.gpu}"; os.makedirs(args.out, exist_ok=True)
m = ViTSeg(args.bands, "dinov3sat", args.vit_scale); m.load_state_dict(torch.load(args.ckpt, map_location="cpu")); m.to(dev).eval()
tiles = sorted(glob.glob(f"data/naip/{args.year}/*.tif"))
if args.tiles != "all":
    keys = {"_".join(os.path.basename(t).split("_")[2:4]) for t in tile_split()[args.tiles]}
    tiles = [t for t in tiles if "_".join(os.path.basename(t).split("_")[2:4]) in keys]
k_, n_ = map(int, args.shard.split("/")); tiles = tiles[k_::n_]
REF_MEAN = [128.8, 124.2, 108.3, 141.2]; REF_STD = [30.6, 24.8, 23.2, 20.3]
S, O = (1024, 128) if args.vit_scale == 1 else (512, 64); step = S - 2 * O
for t in tiles:
    fo = f"{args.out}/{os.path.basename(t)}"
    if os.path.exists(fo): continue
    s = rasterio.open(t); a = s.read(); H, W = a.shape[1:]; nod = a[:3].max(0) == 0
    if args.year != "2022":
        af = a.astype(np.float32); v = ~nod
        for b in range(4): af[b] = (af[b] - af[b][v].mean()) / (af[b][v].std() + 1e-6) * REF_STD[b] + REF_MEAN[b]
        a = np.clip(af, 0, 255).astype(np.uint8)
    xp = np.pad(a, ((0, 0), (O, O + step), (O, O + step)), mode="reflect"); prob = np.zeros((H, W), np.float32)
    coords = [(r, c) for r in range(0, H, step) for c in range(0, W, step)]
    if args.skip_dir:   # only run the ViT where the CNN or the spectral rule sees any water (ensemble cannot fire elsewhere)
        u = rasterio.open(f"{args.skip_dir}/{os.path.basename(t)}").read(1)
        wl = (a[3].astype(np.int16) * 5 < a[0].astype(np.int16) * 4) & (a[3].astype(np.int16) * 5 < a[1].astype(np.int16) * 4) & ~nod
        ev = (u > 13) | wl
        coords = [(r, c) for r, c in coords if ev[r:r + step, c:c + step].sum() >= 10]
    with torch.no_grad():
        for i in range(0, len(coords), 4):
            cs = coords[i:i + 4]; xb = torch.from_numpy(np.stack([xp[:, r:r + S, c:c + S] for r, c in cs])).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                x = prep(xb, args.bands); p = (torch.sigmoid(m(x).float()) + torch.sigmoid(m(x.flip(-1)).float()).flip(-1)) / 2
            p = p[:, 0, O:S - O, O:S - O].cpu().numpy()
            for (r, c), pi in zip(cs, p):
                hh, ww = min(step, H - r), min(step, W - c); prob[r:r + hh, c:c + ww] = pi[:hh, :ww]
    prob[nod] = 0; prof = s.profile.copy(); prof.update(count=1, dtype="uint8", compress="deflate", nodata=None)
    with rasterio.open(fo, "w", **prof) as d: d.write((prob * 255).round().astype(np.uint8), 1)
    print(fo, flush=True)
