"""Sliding-window inference of the pond U-Net over NAIP tiles -> probability GeoTIFFs (uint8 0-255)."""
import os, sys, glob, argparse, numpy as np, torch, rasterio
import segmentation_models_pytorch as smp
sys.path.insert(0, os.path.dirname(__file__)); from common import norm, tile_split
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="runs/unet_rgbn/best.pt"); ap.add_argument("--bands", type=int, default=4)
ap.add_argument("--encoder", default="efficientnet-b4"); ap.add_argument("--year", default="2022")
ap.add_argument("--tiles", default="all"); ap.add_argument("--shard", default="0/1"); ap.add_argument("--match", type=int, default=None, help="1=match tile band stats to 2022 (default: on for non-2022 years)"); ap.add_argument("--out", default=None); ap.add_argument("--gpu", default="0")
args = ap.parse_args()
dev = f"cuda:{args.gpu}"; out = args.out or f"outputs/prob/{args.year}"; os.makedirs(out, exist_ok=True)
model = smp.Unet(args.encoder, encoder_weights=None, in_channels=args.bands, classes=1)
model.load_state_dict(torch.load(args.ckpt, map_location="cpu")); model.to(dev).eval()
tiles = sorted(glob.glob(f"data/naip/{args.year}/*.tif"))
if args.tiles != "all":
    keys = {"_".join(os.path.basename(t).split("_")[2:4]) for t in tile_split()[args.tiles]}
    tiles = [t for t in tiles if "_".join(os.path.basename(t).split("_")[2:4]) in keys]
k_, n_ = map(int, args.shard.split("/")); tiles = tiles[k_::n_]
S, O = 1024, 128
REF_MEAN = [128.8, 124.2, 108.3, 141.2]; REF_STD = [30.6, 24.8, 23.2, 20.3]  # 2022 county-wide NAIP band stats (training domain)
MATCH = bool(args.match) if args.match is not None else args.year != "2022"  # window, overlap margin discarded on each side
for t in tiles:
    fo = f"{out}/{os.path.basename(t)}"
    if os.path.exists(fo): continue
    s = rasterio.open(t); a = s.read(); H, W = a.shape[1:]
    if MATCH:  # radiometric matching: other NAIP years/seasons are much darker than the 2022 training imagery
        v = a[:3].max(0) > 0; af = a.astype(np.float32)
        for b in range(4):
            mu, sd = af[b][v].mean(), af[b][v].std() + 1e-6
            af[b] = (af[b] - mu) / sd * REF_STD[b] + REF_MEAN[b]
        a_in = np.clip(af, 0, 255)
    else:
        a_in = a
    x = norm(a_in)[: args.bands]; prob = np.zeros((H, W), np.float32)
    step = S - 2 * O
    xp = np.pad(x, ((0, 0), (O, O + step), (O, O + step)), mode="reflect")
    coords = [(r, c) for r in range(0, H, step) for c in range(0, W, step)]
    with torch.no_grad():
        for i in range(0, len(coords), 8):
            cs = coords[i:i + 8]
            xb = torch.from_numpy(np.stack([xp[:, r:r + S, c:c + S] for r, c in cs])).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = (torch.sigmoid(model(xb).float()) + torch.sigmoid(model(xb.flip(-1)).float()).flip(-1)) / 2
            p = p[:, 0, O:S - O, O:S - O].cpu().numpy()
            for (r, c), pi in zip(cs, p):
                hh, ww = min(step, H - r), min(step, W - c); prob[r:r + hh, c:c + ww] = pi[:hh, :ww]
    prob[a[:3].max(0) == 0] = 0
    prof = s.profile.copy(); prof.update(count=1, dtype="uint8", compress="deflate", nodata=None)
    with rasterio.open(fo, "w", **prof) as d: d.write((prob * 255).round().astype(np.uint8), 1)
    print(fo, flush=True)
