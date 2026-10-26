"""Domain-adaptive self-supervised pretraining of DINOv3-sat ViT-L on unlabeled Hamilton County imagery.

Objective (DINOv2/v3 family): DINO self-distillation on the [CLS] token over multi-crop views + iBOT masked-patch
prediction + KoLeo regulariser, EMA teacher, and Gram anchoring (patch-similarity structure of the student is tied to the
frozen original DINOv3-sat, preventing the drift/collapse of dense features during continued pretraining).

--source naip   : NAIP 2018/2020/2022, 4 bands (RGB+NIR) -> the model learns to use NIR without labels
--source google : Google z19 tiles (~0.25 m RGB) -> sharper ranch-object features (troughs, tanks, bales)
Run: torchrun --nproc_per_node=2 scripts/ssl_pretrain.py --source naip --out runs/ssl/naip
"""
import os, sys, glob, math, json, time, random, argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
sys.path.insert(0, os.path.dirname(__file__))
from ssl_backbone import build_backbone, forward_tokens, normalize

ap = argparse.ArgumentParser()
ap.add_argument("--source", choices=["naip", "google"], required=True); ap.add_argument("--out", required=True)
ap.add_argument("--iters", type=int, default=20000); ap.add_argument("--bs", type=int, default=48)  # per GPU
ap.add_argument("--n_local", type=int, default=6); ap.add_argument("--lr", type=float, default=4e-5)
ap.add_argument("--head_lr", type=float, default=4e-4); ap.add_argument("--freeze_iters", type=int, default=1000)
ap.add_argument("--K", type=int, default=16384); ap.add_argument("--gram_w", type=float, default=1.0)
ap.add_argument("--workers", type=int, default=14)
args = ap.parse_args()

dist.init_process_group("nccl"); rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(int(os.environ["LOCAL_RANK"])); dev = torch.device("cuda")
torch.manual_seed(rank); random.seed(rank); np.random.seed(rank)
C = 4 if args.source == "naip" else 3
GLOBAL, LOCAL = 224, 96

# ----------------------------------------------------------------------------------------------- data
class NaipWindows(torch.utils.data.IterableDataset):
    """Random 320x320 windows (192 m) from all NAIP tiles/years; skips windows touching nodata."""
    def __iter__(self):
        import rasterio
        from rasterio.windows import Window
        wi = torch.utils.data.get_worker_info(); rng = np.random.default_rng(1000 * rank + (wi.id if wi else 0) + int(time.time()))
        tiles = sorted(glob.glob("data/naip/20*/*.tif")); hs = {}
        while True:
            t = tiles[rng.integers(len(tiles))]
            s = hs.get(t) or hs.setdefault(t, rasterio.open(t))
            r, c = rng.integers(0, s.height - 320), rng.integers(0, s.width - 320)
            a = s.read(window=Window(c, r, 320, 320))
            if (a[:3].max(0) == 0).mean() > 0.01: continue
            yield torch.from_numpy(a)

class GoogleWindows(torch.utils.data.IterableDataset):
    """Random 384x384 windows from 2x2 mosaics of cached Google z19 tiles inside the county."""
    def __iter__(self):
        from PIL import Image
        wi = torch.utils.data.get_worker_info(); rng = np.random.default_rng(1000 * rank + (wi.id if wi else 0) + int(time.time()))
        R = json.load(open("data/tank_chips/county_tile_range.json")); root = "data/webtiles/google/19"
        while True:
            x, y = rng.integers(R["x0"], R["x1"]), rng.integers(R["y0"], R["y1"])
            fs = [f"{root}/{x + i}_{y + j}.jpg" for j in (0, 1) for i in (0, 1)]
            if not all(os.path.exists(f) for f in fs): continue
            ims = [np.asarray(Image.open(f).convert("RGB")) for f in fs]
            m = np.concatenate([np.concatenate(ims[:2], 1), np.concatenate(ims[2:], 1)], 0)
            if m.mean() < 8 or m.std() < 6: continue           # blank / no-data tiles
            ox, oy = rng.integers(0, 128, 2)
            yield torch.from_numpy(np.ascontiguousarray(m[oy:oy + 384, ox:ox + 384].transpose(2, 0, 1)))

def collate(b): return torch.stack(b)

# --------------------------------------------------------------------------------- GPU multi-crop aug
def rrc(x, size, scale):
    """Batched random-resized-crop + random flip / 90-degree rotation, x: (B,C,H,W) float [0,1]."""
    B, _, H, W = x.shape; out = []
    for i in range(B):
        s = random.uniform(*scale); ar = math.exp(random.uniform(math.log(3 / 4), math.log(4 / 3)))
        h = int(min(H, round(math.sqrt(s * ar) * H))); w = int(min(W, round(math.sqrt(s / ar) * W)))
        r, c = random.randint(0, H - h), random.randint(0, W - w)
        crop = F.interpolate(x[i:i + 1, :, r:r + h, c:c + w], size=(size, size), mode="bilinear", antialias=True, align_corners=False)
        crop = torch.rot90(crop, random.randint(0, 3), (2, 3))
        if random.random() < 0.5: crop = crop.flip(3)
        out.append(crop)
    return torch.cat(out)

def photometric(x, p_blur):
    """Per-sample brightness/contrast/per-band gain jitter (keeps band ratios roughly physical) + optional blur."""
    B, Cc = x.shape[:2]
    gain = 1 + 0.25 * (torch.rand(B, 1, 1, 1, device=x.device) - 0.5) * 2
    band = 1 + 0.08 * (torch.rand(B, Cc, 1, 1, device=x.device) - 0.5) * 2
    mu = x.mean((1, 2, 3), keepdim=True); con = 1 + 0.3 * (torch.rand(B, 1, 1, 1, device=x.device) - 0.5) * 2
    x = ((x - mu) * con + mu) * gain * band
    if p_blur > 0:
        k = torch.tensor([1, 4, 6, 4, 1.], device=x.device); k = (k[:, None] * k[None]) / 256
        blur = F.conv2d(F.pad(x, (2, 2, 2, 2), mode="reflect"), k.expand(Cc, 1, 5, 5), groups=Cc)
        sel = (torch.rand(B, 1, 1, 1, device=x.device) < p_blur).float(); x = sel * blur + (1 - sel) * x
    return x.clamp(0, 1)

def ibot_masks(B, h, w):
    """Half the samples get a random patch mask with ratio U(0.1, 0.5) (DINOv2 setting)."""
    m = torch.zeros(B, h, w, dtype=torch.bool, device=dev)
    for i in range(B):
        if random.random() < 0.5:
            n = int(random.uniform(0.1, 0.5) * h * w); idx = torch.randperm(h * w, device=dev)[:n]
            m[i].view(-1)[idx] = True
    return m

# ---------------------------------------------------------------------------------------- model
class Head(nn.Module):
    def __init__(self, d=1024, K=16384, hidden=2048, bott=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, bott))
        self.last = nn.utils.parametrizations.weight_norm(nn.Linear(bott, K, bias=False))
        self.last.parametrizations.weight.original0.data.fill_(1); self.last.parametrizations.weight.original0.requires_grad = False
    def forward(self, x): return self.last(F.normalize(self.mlp(x), dim=-1))

class Net(nn.Module):
    def __init__(self, bb):
        super().__init__(); self.bb = bb; self.dino = Head(K=args.K); self.ibot = Head(K=args.K)
    def forward(self, g, masks, l):
        """Student pass (must go through the DDP wrapper so gradients are all-reduced)."""
        s_g = [forward_tokens(self.bb, x, m) for x, m in zip(g, masks)]
        s_l = [forward_tokens(self.bb, x) for x in l]
        cls_logits = [self.dino(c) for c, _ in s_g + s_l]
        pat_logits = [self.ibot(p[m.flatten(1)]) if m.any() else None for (_, p), m in zip(s_g, masks)]
        return s_g, cls_logits, pat_logits

student = Net(build_backbone(C, drop_path=0.1)).to(dev)
teacher = Net(build_backbone(C)).to(dev); teacher.load_state_dict(student.state_dict(), strict=False)
for p in teacher.parameters(): p.requires_grad = False
anchor = build_backbone(3).to(dev).eval()                       # frozen original DINOv3-sat (RGB) for Gram anchoring
for p in anchor.parameters(): p.requires_grad = False
student_ddp = DDP(student, find_unused_parameters=False)

def layer_id(n):
    if n.startswith("bb.blocks."): return int(n.split(".")[2]) + 1
    if n.startswith("bb.") and not n.startswith("bb.norm"): return 0
    return 25
groups = {}
for n, p in student.named_parameters():
    if not p.requires_grad: continue
    is_head = not n.startswith("bb.")
    lr = args.head_lr if is_head else args.lr * (0.9 ** (25 - layer_id(n)))  # layer-wise lr decay
    if n == "bb.patch_embed.proj.weight": lr = args.lr * 5               # NIR filters start from zero
    wd = 0.0 if (p.ndim == 1 or "token" in n or "pos" in n) else 0.04
    groups.setdefault((lr, wd, is_head), []).append(p)
opt = torch.optim.AdamW([dict(params=v, lr=k[0], weight_decay=k[1], base_lr=k[0], is_head=k[2]) for k, v in groups.items()], betas=(0.9, 0.999))

center_cls = torch.zeros(1, args.K, device=dev); center_patch = torch.zeros(1, args.K, device=dev)

def sched(it):
    warm = args.freeze_iters + 1000
    bb_f = 0.0 if it < args.freeze_iters else min(1.0, (it - args.freeze_iters) / 1000)
    cos = 0.5 * (1 + math.cos(math.pi * min(1.0, max(0, it - warm) / max(1, args.iters - warm))))
    for g in opt.param_groups:
        g["lr"] = g["base_lr"] * (min(1.0, it / 500) if g["is_head"] else bb_f) * (0.1 + 0.9 * cos)
    mom = 1 - (1 - 0.994) * (math.cos(math.pi * it / args.iters) + 1) / 2
    t_temp = 0.04 + (0.07 - 0.04) * min(1.0, it / 3000)
    return mom, t_temp

def koleo(x):
    x = F.normalize(x, dim=-1); d = torch.cdist(x, x) + torch.eye(len(x), device=x.device) * 1e4
    return -torch.log(d.min(1).values + 1e-8).mean()

def gram(p):
    p = F.normalize(p.float(), dim=-1); return p @ p.transpose(1, 2)

# ----------------------------------------------------------------------------------------- loop
ds = NaipWindows() if args.source == "naip" else GoogleWindows()
dl = iter(torch.utils.data.DataLoader(ds, batch_size=args.bs, num_workers=args.workers, collate_fn=collate, pin_memory=True, persistent_workers=True))
os.makedirs(args.out, exist_ok=True); log = open(f"{args.out}/log.txt", "a") if rank == 0 else None
t0 = time.time()
for it in range(args.iters + 1):
    mom, t_temp = sched(it)
    raw = next(dl).to(dev, non_blocking=True).float() / 255.0
    with torch.no_grad():
        g = [normalize(photometric(rrc(raw, GLOBAL, (0.35, 1.0)), 0.5 if k == 0 else 0.1), C) for k in range(2)]
        l = [normalize(photometric(rrc(raw, LOCAL, (0.08, 0.35)), 0.5), C) for _ in range(args.n_local)]
        h = GLOBAL // 16; masks = [ibot_masks(raw.shape[0], h, h) for _ in range(2)]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            t_out = [forward_tokens(teacher.bb, x) for x in g]
            t_cls = [F.softmax((teacher.dino(c).float() - center_cls) / t_temp, -1) for c, _ in t_out]
            t_pat = [F.softmax((teacher.ibot(p[m.flatten(1)]).float() - center_patch) / t_temp, -1) for (_, p), m in zip(t_out, masks)]
            a_pat = [forward_tokens(anchor, x[:, :3] if C == 4 else x)[1] for x in g]
        s_g, s_cls_logits, s_pat_logits = student_ddp(g, masks, l)
        s_cls_all = [c.float() / 0.1 for c in s_cls_logits]
        # DINO: every student view vs every *other* teacher global view
        dino = 0; n = 0
        for ti, tc in enumerate(t_cls):
            for si, sc in enumerate(s_cls_all):
                if si == ti: continue
                dino = dino + torch.sum(-tc * F.log_softmax(sc, -1), -1).mean(); n += 1
        dino = dino / n
        ibot = sum(torch.sum(-tp * F.log_softmax(sp.float() / 0.1, -1), -1).mean()
                   for sp, tp in zip(s_pat_logits, t_pat) if sp is not None) / 2
        ko = sum(koleo(c.float()) for c, _ in s_g) / 2
        # Gram anchoring on unmasked samples
        gr = 0
        for (_, p), m, ap_ in zip(s_g, masks, a_pat):
            keep = ~m.flatten(1).any(1)
            if keep.any(): gr = gr + F.mse_loss(gram(p[keep]), gram(ap_[keep]))
        loss = dino + ibot + 0.1 * ko + args.gram_w * gr * 100
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(student.parameters(), 3.0); opt.step()
    with torch.no_grad():
        for ps, pt in zip(student.parameters(), teacher.parameters()): pt.mul_(mom).add_(ps.detach(), alpha=1 - mom)
        bc = torch.cat([teacher.dino(c.float()) for c, _ in t_out]).float().mean(0, keepdim=True)
        bp = torch.cat([teacher.ibot(p[m.flatten(1)].float()) for (_, p), m in zip(t_out, masks) if m.any()]).float().mean(0, keepdim=True)
        dist.all_reduce(bc); dist.all_reduce(bp); center_cls.mul_(0.9).add_(bc / world, alpha=0.1); center_patch.mul_(0.9).add_(bp / world, alpha=0.1)
    if rank == 0 and it % 50 == 0:
        with torch.no_grad():
            ent = -(t_cls[0] * torch.log(t_cls[0] + 1e-8)).sum(-1).mean().item()
            fstd = F.normalize(t_out[0][0].float(), dim=-1).std(0).mean().item()
        msg = (f"it {it} loss {loss.item():.4f} dino {dino.item():.4f} ibot {float(ibot):.4f} koleo {ko.item():.3f} gram {float(gr) * 100:.4f} "
               f"t_ent {ent:.2f} feat_std {fstd:.4f} mom {mom:.4f} {(time.time() - t0) / max(it, 1):.2f}s/it")
        print(msg, flush=True); log.write(msg + "\n"); log.flush()
    if rank == 0 and it > 0 and (it % 2000 == 0 or it == args.iters):
        torch.save({"teacher_bb": teacher.bb.state_dict(), "student_bb": student.bb.state_dict(), "it": it, "in_chans": C, "args": vars(args)}, f"{args.out}/ckpt.pt")
dist.destroy_process_group()
