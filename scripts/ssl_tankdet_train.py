"""Dense round-tank detector on Google z19 crops: SSL ViT-L backbone + detail branch -> tank-disk probability map.
args: --init dinov3sat | runs/ssl/google/ckpt.pt"""
import os, sys, glob, argparse, numpy as np, torch, torch.nn.functional as Fn
from scipy import ndimage as ndi
sys.path.insert(0, os.path.dirname(__file__)); from ssl_models import ViTSeg, prep
ap = argparse.ArgumentParser(); ap.add_argument("--init", default="dinov3sat"); ap.add_argument("--out", required=True)
ap.add_argument("--iters", type=int, default=3000); ap.add_argument("--bs", type=int, default=16); ap.add_argument("--gpu", default="0")
args = ap.parse_args(); dev = f"cuda:{args.gpu}"; os.makedirs(args.out, exist_ok=True); torch.manual_seed(0); np.random.seed(0)
tr = [np.load(f) for f in sorted(glob.glob("data/tankdet/train_*.npy"))]; va = [np.load(f) for f in sorted(glob.glob("data/tankdet/val_*.npy"))]
is_pos = np.array([(a[3] == 1).any() for a in tr]); print(len(tr), "train crops", is_pos.sum(), "with tanks")
def batch(arrs, aug):
    out = []
    for a in arrs:
        if aug:
            a = np.rot90(a, np.random.randint(4), (1, 2))
            if np.random.rand() < .5: a = a[:, :, ::-1]
        out.append(np.ascontiguousarray(a))
    t = torch.from_numpy(np.stack(out)); x, y = t[:, :3].float(), t[:, 3].long()
    if aug: x = (x * torch.empty(len(x), 1, 1, 1).uniform_(.85, 1.15) + torch.empty(len(x), 3, 1, 1).uniform_(-10, 10)).clamp(0, 255)
    return x.to(dev), y.to(dev)
def loss_fn(logit, y):
    valid = (y != 255).float(); t = (y == 1).float(); logit = logit[:, 0]
    bce = (Fn.binary_cross_entropy_with_logits(logit, t, reduction="none", pos_weight=torch.tensor(5.0, device=dev)) * valid).sum() / valid.sum().clamp(min=1)
    p = torch.sigmoid(logit) * valid
    return bce + 1 - (2 * (p * t).sum() + 1) / (p.sum() + (t * valid).sum() + 1)
model = ViTSeg(3, args.init).to(dev)
def lid(n): return int(n.split(".")[2]) + 1 if n.startswith("bb.blocks.") else (0 if n.startswith("bb.") and not n.startswith("bb.norm") else 25)
groups = {}
for n, p in model.named_parameters(): groups.setdefault(3e-4 if not n.startswith("bb.") else 3e-5 * 0.9 ** (25 - lid(n)), []).append(p)
opt = torch.optim.AdamW([dict(params=v, lr=k) for k, v in groups.items()], weight_decay=0.05)
sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in opt.param_groups], total_steps=args.iters, pct_start=0.05)
def evaluate():
    model.eval(); tp = fn = fp = 0
    with torch.no_grad():
        for j in range(0, len(va), 16):
            x, y = batch(va[j:j + 16], False)
            with torch.autocast("cuda", dtype=torch.bfloat16): p = torch.sigmoid(model(prep(x, 3)).float())[:, 0].cpu().numpy()
            for pi, yi in zip(p, y.cpu().numpy()):
                gl, ng = ndi.label(yi == 1); pl, npd = ndi.label(pi > 0.5)
                for k in range(1, ng + 1): hit = (pi[gl == k] > 0.5).any(); tp += hit; fn += (not hit)
                for k in range(1, npd + 1):
                    reg = pl == k
                    if reg.sum() < 12: continue
                    if not (yi[reg] == 1).any() and not (yi[reg] == 255).mean() > 0.5: fp += 1
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1); return P, R, 2 * P * R / max(P + R, 1e-9)
best = -1; log = open(f"{args.out}/log.txt", "w")
for s in range(1, args.iters + 1):
    model.train()
    ids = np.concatenate([np.random.choice(np.where(is_pos)[0], args.bs // 2), np.random.choice(np.where(~is_pos)[0], args.bs - args.bs // 2)])
    x, y = batch([tr[i] for i in ids], True)
    with torch.autocast("cuda", dtype=torch.bfloat16): out = model(prep(x, 3))
    loss = loss_fn(out.float(), y); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
    if s % 200 == 0:
        P, R, F1 = evaluate(); msg = f"step {s} loss {loss.item():.4f} val_obj P {P:.3f} R {R:.3f} F1 {F1:.3f}"; print(msg, flush=True); log.write(msg + "\n"); log.flush()
        if F1 > best: best = F1; torch.save(model.state_dict(), f"{args.out}/best.pt")
print("best", best)
