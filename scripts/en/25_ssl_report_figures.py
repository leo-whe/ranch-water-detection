"""Figures for the self-supervised learning chapter of the PDF report (reads whatever results exist)."""
import os, re, json, glob, sys, numpy as np, pandas as pd, rasterio, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch
from scipy import ndimage as ndi
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..')); from common import tile_split
for f in ("NotoSansCJK-Regular.ttc", "NotoSansCJK-Bold.ttc"): fm.fontManager.addfont(f"/usr/share/fonts/opentype/noto/{f}")
plt.rcParams.update({"font.family": "Noto Sans CJK SC", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 10, "axes.titleweight": "bold", "savefig.dpi": 200, "savefig.bbox": "tight"})
OUT = "report/pdf_en"; INK, WATER, TANK, GREY, SSL, PUB = "#1b2320", "#0d6f6a", "#b4307a", "#9aa9a2", "#3a62b0", "#7fa3d6"

# S1: SSL method schematic
fig, ax = plt.subplots(figsize=(7.2, 3.1)); ax.set_xlim(-2, 102); ax.set_ylim(0, 44); ax.axis("off")
def B(x, y, w, h, t, c, fs=6.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=c, ec=INK, lw=.6)); ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs * 0.9, linespacing=1.4)
def A(x0, y0, x1, y1): ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=INK, lw=.8))
B(0, 16, 14, 12, "Unlabeled\ncounty imagery\nNAIP RGB+NIR\nor Google\n0.25 m", "#eef3f1")
B(18, 27, 15, 11, "2 global views\n224 px, some\npatches masked\n(iBOT)", "#dcebe9"); B(18, 6, 15, 11, "6 local views\n96 px", "#dcebe9")
A(14.5, 24, 17.5, 31); A(14.5, 20, 17.5, 12)
B(38, 25, 18, 15, "Student ViT-L/16\n300 M params\nDINOv3-sat init\n(NIR filters zero-init)", "#dbe4f5")
B(38, 3, 18, 13, "Teacher\n= EMA of student\nno backprop", "#e9eef8")
A(33.5, 33, 37.5, 33); A(33.5, 11, 37.5, 29); A(33.5, 30, 37.5, 12)
B(61, 3, 21, 37, "Objectives\n\nDINO: cross-view [CLS]\nself-distillation\n(16,384 prototypes)\n\niBOT: predict teacher\nfeatures of\nmasked patches\n\nGram anchoring: keep\npatch-similarity\nstructure of the\noriginal model\n\nKoLeo: spread features", "#f3efe2", fs=6.4)
A(56.5, 32, 60.5, 28); A(56.5, 10, 60.5, 14)
B(86, 15, 14, 14, "Domain-adapted\nbackbone\n→ fine-tuning", "#e3f0e0"); A(82.5, 22, 85.5, 22)
plt.savefig(f"{OUT}/s1_ssl_method.png"); plt.close()

# S2: pond recall by size for each model (weak labels, test area)
models = [("U-Net v2\n(ImageNet CNN)", "outputs/prob/2022", GREY), ("DINOv3-sat 1×", "outputs/prob_ssl/dinov3sat_rgbn", PUB),
          ("DINOv3-sat 2×", "outputs/prob_ssl/dinov3sat_rgbn_x2", SSL), ("Domain SSL 2×", "outputs/prob_ssl/sslnaip_rgbn_x2", WATER)]
models = [m for m in models if os.path.isdir(m[1]) and len(glob.glob(m[1] + "/*.tif")) >= 16]
bins = [25, 50, 100, 200, 500, 2000, 1e9]; labels = ["25–50", "50–100", "100–200", "200–500", "500–2k", ">2k"]
rec = {}
for name, d, _ in models:
    hits = np.zeros(6); tot = np.zeros(6)
    for t in tile_split()["test"]:
        y = rasterio.open(t.replace("naip", "labels")).read(1); p = rasterio.open(f"{d}/{os.path.basename(t)}").read(1) > 127
        lg, ng = ndi.label(y == 1); idx = np.arange(1, ng + 1); sz = ndi.sum(np.ones_like(lg), lg, idx) * 0.36; hit = ndi.mean(p, lg, idx) >= 0.3
        b = np.digitize(sz, bins) - 1; ok = (b >= 0) & (b < 6); np.add.at(tot, b[ok], 1); np.add.at(hits, b[ok], hit[ok])
    rec[name] = hits / np.maximum(tot, 1)
fig, ax = plt.subplots(figsize=(7.2, 2.6)); w = 0.8 / max(1, len(models)); x = np.arange(6)
for i, (name, _, c) in enumerate(models): ax.bar(x + (i - (len(models) - 1) / 2) * w, rec[name], w * 0.95, color=c, label=name.replace("\n", " "))
ax.set_xticks(x); ax.set_xticklabels([f"{l} m²\n(n={int(n)})" for l, n in zip(labels, tot)]); ax.set_ylim(0, 1.05)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1)); ax.set_ylabel("Object recall (vs weak labels)"); ax.grid(axis="y", alpha=.25)
ax.legend(fontsize=7, frameon=False, ncol=len(models), loc="upper center", bbox_to_anchor=(0.5, -0.22)); ax.set_title("Test-area recall by pond size")
plt.savefig(f"{OUT}/s2_pond_size_recall.png"); plt.close(); json.dump({k: v.tolist() for k, v in rec.items()}, open("outputs/eval/ssl/size_recall.json", "w"))

# S3: tank-type classifier label efficiency
cv = json.load(open("outputs/eval/ssl_tank_cls_cv.json"))
series = [("resnet18_in", "ResNet18 (ImageNet supervised)", GREY, "o"), ("dinov3_lin", "DINOv3-sat frozen + linear", PUB, "s"),
          ("dinov3_ft", "DINOv3-sat fine-tuned", SSL, "s"), ("ssl_lin", "Domain SSL frozen + linear", "#8ccfc6", "D"), ("ssl_ft", "Domain SSL fine-tuned", WATER, "D")]
fig, ax = plt.subplots(figsize=(5.2, 2.9))
for k, name, c, mk in series:
    xs = [f for f in (0.25, 0.5, 1.0) if f"{k}@{f}" in cv]
    if xs: ax.plot([f * 100 for f in xs], [cv[f"{k}@{f}"]["water_f1"] for f in xs], marker=mk, color=c, label=name, lw=1.6)
ax.set_xticks([25, 50, 100]); ax.set_xticklabels(["25%\n(~89)", "50%\n(~177)", "100%\n(~354)"]); ax.set_xlabel("Training labels (per fold)")
ax.set_ylabel("Water-facility F1 (5-fold CV)"); ax.grid(alpha=.25); ax.legend(fontsize=7, frameon=False, loc="lower right"); ax.set_title("Tank-type classifier: label efficiency")
plt.savefig(f"{OUT}/s3_tank_cls_label_eff.png"); plt.close()

# S4: tank system precision-recall operating points
fig, ax = plt.subplots(figsize=(5.2, 3.0))
y_ = json.load(open("outputs/eval/tank_2stage_eval.json"))["operating_points"]["conf>=0.2,p_water>=0.8"]
ax.errorbar([y_["recall"]], [(y_["precision_strict"] + y_["precision_lenient"]) / 2], yerr=[[(y_["precision_lenient"] - y_["precision_strict"]) / 2]], fmt="o", color=GREY, capsize=3, label="YOLO11s + ResNet18 (original)")
for f, name, c in (("outputs/eval/ssl/tankdet_dinov3sat_v2_2stage.json", "DINOv3-sat dense detector + ViT classifier", SSL), ("outputs/eval/ssl/tankdet_sslgoogle_2stage.json", "Domain SSL (Google) dense detector + classifier", WATER)):
    if not os.path.exists(f): continue
    d = json.load(open(f)); ks = sorted(d, key=float, reverse=True)
    r = [d[k]["recall"] for k in ks]; pm = [(d[k]["precision_strict"] + d[k]["precision_lenient"]) / 2 for k in ks]; pe = [(d[k]["precision_lenient"] - d[k]["precision_strict"]) / 2 for k in ks]
    ax.errorbar(r, pm, yerr=pe, fmt="-o", color=c, capsize=3, label=name, ms=4)
    seen = set()
    for k, rr, pp in zip(ks, r, pm):
        if rr in seen: continue
        seen.add(rr); ax.annotate(f"p≥{k}", (rr, pp), textcoords="offset points", xytext=(5, 5), fontsize=6.5, color=c)
ax.set_xlabel("Recall (18 reference facilities)"); ax.set_ylabel("Precision (mid-point of strict–lenient)"); ax.set_xlim(0.6, 1.0); ax.set_ylim(0.3, 1.0)
ax.grid(alpha=.25); ax.legend(fontsize=7, frameon=False, loc="lower left"); ax.set_title("Test-area tank detection: precision–recall")
plt.savefig(f"{OUT}/s4_tank_pr.png"); plt.close()
print("ssl figures done:", [os.path.basename(p) for p in sorted(glob.glob(f"{OUT}/s*.png"))])
