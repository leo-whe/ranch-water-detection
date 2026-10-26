"""Figures for the PDF report (matplotlib, Noto Sans CJK)."""
import json, glob, os, re, numpy as np, pandas as pd, geopandas as gpd, rasterio, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.lines import Line2D
from shapely.geometry import box
from pyproj import Transformer
fm.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
fm.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
plt.rcParams.update({"font.family": "Noto Sans CJK SC", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 10, "axes.titleweight": "bold", "savefig.dpi": 200, "savefig.bbox": "tight"})
OUT = "report/pdf_en"; WATER, DRY, NEW, MIX, TANK, INK = "#0d6f6a", "#a8661d", "#3a62b0", "#7d6f86", "#b4307a", "#1b2320"
T = Transformer.from_crs(4326, 3857, always_xy=True)
ov = plt.imread("report/naip2022_overview.jpg"); bb = json.load(open("report/overview_bounds.json"))["bounds"]
(x0, y0), (x1, y1) = T.transform(bb[0][1], bb[0][0]), T.transform(bb[1][1], bb[1][0])
cty = gpd.read_file("data/vector/hamilton_county.geojson").to_crs(3857)
def basemap(ax, alpha=0.8):
    ax.imshow(ov, extent=[x0, x1, y0, y1], alpha=alpha); cty.boundary.plot(ax=ax, color=INK, lw=1.0, ls="--")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel(""); [s.set_visible(False) for s in ax.spines.values()]
    b = cty.total_bounds; ax.set_xlim(b[0] - 1500, b[2] + 1500); ax.set_ylim(b[1] - 1500, b[3] + 1500)
    # 10 km scale bar (web mercator scale ~1/cos(lat))
    k = 1 / np.cos(np.radians(31.7)); sx, sy = b[0] + 2000, b[1] + 2500
    ax.plot([sx, sx + 10000 * k], [sy, sy], color="white", lw=4); ax.plot([sx, sx + 10000 * k], [sy, sy], color=INK, lw=2)
    ax.text(sx + 5000 * k, sy + 1200, "10 km", ha="center", fontsize=8, color=INK, bbox=dict(fc="white", ec="none", alpha=.7, pad=1))

# F1 study area + split + wells
sp = json.load(open("data/split.json")); fig, ax = plt.subplots(figsize=(6.2, 6.2)); basemap(ax, .9)
col = {"train": "#e8e2d0", "val": "#3a62b0", "test": "#b0412c"}
for t in glob.glob("data/naip/2022/*.tif"):
    k = "_".join(os.path.basename(t).split("_")[2:4]); s = next(n for n, v in sp.items() if k in v)
    g = gpd.GeoSeries([box(*rasterio.open(t).bounds)], crs=26914).to_crs(3857)
    g.boundary.plot(ax=ax, color=col[s], lw=1.6 if s != "train" else .6, alpha=1 if s != "train" else .9)
w = gpd.read_file("data/vector/twdb_wells_hamilton.gpkg"); w = w[w.use == "Stock"].to_crs(3857)
w.plot(ax=ax, color="#f5c542", markersize=5, edgecolor=INK, lw=.3)
ax.legend(handles=[Line2D([], [], color="#b0412c", lw=2, label="Test area (16 tiles, ~650 km²)"), Line2D([], [], color="#3a62b0", lw=2, label="Validation (8 tiles)"),
                   Line2D([], [], color="#e8e2d0", lw=1.5, label="Training (53 tiles)"), Line2D([], [], marker="o", ls="", color="#f5c542", mec=INK, label="TWDB livestock wells (292)"),
                   Line2D([], [], color=INK, ls="--", label="Hamilton County boundary")], loc="upper left", fontsize=7.5, framealpha=.9)
ax.set_xlabel(""); ax.set_ylabel(""); plt.savefig(f"{OUT}/f1_study_area.png"); plt.close()

# F2 pipeline
fig, ax = plt.subplots(figsize=(7.2, 3.9)); ax.set_xlim(0, 100); ax.set_ylim(0, 56); ax.axis("off")
def B(x, y, w, h, t, c):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=c, ec=INK, lw=.6)); ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=6.0, linespacing=1.4)
def A(x0_, y0_, x1_, y1_): ax.annotate("", (x1_, y1_), (x0_, y0_), arrowprops=dict(arrowstyle="-|>", color=INK, lw=.8))
ax.text(1, 53, "Pond branch (public data)", fontsize=9, fontweight="bold", color=WATER); ax.text(1, 24.5, "Round-tank branch (Google satellite)", fontsize=9, fontweight="bold", color=TANK)
c1, c2 = "#dcebe9", "#f3dbe7"
B(1, 36, 17, 13, "NAIP aerial imagery\n2018 / 2020 / 2022\n0.6 m · RGB+NIR\n77 tiles × 3 years", c1); B(23, 36, 17, 13, "Weak labels\nNWI pond locations\n+ NIR spectral water\nroads, roofs = neg.", c1)
B(45, 36, 16, 13, "U-Net\nEfficientNet-B4\n4-band input", c1); B(66, 36, 15, 13, "Inference +\nvectorisation\nradiometric match\n(2018/2020)", c1); B(85, 36, 14, 13, "Pond inventory\nwater area\nper year\nwet / dry", c1)
for a_, b_ in ((18.8, 23), (40.8, 45), (61.8, 66), (81.8, 85)): A(a_, 42.5, b_, 42.5)
B(1, 6, 17, 14, "Google satellite\nz19 ≈0.25 m\n520 k county tiles\n+ TWDB wells", c2); B(23, 6, 17, 14, "Manual labels\n203 tank boxes\n2 active-learning\nrounds", c2)
B(45, 6, 16, 14, "YOLO11s\ncandidate detection\n(recall first)", c2); B(66, 6, 15, 14, "z20 (0.13 m)\nResNet18, 4 classes\ntank / hay ring\nbale / other", c2); B(85, 6, 14, 14, "Tank inventory\n1,849\n(county)", c2)
for a_, b_ in ((18.8, 23), (40.8, 45), (61.8, 66), (81.8, 85)): A(a_, 13, b_, 13)
plt.savefig(f"{OUT}/f2_pipeline.png"); plt.close()

# F3 weak-label examples: RGB | CIR | label
from scipy import ndimage as ndi
rng = np.random.default_rng(4); ex = []
for t in rng.permutation(sorted(glob.glob("data/labels/2022/*.tif"))):
    l = rasterio.open(t).read(1); lab, n = ndi.label(l == 1)
    if n < 5: continue
    sz = ndi.sum(np.ones_like(lab), lab, np.arange(1, n + 1)); k = int(rng.choice(np.where((sz > 800) & (sz < 8000))[0])) + 1
    r, c = map(int, ndi.center_of_mass(lab == k)); s = rasterio.open(t.replace("labels", "naip"))
    w_ = rasterio.windows.Window(c - 160, r - 160, 320, 320); a = s.read(window=w_, boundless=True); y = rasterio.open(t).read(1, window=w_, boundless=True, fill_value=0)
    ex.append((a, y))
    if len(ex) == 3: break
fig, axs = plt.subplots(3, 3, figsize=(6.2, 6.2))
for i, (a, y) in enumerate(ex):
    rgb = a[:3].transpose(1, 2, 0); axs[i, 0].imshow(rgb); axs[i, 1].imshow(a[[3, 0, 1]].transpose(1, 2, 0))
    o = rgb.copy().astype(float); o[y == 1] = o[y == 1] * .3 + np.array([0, 110, 255]) * .7; o[y == 255] = o[y == 255] * .4 + np.array([240, 200, 0]) * .6
    axs[i, 2].imshow(o.astype(np.uint8))
for a_, t_ in zip(axs[0], ["NAIP true colour", "NAIP NIR false colour", "Weak label (blue = water, yellow = ignore)"]): a_.set_title(t_, fontsize=8.5)
for a_ in axs.flat: a_.axis("off")
plt.tight_layout(); plt.savefig(f"{OUT}/f3_labels.png"); plt.close()

# F4 training curves
def parse(f): return [(int(m.group(1)), float(m.group(2))) for m in re.finditer(r"^ep (\d+) loss [\d.]+ val_IoU ([\d.]+)", open(f).read(), re.M)]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6))
for f, lab_, c in (("logs/train_rgbn.log", "U-Net v1 (4-band)", "#9bb8b5"), ("logs/train_rgbn_v2.log", "U-Net v2 (4-band, final)", WATER), ("logs/train_rgb_v2.log", "U-Net v2 (RGB only)", DRY)):
    d = parse(f); axs[0].plot([e for e, _ in d], [v for _, v in d], label=lab_, color=c, lw=1.5)
axs[0].set_xlabel("epoch"); axs[0].set_ylabel("Validation IoU"); axs[0].set_title("Pond segmentation"); axs[0].legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2); axs[0].grid(alpha=.25)
off = 0
for v, c in (("tank_v1", "#e3b3cc"), ("tank_v2", "#cf6fa2"), ("tank_v3", TANK)):
    r = pd.read_csv(f"runs/yolo/{v}/results.csv"); r.columns = [x.strip() for x in r.columns]
    axs[1].plot(r.epoch + off, r["metrics/mAP50(B)"], color=c, lw=1.3, label=f"YOLO {v[-2:]}"); off += len(r)
axs[1].set_xlabel("Cumulative epoch (3 successive rounds)"); axs[1].set_ylabel("Validation mAP@0.5"); axs[1].set_title("Tank detector (YOLO11s)"); axs[1].legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3); axs[1].grid(alpha=.25)
plt.tight_layout(); plt.savefig(f"{OUT}/f4_training.png"); plt.close()

# F5 pond trend map
fp = gpd.read_file("outputs/vector/pond_inventory_multiyear.gpkg"); fp = fp.to_crs(3857); fp = fp[fp.centroid.within(cty.geometry.iloc[0])]
fig, ax = plt.subplots(figsize=(6.4, 6.4)); basemap(ax, .55)
TR = {"persistent": ("Wet all three years", WATER), "dried_by_2022": ("Dry by 2022", DRY), "intermittent": ("Intermittent", MIX), "new_or_refilled": ("New / refilled", NEW)}
for k, (lab_, c) in TR.items():
    s = fp[fp.trend == k]; ax.scatter(s.centroid.x, s.centroid.y, s=np.clip(np.sqrt(s.fp_area_m2) / 6, 1.5, 30), c=c, alpha=.8, lw=0, label=f"{lab_} ({len(s):,})")
ax.legend(loc="upper left", fontsize=7.5, framealpha=.9, markerscale=2.2, title="2018 → 2022 water status", title_fontsize=8)
plt.savefig(f"{OUT}/f5_pond_trend_map.png"); plt.close()

# F6 wet by size/year + totals
S = json.load(open("outputs/summary_hamilton.json"))["inventory"]; wb = S["wet_by_size"]; sizes = list(wb["2018"].keys())
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw=dict(width_ratios=[1.6, 1]))
xs = np.arange(len(sizes))
for j, (yr, a_) in enumerate((("2018", .4), ("2020", .7), ("2022", 1))):
    v = [wb[yr][s] for s in sizes]; axs[0].bar(xs + (j - 1) * .27, v, .25, color=WATER, alpha=a_, label=yr)
    if yr == "2022": [axs[0].text(x + .27, vv + .02, f"{vv:.0%}", ha="center", fontsize=7) for x, vv in zip(xs, v)]
axs[0].set_xticks(xs); axs[0].set_xticklabels([f"{s} m²" for s in sizes]); axs[0].set_ylim(0, 1.05); axs[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
axs[0].set_title("Share of ponds holding water (by size)"); axs[0].legend(fontsize=7, frameon=False, ncol=3, loc="upper left"); axs[0].grid(axis="y", alpha=.25)
yrs = ["2018", "2020", "2022"]; wet = [S["wet"][y] for y in yrs]; km = [S["water_km2"][y] for y in yrs]
axs[1].bar(yrs, wet, color=WATER, width=.55); [axs[1].text(i, v + 120, f"{v:,}", ha="center", fontsize=7.5) for i, v in enumerate(wet)]
ax2 = axs[1].twinx(); ax2.plot(yrs, km, color=DRY, marker="o", lw=1.5); ax2.set_ylabel("Pond open water (km²)", color=DRY); ax2.spines["top"].set_visible(False); ax2.set_ylim(0, 12)
axs[1].set_ylabel("Ponds holding water"); axs[1].set_title("Wet ponds and open water, county-wide"); axs[1].set_ylim(0, 7800)
plt.tight_layout(); plt.savefig(f"{OUT}/f6_wet_stats.png"); plt.close()

# F7 tank map
tk = gpd.read_file("outputs/vector/round_tanks_hamilton_final.geojson").to_crs(3857)
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.9), gridspec_kw=dict(width_ratios=[1.35, 1]))
basemap(axs[0], .55); axs[0].scatter(tk.geometry.x, tk.geometry.y, s=4, marker="D", c=TANK, lw=0, alpha=.85)
axs[0].set_title(f"Round tanks / troughs, county-wide ({len(tk):,})")
axs[1].hist(tk.diam_m.clip(upper=14), bins=np.arange(2, 14.5, .5), color=TANK, alpha=.85); axs[1].set_xlabel("Estimated diameter (m)"); axs[1].set_ylabel("Count")
axs[1].axvline(tk.diam_m.median(), color=INK, lw=1, ls="--"); axs[1].text(tk.diam_m.median() + .3, axs[1].get_ylim()[1] * .9, f"median {tk.diam_m.median():.1f} m", fontsize=7.5)
axs[1].set_title("Tank diameter distribution"); axs[1].grid(axis="y", alpha=.25)
plt.tight_layout(); plt.savefig(f"{OUT}/f7_tank_map.png"); plt.close()

# F8 tank-type composition of candidates (county) + single vs two-stage precision
ty = gpd.read_file("outputs/vector/round_tanks_county_v3_typed.gpkg").tank_type.value_counts()
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.4))
names = {"water": "Water facility", "other": "Other", "bale": "Round bale", "hay_ring": "Hay-ring feeder"}; cols = {"water": TANK, "other": "#9aa9a2", "bale": "#d9b77a", "hay_ring": DRY}
o = ["water", "bale", "hay_ring", "other"]; axs[0].barh([names[k] for k in o][::-1], [ty[k] for k in o][::-1], color=[cols[k] for k in o][::-1])
[axs[0].text(ty[k] + 40, i, f"{ty[k]:,}", va="center", fontsize=7.5) for i, k in enumerate(o[::-1])]
axs[0].set_title("z20 classes of county YOLO candidates (conf ≥ 0.2)"); axs[0].set_xlim(0, max(ty) * 1.2)
lab_ = ["Single-stage YOLO\n(high confidence)", "Two-stage\n(final)"]; strict = [0.29, 0.667]; len_ = [0.29, 0.848]
axs[1].bar(lab_, len_, color="#f3dbe7", width=.5, label="lenient"); axs[1].bar(lab_, strict, color=TANK, width=.5, label="strict")
[axs[1].text(i, v + .03, f"{s:.0%}–{v:.0%}" if v != s else f"{v:.0%}", ha="center", fontsize=7.5) for i, (s, v) in enumerate(zip(strict, len_))]
axs[1].set_ylim(0, 1); axs[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1)); axs[1].set_title("Test-area water-facility precision"); axs[1].legend(fontsize=7, frameon=False)
plt.tight_layout(); plt.savefig(f"{OUT}/f8_tank_types.png"); plt.close()
print("figures done")
