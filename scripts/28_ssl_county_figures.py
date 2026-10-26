"""County maps/statistics from the self-supervised production models (pond ensemble inventory + ViT tank detections)."""
import json, numpy as np, pandas as pd, geopandas as gpd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
from pyproj import Transformer
for f in ("NotoSansCJK-Regular.ttc", "NotoSansCJK-Bold.ttc"): fm.fontManager.addfont(f"/usr/share/fonts/opentype/noto/{f}")
plt.rcParams.update({"font.family": "Noto Sans CJK SC", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 10, "axes.titleweight": "bold", "savefig.dpi": 200, "savefig.bbox": "tight"})
OUT = "report/pdf"; INK, WATER, DRY, NEW, MIX, TANK = "#1b2320", "#0d6f6a", "#a8661d", "#3a62b0", "#7d6f86", "#b4307a"
T = Transformer.from_crs(4326, 3857, always_xy=True)
ov = plt.imread("report/naip2022_overview.jpg"); bb = json.load(open("report/overview_bounds.json"))["bounds"]
(x0, y0), (x1, y1) = T.transform(bb[0][1], bb[0][0]), T.transform(bb[1][1], bb[1][0])
cty = gpd.read_file("data/vector/hamilton_county.geojson").to_crs(3857); C = cty.geometry.iloc[0]
def basemap(ax, a=.55):
    ax.imshow(ov, extent=[x0, x1, y0, y1], alpha=a); cty.boundary.plot(ax=ax, color=INK, lw=1, ls="--")
    b = cty.total_bounds; ax.set_xlim(b[0] - 1500, b[2] + 1500); ax.set_ylim(b[1] - 1500, b[3] + 1500); ax.set_axis_off()
S = {}
fp = gpd.read_file("outputs/vector/pond_inventory_multiyear_ssl.gpkg").to_crs(3857); fp = fp[fp.centroid.within(C)]
old = gpd.read_file("outputs/vector/pond_inventory_multiyear.gpkg").to_crs(3857); old = old[old.centroid.within(C)]
for nm, d in (("ssl", fp), ("old", old)):
    S[nm] = dict(footprints=len(d), trend=d.trend.value_counts().to_dict(), wet={y: int(d[f"wet_{y}"].astype(bool).sum()) for y in ("2018", "2020", "2022")},
                 water_km2={y: round(d[f"water_m2_{y}"].sum() / 1e6, 2) for y in ("2018", "2020", "2022")})
fig, ax = plt.subplots(figsize=(6.4, 6.4)); basemap(ax)
TR = {"persistent": ("三年都有水", WATER), "dried_by_2022": ("2022 年已干", DRY), "intermittent": ("间歇", MIX), "new_or_refilled": ("新蓄水/重新蓄水", NEW)}
for k, (lab, c) in TR.items():
    s = fp[fp.trend == k]; ax.scatter(s.centroid.x, s.centroid.y, s=np.clip(np.sqrt(s.fp_area_m2) / 6, 1.5, 30), c=c, alpha=.8, lw=0, label=f"{lab}（{len(s):,}）")
ax.legend(loc="upper left", fontsize=7.5, framealpha=.9, markerscale=2.2, title="2018 → 2022 水情（自监督集成模型）", title_fontsize=8)
plt.savefig(f"{OUT}/s5_pond_trend_map_ssl.png"); plt.close()
tk = gpd.read_file("outputs/vector/round_tanks_hamilton_ssl.geojson").to_crs(3857)
S["tanks"] = dict(n=len(tk), diam_median=round(float(tk.diam_m.median()), 1))
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.9), gridspec_kw=dict(width_ratios=[1.35, 1]))
basemap(axs[0]); axs[0].scatter(tk.geometry.x, tk.geometry.y, s=4, marker="D", c=TANK, lw=0, alpha=.85); axs[0].set_title(f"全县圆形水槽/储水罐（{len(tk):,} 个，自监督模型）")
axs[1].hist(tk.diam_m.clip(upper=14), bins=np.arange(1.5, 14.5, .5), color=TANK, alpha=.85); axs[1].set_xlabel("估计直径 (m)"); axs[1].set_ylabel("数量")
axs[1].axvline(tk.diam_m.median(), color=INK, lw=1, ls="--"); axs[1].text(tk.diam_m.median() + .3, axs[1].get_ylim()[1] * .9, f"中位数 {tk.diam_m.median():.1f} m", fontsize=7.5)
axs[1].set_title("水槽直径分布"); axs[1].grid(axis="y", alpha=.25)
plt.tight_layout(); plt.savefig(f"{OUT}/s6_tank_map_ssl.png"); plt.close()
json.dump(S, open("outputs/summary_hamilton_ssl.json", "w"), indent=1); print(json.dumps(S, indent=1))
