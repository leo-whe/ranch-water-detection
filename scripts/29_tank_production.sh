#!/bin/bash
cd "$(dirname "$0")/.."; PY=${PY:-python}; export GDAL_CACHEMAX=256
until [ $(grep -lE "detections;|Traceback" logs/county_ssl_tank_s*.log 2>/dev/null | wc -l) -ge 8 ]; do sleep 60; done
grep -l Traceback logs/county_ssl_tank_s*.log && { echo "shard failed"; exit 1; }
$PY scripts/23_merge_county_tanks.py outputs/vector/county_ssl_shards outputs/vector/round_tanks_county_ssl_raw.gpkg
$PY scripts/ssl_tank_cls_final.py --init dinov3sat --load runs/tank_type_vit_dinov3.pt --score outputs/vector/round_tanks_county_ssl_raw.gpkg --out outputs/vector/round_tanks_county_ssl_typed.gpkg --gpu 2
$PY - <<'PYEOF'
import geopandas as gpd
g = gpd.read_file("outputs/vector/round_tanks_county_ssl_typed.gpkg"); g["water_source"] = (g.tank_type == "water") & (g.p_water >= 0.7)
g.to_file("outputs/vector/round_tanks_county_ssl_typed.gpkg")
w = g[g.water_source]; w.to_file("outputs/vector/round_tanks_hamilton_ssl.geojson", driver="GeoJSON"); w.drop(columns="geometry").to_csv("outputs/round_tanks_hamilton_ssl.csv", index=False)
print(len(g), "candidates;", len(w), "water sources (p_water>=0.7)"); print(g.tank_type.value_counts().to_dict())
PYEOF
echo "tank production done $(date)"
