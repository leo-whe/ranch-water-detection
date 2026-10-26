#!/bin/bash
# Production pond products from the ensemble (U-Net v2 + DINOv3-sat ViT-L 2x): ensemble probs -> vectors -> multi-year inventory.
cd "$(dirname "$0")/.."; PY=${PY:-python}; export GDAL_CACHEMAX=256
# ViT inference is launched separately (sharded across free GPUs); this script only waits for complete outputs.
for Y in 2022 2020 2018; do
  until [ $(ls outputs/prob_vit/$Y/*.tif 2>/dev/null | wc -l) -ge 77 ] && [ -z "$(ps -eo args | grep -E "^/home.*ssl_seg_predict.py .*--year $Y " )" ]; do sleep 60; done
  mkdir -p outputs/prob_ens/$Y
  $PY - <<PYEOF
import glob, os, numpy as np, rasterio
for f in sorted(glob.glob("outputs/prob/$Y/*.tif")):
    b = os.path.basename(f); o = f"outputs/prob_ens/$Y/{b}"
    if os.path.exists(o): continue
    a = rasterio.open(f); v = rasterio.open(f"outputs/prob_vit/$Y/{b}")
    p = ((a.read(1).astype(np.float32) + v.read(1).astype(np.float32)) / 2).round().astype(np.uint8)
    with rasterio.open(o, "w", **a.profile) as d: d.write(p, 1)
print("ensembled $Y")
PYEOF
  $PY scripts/06_vectorize.py --year $Y --prob_dir outputs/prob_ens --tag _ssl > logs/vectorize_ssl_$Y.log 2>&1
  echo "year $Y vectorized $(date)"
done
TAG=_ssl $PY scripts/10_temporal.py > logs/temporal_ssl.log 2>&1
echo "production done $(date)"
