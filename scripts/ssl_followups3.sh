#!/bin/bash
# Downstream experiments after SSL pretraining (NAIP run ended at it 6000 after a dataloader OOM-kill; ckpt it 6000 is used).
cd "$(dirname "$0")/.."; PY=${PY:-python}; export GDAL_CACHEMAX=256
$PY scripts/ssl_seg_train.py --init runs/ssl/naip/ckpt.pt --bands 4 --out runs/seg_sslnaip_rgbn --gpu 0 > logs/seg_sslnaip_rgbn.log 2>&1 &
$PY scripts/ssl_seg_train.py --init runs/ssl/naip/ckpt.pt --bands 4 --vit_scale 2 --size 256 --bs 16 --out runs/seg_sslnaip_rgbn_x2 --gpu 1 > logs/seg_sslnaip_rgbn_x2.log 2>&1 &
until grep -q "^it 7000 " runs/ssl/google/log.txt; do sleep 60; done
while pgrep -f "ssl_pretrain.py" > /dev/null; do sleep 20; done
echo "google ssl done $(date)"
( $PY scripts/ssl_tankdet_train.py --init runs/ssl/google/ckpt.pt --out runs/tankdet_sslgoogle --gpu 2 > logs/tankdet_sslgoogle.log 2>&1
  $PY scripts/ssl_tankdet_deploy.py runs/tankdet_sslgoogle/best.pt data/tank_chips/test_tile_range.json outputs/vector/tankdet_sslgoogle_test.gpkg 0/1 2 > logs/tankdet_sslgoogle_deploy_test.log 2>&1 ) &
( $PY scripts/ssl_tank_cls_cv.py --methods ssl_lin,ssl_ft --ssl runs/ssl/google/ckpt.pt --gpu 3 > logs/ssl_tank_cls_cv_ssl.log 2>&1
  $PY scripts/04_train.py --bands 4 --epochs 20 --iters 400 --frac 0.1 --out runs/unet_rgbn_frac0.1 --gpu 3 > logs/unet_frac0.1.log 2>&1
  $PY scripts/ssl_seg_train.py --init runs/ssl/naip/ckpt.pt --bands 4 --vit_scale 2 --size 256 --bs 16 --frac 0.1 --out runs/seg_sslnaip_rgbn_x2_frac0.1 --gpu 3 > logs/seg_sslnaip_x2_frac0.1.log 2>&1 ) &
wait; echo "followups3 done $(date)"
