#!/bin/bash
# Launch downstream experiments as soon as the SSL pretraining runs finish.
cd "$(dirname "$0")/.."; PY=${PY:-python}
until grep -q "^it 7000 " runs/ssl/naip/log.txt && grep -q "^it 7000 " runs/ssl/google/log.txt; do sleep 60; done
while pgrep -f "ssl_pretrain.py" > /dev/null; do sleep 20; done      # wait for final checkpoint write + exit
echo "ssl done $(date)"
$PY scripts/ssl_seg_train.py --init runs/ssl/naip/ckpt.pt --bands 4 --out runs/seg_sslnaip_rgbn --gpu 0 > logs/seg_sslnaip_rgbn.log 2>&1 &
$PY scripts/ssl_tankdet_train.py --init runs/ssl/google/ckpt.pt --out runs/tankdet_sslgoogle --gpu 1 > logs/tankdet_sslgoogle.log 2>&1 &
$PY scripts/ssl_tank_cls_cv.py --methods ssl_lin,ssl_ft --ssl runs/ssl/google/ckpt.pt --gpu 2 > logs/ssl_tank_cls_cv_ssl.log 2>&1 &
( $PY scripts/04_train.py --bands 4 --epochs 20 --iters 400 --frac 0.1 --out runs/unet_rgbn_frac0.1 --gpu 3 > logs/unet_frac0.1.log 2>&1
  $PY scripts/ssl_seg_train.py --init runs/ssl/naip/ckpt.pt --bands 4 --frac 0.1 --out runs/seg_sslnaip_rgbn_frac0.1 --gpu 3 > logs/seg_sslnaip_frac0.1.log 2>&1
  $PY scripts/ssl_seg_train.py --init dinov3sat --bands 4 --frac 0.1 --out runs/seg_dinov3sat_rgbn_frac0.1 --gpu 3 > logs/seg_dinov3sat_frac0.1.log 2>&1 ) &
wait; echo "followups done $(date)"
