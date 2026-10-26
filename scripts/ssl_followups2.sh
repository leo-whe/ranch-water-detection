#!/bin/bash
cd "$(dirname "$0")/.."; PY=${PY:-python}
until grep -q "ssl done" logs/ssl_followups.log; do sleep 60; done
( until grep -q "ssl_ft@1.0" logs/ssl_tank_cls_cv_ssl.log 2>/dev/null || grep -q Traceback logs/ssl_tank_cls_cv_ssl.log 2>/dev/null; do sleep 60; done
  $PY scripts/ssl_seg_train.py --init runs/ssl/naip/ckpt.pt --bands 4 --vit_scale 2 --size 256 --bs 16 --out runs/seg_sslnaip_rgbn_x2 --gpu 2 > logs/seg_sslnaip_rgbn_x2.log 2>&1 ) &
( until grep -qE "^best|Traceback" logs/tankdet_sslgoogle.log 2>/dev/null; do sleep 60; done
  $PY scripts/ssl_tankdet_deploy.py runs/tankdet_sslgoogle/best.pt data/tank_chips/test_tile_range.json outputs/vector/tankdet_sslgoogle_test.gpkg 0/1 1 > logs/tankdet_sslgoogle_deploy_test.log 2>&1 ) &
wait; echo "followups2 done $(date)"
