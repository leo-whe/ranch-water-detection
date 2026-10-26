#!/bin/bash
# After 2022 finishes, split the remaining 2020 ViT inference over GPU 9 and GPU 1 (2018 already runs on GPU 3).
cd "$(dirname "$0")/.."; PY=${PY:-python}; export GDAL_CACHEMAX=256
CK="--ckpt runs/seg_dinov3sat_rgbn_x2/best.pt --bands 4 --vit_scale 2 --tiles all --year 2020 --skip_dir outputs/prob/2020 --out outputs/prob_vit/2020"
until [ $(ls outputs/prob_vit/2022/*.tif | wc -l) -ge 77 ] && [ -z "$(ps -eo args | grep -E '^/home.*ssl_seg_predict.py .*--year 2022 ')" ]; do sleep 60; done
for p in $(ps -eo pid,args | grep -E '^ *[0-9]+ /home.*ssl_seg_predict.py .*--year 2020 ' | awk '{print $1}'); do kill $p; done; sleep 5
$PY scripts/ssl_seg_predict.py $CK --shard 0/2 --gpu 9 > logs/county_vit_2020_s0.log 2>&1 &
$PY scripts/ssl_seg_predict.py $CK --shard 1/2 --gpu 1 > logs/county_vit_2020_s1.log 2>&1 &
wait; echo "2020 done $(date)"
