# Ranch Water-Source Detection

Detection of livestock water sources on ranches in Hamilton County, Texas, from public
aerial imagery: earthen stock ponds, and the small round man-made tanks and troughs that
cattle drink from. The county-wide inventory covers three NAIP years (2018, 2020, 2022),
so each pond footprint also gets a wet / dry history. This is the first step towards
longer-term "is there still water?" monitoring for individual ranches.

Two model generations are included:

* **Baseline**: a 4-band U-Net for ponds, and a YOLO + ResNet18 two-stage system for round tanks.
* **Self-supervised**: a DINOv3-sat ViT-L/16 backbone, further pretrained on unlabeled county
  imagery, used for pond segmentation, dense tank detection and tank-type classification.
  The production pond product is an ensemble of the U-Net and the ViT model.

Reports: `report/Hamilton_ranch_water_report.pdf` (Chinese) and
`report/Hamilton_ranch_water_report_EN.pdf` (English). `report/web/index.html` is an
interactive map viewer of the county results (open it from a local web server).

## Data (all public)

| What | Source | Notes |
|---|---|---|
| Imagery | USDA NAIP 2018 / 2020 / 2022, 0.6 m, RGB+NIR, via Microsoft Planetary Computer | 77 quarter-quads cover the county. 2022 = July (drought), 2020 = Aug-Dec, 2018 = Oct-Nov (very wet autumn) |
| Weak labels | USFWS National Wetlands Inventory, 5,497 "Freshwater Pond" polygons | often offset 20-50 m and many are dry; used only as a location prior |
| Hard negatives | Census TIGER 2023 roads, Microsoft building footprints | |
| Tank prior | TWDB water-well databases (SDR + GWDB), 292 wells with use = "Stock" | only coordinates and use type are used |
| Round-tank imagery | Google satellite XYZ tiles, z19 (~0.25 m) and z20 (~0.13 m) | NAIP is too coarse for 3-6 m tanks. Capture dates are unknown, so these tiles locate facilities but cannot say whether a tank holds water on a given date |
| Visual QA | Esri World Imagery (WorldView-2, Dec 2021) | review chips only |

Spatial split (`data/split.json`): test = quads 3109823/24/31/32 (16 tiles), val = 3109815 + 3109717,
train = the rest. All manual labels and review records are in `data/manual/`.

## Results

Pond models, held-out test area, NAIP 2022 (`outputs/eval/ssl/summary.md` has the full table):

| Model | Pixel IoU | Object P | Object R | Object F1 | Recall on 42 verified wet NWI ponds |
|---|---|---|---|---|---|
| Spectral rule baseline | | | | 0.21 | |
| U-Net v2 (EfficientNet-B4, 4-band) | 0.829 | 0.915 | 0.764 | 0.833 | 0.81 |
| DINOv3-sat ViT-L, 2x upsampled input | 0.898 | 0.932 | 0.764 | 0.840 | 0.76 |
| **Ensemble U-Net v2 + ViT-L 2x** (production) | **0.905** | 0.943 | 0.766 | **0.845** | 0.81 |

Manual checks of the U-Net: 60 random detections gave precision 88 % strict / 93 % lenient
(errors are roads and one dark roof). Of 60 random NWI ponds, 18 were dry in July 2022.
Misses are tiny water under tree shade and algae-covered ponds. ViT models at native
resolution (9.6 m tokens) miss ponds under 500 m2; feeding the ViT a 2x upsampled window fixes
this, which is why the 2x variant is used.

Round tanks, test area, against an independent reference set of 18 water facilities:

| System | Recall | Precision (strict / lenient) | Notes |
|---|---|---|---|
| LoG blobs + CNN on NAIP 0.6 m | | 12-33 % | not usable; superseded |
| YOLO11s (z19) + ResNet18 type classifier (z20) | 72 % | 67 % / 85 % | county total 1,849 water facilities |
| Dense ViT-L detector + ViT-L type classifier | 78 % | | ~30 % more true facilities than YOLO at the same recall; county total 3,026 |

The main confusers are round hay bales and hay-ring feeders: with the single-stage YOLO
alone only 29 % of high-confidence hits were water facilities, which is why the z20 type
classifier exists. Domain-adaptive SSL on county imagery gave only modest, inconsistent gains
over the public DINOv3-sat weights; its clearest benefit is label efficiency on the manually
labelled tank-type task (water-facility F1 at a quarter of the labels: 0.61 ResNet18 vs 0.71-0.76 ViT).

County inventory (`outputs/summary_hamilton*.json`, ensemble model): 7,707 pond footprints,
of which 6,735 held water in 2018, 4,829 in 2020 and 3,072 in July 2022. Open pond water fell
from 12.0 km2 to 4.6 km2 over the same period. Ponds under 500 m2 dried first.

## Shipped files

| Path | Content |
|---|---|
| `weights/unet_rgbn_v2.pt` | pond U-Net, 4-band NAIP input (production baseline) |
| `weights/unet_rgb_v2.pt` | pond U-Net, RGB only, for Google / Esri type imagery (not separately evaluated) |
| `weights/tank_yolo_v3.pt` | round-tank YOLO11s detector on z19 tiles, after two active-learning rounds |
| `weights/tank_type_cls.pt` | ResNet18 tank-type classifier on z20 chips: water / hay_ring / bale / other |
| `outputs/pond_inventory_multiyear_ssl_hamilton.csv` | per-footprint water area and wet / dry status for each year (ensemble model, recommended) |
| `outputs/pond_inventory_multiyear_hamilton.csv` | the same from the U-Net alone |
| `outputs/vector/round_tanks_hamilton_ssl.geojson` | 3,026 water tanks / troughs county-wide (ViT system, recommended) |
| `outputs/vector/round_tanks_hamilton_final.geojson` | 1,849 water tanks / troughs (YOLO system) |
| `outputs/eval/` | weak-label metrics and every manual verification record |
| `data/manual/` | tank boxes, z20 type labels, active-learning review lists |

The ViT-L checkpoints (1.2 GB each) and the SSL pretraining checkpoints (2.4 GB each) are not
in the repository. Rasters, per-year vector layers and the Google tile cache are also left out.

## Pipeline

Scripts assume the repository root as working directory and the layout `data/`, `runs/`,
`outputs/`, `logs/`. Scripts that load a checkpoint take it as an argument or default to the
`runs/` path the model was trained to; copy or symlink the shipped weights there, e.g.
`runs/unet_rgbn_v2/best.pt`, `runs/yolo/tank_v3/weights/best.pt`, `runs/tank_type_cls.pt`.

Ponds on NAIP:

1. `01_fetch_nwi.py`, `02_fetch_naip.py` (`NAIP_YEAR=2018|2020|2022`) download the inputs.
2. `03_make_labels.py` and `03b_ignore_nwi_interior.py` build weak labels: spectral water
   components (NIR < 0.8 R and NIR < 0.8 G) near NWI ponds, or compact and clearly watery, are
   positives; roads, roofs, elongated features and dark tilled soil are negatives; everything
   uncertain is ignored.
3. `04_train.py` trains the U-Net with positive / hard-negative / random patch sampling.
4. `05_predict.py` runs sliding-window inference to uint8 probability GeoTIFFs. Years other
   than 2022 are radiometrically matched to the 2022 band statistics (`--match`).
5. `06_vectorize.py` turns probabilities into instances with area, roundness, a class
   (`pond`, `creek_or_river`, `small_round_water`) and flags (`likely_false_positive`,
   `near_building`, `possible_pool`).
6. `07_eval_pseudo.py` scores the test area against weak labels; `review_chips.py`,
   `review_points.py` render montages for the manual checks.
7. `10_temporal.py` builds the multi-year footprint inventory and wet / dry table.

Round tanks on Google tiles (`webtiles.py` is the tile fetcher and cache):

8. `11_well_chips.py`, `12_candidate_review_gmap.py`, `13_snap_boxes.py`, `label_grid.py`
   are the labelling helpers; `15_build_yolo_dataset.py` and `17_add_reviewed_dets.py`
   build the YOLO data and add the two active-learning rounds.
9. `14_fetch_test_tiles.py` / `19_fetch_county_tiles.py` download z19 tiles;
   `16_deploy_tanks.py` runs the sharded sliding-window YOLO; `23_merge_county_tanks.py`
   re-runs NMS across shards.
10. `20_z20_review.py` renders z20 chips for the four-way review, `21_tank_type_classifier.py`
    trains the type classifier, `22_score_tank_types.py` scores detections, and
    `18_eval_tanks.py` evaluates recall and precision on the test area.
11. `08_tank_candidates_county.py`, `09_tank_classifier.py`, `tank_blobs.py`,
    `tank_candidates.py` are the earlier NAIP-only tank attempt, kept for reference.

Self-supervised models (`ssl_*.py`):

* `ssl_backbone.py` loads DINOv3-sat ViT-L/16 (timm `vit_large_patch16_dinov3.sat493m`),
  widens the patch embedding to RGB+NIR with zero-initialised NIR filters, and supports
  iBOT-style masked forwards.
* `ssl_pretrain.py` is DDP continued pretraining on unlabeled county imagery: DINO + iBOT +
  KoLeo + Gram anchoring to the frozen original weights, EMA teacher. Two runs were made,
  NAIP RGB+NIR (6,000 iterations) and Google z19 RGB (7,000 iterations), each about 4 h on
  2x A100. Set `GDAL_CACHEMAX=256` for the NAIP dataloaders or they get OOM-killed.
* `ssl_models.py`, `ssl_seg_train.py`, `ssl_seg_predict.py`, `ssl_seg_eval.py`: ViT-L with a
  multi-level decoder and a full-resolution CNN detail branch for ponds. `--vit_scale 2`
  upsamples the ViT input to 4.8 m tokens.
* `ssl_tankdet_data.py`, `ssl_tankdet_train.py`, `ssl_tankdet_deploy.py`: dense ViT tank
  detector producing a tank-disk probability map.
* `ssl_tank_cls_cv.py` compares backbones and label budgets under 5-fold CV;
  `ssl_tank_cls_final.py` trains the production ViT type classifier;
  `ssl_tank_2stage_eval.py`, `ssl_tank_eval.py` evaluate; `ssl_summary.py` collects results;
  `ssl_viz_pca.py` draws feature PCA maps.
* `26_ensemble_production.sh`, `27_shard_vit_years.sh`, `29_tank_production.sh` are the
  county production runs. County ViT inference skips 512-px windows with no water evidence
  (U-Net probability <= 0.05 and no NIR-water pixels). `ssl_followups*.sh` are the experiment
  launchers; they wait on log files and GPU slots specific to the machine they ran on.

Report figures: `24_pdf_figures.py`, `25_ssl_report_figures.py`, `28_ssl_county_figures.py`
(Chinese labels, Noto Sans CJK) and `scripts/en/` for the English versions, generated by
`make_en_figure_scripts.py`.

## Setup

```
pip install -r requirements.txt
```

GDAL / rasterio need a working GDAL install; a conda environment with `rasterio` and
`geopandas` from conda-forge is the easiest route. Training and inference were run on
NVIDIA A100s with bfloat16 autocast.

## Limitations and next steps

* Google imagery locates tanks but cannot say whether they hold water on a date, and its
  terms restrict bulk download. A licensed high-resolution source should replace it for a
  production service.
* NAIP is flown every two years, too slow for monitoring. Large ponds can be tracked with
  Sentinel-2 (10 m, 5-day) or PlanetScope (3 m, daily); tanks need 0.3 m or better imagery.
* The pond branch needs ranch boundaries to report per-ranch inventories, and about 200
  local pond labels would improve recall on algae-covered ponds.
