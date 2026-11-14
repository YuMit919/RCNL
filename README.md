# RCNL: Reference-Certified Negative Learning

Code for *Out-of-Reference Is Not Out-of-Legend: Reference-Certified Negative Learning for Few-Shot Open-Legend Hyperspectral Mapping*.

With only a few references per class, unseen appearances of legend classes and land covers outside the legend both
look unfamiliar. RCNL uses the unlabeled pixels of the scene to learn rejection, but lets the references decide which
negative losses are safe:

1. **Reference support threshold.** References of each class are clustered by image coordinates and split into 5
   spatial folds. A reference model trained on the other folds scores each held-out reference; the 10 % quantile of the
   pooled true-class probabilities is a threshold *t* shared by all classes.
2. **Certification.** A pixel receives reference support for class *k* if the reference model gives it
   probability ≥ *t* for *k*. Only when a pixel is supported by exactly one class is its negative loss for that class
   masked; it remains a negative for every other class.
3. **Membership learning.** Per-class membership heads are trained with a logistic PU risk (references as positives,
   the unlabeled scene as contrast), jointly with a known-class head.

This release implements the final RCNL model.

The representation is frozen: a spectral view (PCA-30 + window means) and true-colour DINOv3 views
(LVD-1689M and SAT-493M ViT-L/16, native RGB interface, image upsampled ×4). One run (certification + training)
takes about 13 s per scene on a single GPU.

## Installation

```bash
pip install -r requirements.txt
```

DINOv3 weights (Hugging Face format) are needed only for feature extraction:
`dinov3-vitl16-pretrain-lvd1689m` and `dinov3-vitl16-pretrain-sat493m` (see the DINOv3 release for access).

## Data

Each scene is given as `Name=data.mat,gt.mat,first_band_nm,last_band_nm`. The wavelength range is used only to pick
the bands closest to 640 / 550 / 470 nm for the true-colour composite. Ground truth uses 0 for unlabelled pixels.
Scene strings used in the paper (file names as distributed with each dataset):

| Scene | Scene string |
|---|---|
| Pavia University | `PaviaU=PaviaU.mat,PaviaU_gt.mat,430,860` |
| Pavia Centre | `PaviaC=Pavia.mat,Pavia_gt.mat,430,860` |
| Houston 2013 | `Houston2013=Houston2013.mat,Houston2013_gt.mat,380,1050` |
| Houston 2018 | `Houston2018=houstonU2018.mat,houstonU2018.mat,380,1050` |
| WHU-Hi-LongKou | `LongKou=WHU_Hi_LongKou.mat,WHU_Hi_LongKou_gt.mat,400,1000` |
| WHU-Hi-HanChuan | `HanChuan=WHU_Hi_HanChuan.mat,WHU_Hi_HanChuan_gt.mat,400,1000` |
| WHU-Hi-HongHu | `HongHu=WHU_Hi_HongHu.mat,WHU_Hi_HongHu_gt.mat,400,1000` |
| KSC | `KSC=KSC.mat,KSC_gt.mat,400,2500` |
| Botswana | `Botswana=Botswana.mat,Botswana_gt.mat,400,2500` |
| HyRANK Loukia | `Loukia=Loukia.mat,Loukia_gt.mat,400,2500` |
| HyRANK Dioni | `Dioni=Dioni.mat,Dioni_gt.mat,400,2500` |
| Salinas | `Salinas=Salinas_corrected.mat,Salinas_gt.mat,400,2500` |
| Indian Pines | `IndianPines=Indian_pines_corrected.mat,Indian_pines_gt.mat,400,2500` |
| Chikusei | `Chikusei=HyperspecVNIR_Chikusei_20140729.mat,HyperspecVNIR_Chikusei_20140729_Ground_Truth.mat,363,1018` |
| MUUFL | `MUUFL=muufl_gulfport_campus_1_hsi_220_label.mat,muufl_gulfport_campus_1_hsi_220_label.mat,405.8,1005.3` |

The scene name also fixes which classes are withheld from the legend (about 30 %, seed 777), so keep the names above
to reproduce the paper.

## Usage

```bash
# 1. cache the DINOv3 views (once per scene)
python scripts/extract_dinov3.py --out features \
    --fm lvd=/path/to/dinov3-vitl16-pretrain-lvd1689m sat=/path/to/dinov3-vitl16-pretrain-sat493m \
    --scenes "Salinas=/data/Salinas_corrected.mat,/data/Salinas_gt.mat,400,2500"

# 2. open-legend benchmark: 10 references per class, spatially independent (D) and random (R) splits, 3 seeds
python scripts/run_benchmark.py --features features --out results/rcnl.json \
    --scenes "Salinas=/data/Salinas_corrected.mat,/data/Salinas_gt.mat,400,2500"

# 3. wall-to-wall map (class map + legend-membership map)
python scripts/map_scene.py --features features \
    --scene "Salinas=/data/Salinas_corrected.mat,/data/Salinas_gt.mat,400,2500"
```

`map_scene.py` also accepts your own references (`--refs refs.csv` with `row,col,class`) and an optional membership
threshold (`--threshold`) below which pixels are left unclassified.

Python API:

```python
import numpy as np
from rcnl import RCNL, build_views, load_scene, make_split, select_out_of_legend

cube, gt = load_scene("Salinas_corrected.mat", "Salinas_gt.mat")
views = build_views(cube, 400, 2500, {k: np.load(f"features/fm_Salinas_{k}.npy") for k in ("lvd", "sat")})
unk = select_out_of_legend(gt, "Salinas")                 # classes withheld from the legend
gt_legend = np.where(np.isin(gt, unk), 0, gt)
y, tr, te, K = make_split(gt_legend, "D", 10, np.random.default_rng(10000))   # tr: flat indices of the references
model = RCNL().fit(views, tr, y[tr], K, img_w=cube.shape[1], seed=0)
cls, membership = model.predict(np.arange(gt.size))       # legend class and membership s(x) in [0, 1]
```

## Protocol and metrics

- **D (main):** the image is divided into a 4 × 4 checkerboard; references come from one colour, test pixels from
  the other, with a 7-pixel buffer. **R:** references drawn at random from the whole scene.
- About 30 % of the classes (`round(0.3 × #classes)`) are out of the legend: absent from the references, present in
  the unlabeled pool and the test pixels. The unlabeled pool (4,000 pixels) is drawn from all non-reference pixels;
  its labels are never used.
- **OSCR** (main), **AUROC** and **OA** on known test pixels; silent commission and area inflation at the diagnostic
  operating point that accepts 90 % of known test pixels (it uses test labels and is meant only for comparing methods).

## Hyper-parameters (identical for all scenes)

| | |
|---|---|
| encoders | one linear encoder per view, width 128, dropout 0.3, view dropout 0.2 |
| optimisation | AdamW, lr 1e-3, weight decay 1e-2, 1,500 full-batch steps |
| unlabeled pool | 4,000 pixels |
| certification | 5 spatial folds, α = 0.1 |

All are in `rcnl.core.CFG` and can be overridden, e.g. `RCNL(steps=1000, alpha=0.2)`.

## Expected results (15 scenes, 10 references per class, mean of 3 seeds)

| Split | OA | AUROC | OSCR |
|---|---|---|---|
| D (spatially independent) | 83.05 | 84.62 | 74.16 |
| R (random) | 90.49 | 93.05 | 86.38 |

Results depend slightly on GPU and library versions.

## Citation

```
(to be added)
```

## License

(to be added)
