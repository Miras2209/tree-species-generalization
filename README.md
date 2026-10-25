# Tree species classification beyond the training set

Code for the study "Tree Species Classification Using Machine Learning and Computer Vision"
(Research Methods and Tools, Astana IT University). It compares three models on
public iNaturalist photographs and tests how well they generalize to a new region
(Central Asia) and a new season (leaf-off months).

| Model | Description |
|---|---|
| M1 | EfficientNetV2-S, ImageNet-21k pre-trained, fine-tuned |
| M2 | ViT-B/16, ImageNet-21k pre-trained, fine-tuned |
| M3 | Features of the fine-tuned M1 + SVM or k-NN (chosen on validation macro-F1) |

**Status:** the pipeline is planned and tested on synthetic data (`tests/`). The full
experiment has not been run yet; the article reports expected results.

## Data

Research-grade iNaturalist observations downloaded through the GBIF API
(dataset key `50c9509d-22c7-4a22-a47d-8c48425ef4a7`), openly licensed images only.

* Source (train / validation / in-distribution test): Europe and North America, up to
  2,000 observations and 3 images per observation per species.
* Shifted test sets: Kazakhstan, Kyrgyzstan, Uzbekistan, Tajikistan and Turkmenistan,
  split by month into leaf-on (June-September) and leaf-off (December-February).

## Run (Kaggle notebook with a GPU, about 35 GPU-hours in total)

```bash
pip install -r requirements.txt
python -m src.download_gbif --part source
python -m src.download_gbif --part central_asia
python -m src.download_gbif --images
python -m src.split                                   # grouped and image-level splits

for s in 0 1 2; do
  python -m src.train --model M1 --seed $s
  python -m src.train --model M2 --seed $s
  python -m src.m3_classical --seed $s
  python -m src.train --model M1 --seed $s --split image_level   # ablation
done

python -m src.gradcam --model M1 --seed <median seed>   # needs data/masks/*.png
python -m src.gradcam --model M2 --seed <median seed>
python -m src.evaluate                                # writes outputs/table_*.csv
python -m pytest -q tests                             # checks of split and metric code
```

## Reproducibility

* Split seed 42, training seeds 0, 1, 2, bootstrap seed 2026 (1,000 resamples), all in `config.py`.
* Learning rate is tuned on the validation set only (`--lr`); test sets are used once.
* Hardware: Kaggle NVIDIA Tesla P100 (16 GB); Google Colab as a backup.
