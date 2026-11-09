# Tree species classification beyond the training set

Code and results for the article "Tree Species Classification Using Machine Learning and
Computer Vision: How Well Do Models Transfer to a New Region and Season?"
(Research Methods and Tools, Astana IT University, 2026).

The study fine-tunes three models on citizen-science photographs of 15 tree species from
Europe and North America and tests them on photographs from Central Asia, in summer
(leaf-on, June-September) and winter (leaf-off, December-February).

| Model | Description |
|---|---|
| M1 | EfficientNetV2-S (`tf_efficientnetv2_s.in21k_ft_in1k`), fine-tuned |
| M2 | ViT-B/16 (`vit_base_patch16_224.augreg_in21k`), fine-tuned |
| M3 | Features of the fine-tuned M1 + SVM or k-NN (chosen on validation macro-F1) |

## What is where

| Path | Content |
|---|---|
| `tree_species_pilot.ipynb` | The pilot experiment used in the article, with its outputs. Runs end to end in Google Colab: data download, splits, training, evaluation, figures. |
| `results/` | Everything the notebook produced: `report.txt`, `tables/*.csv`, `figures/*.pdf`, `runs_log.jsonl`, `environment.json`. |
| `results/data/metadata.csv` | All 6,577 photographs used: GBIF occurrence id, photo URL, species, observer, country, coordinates, month, set. |
| `paper/make_paper_figures.py` | Creates the two figures of the article from `results/tables`. |
| `src/`, `config.py`, `tests/` | Full-scale version of the pipeline (2,000 observations per species), planned for the complete study; not used for the pilot numbers. |

## Reproduce the pilot

1. Open `tree_species_pilot.ipynb` in Google Colab, choose a T4 GPU (`Runtime` -> `Change runtime type`).
2. `Runtime` -> `Run all`. Progress is saved to Google Drive (`tree_pilot`), so an interrupted run continues where it stopped.
3. About 78 minutes of training on a T4 GPU, plus the image download.

Seeds: data split 42, training 0, 1, 2, bootstrap 2026. Data were downloaded from GBIF on 28 October 2026;
because iNaturalist keeps growing, a new download may return slightly different observations. Use
`results/data/metadata.csv` to rebuild exactly the same image set.

To redraw the article figures: `python paper/make_paper_figures.py`.

## Data

Research-grade iNaturalist observations via GBIF ("iNaturalist Research-grade Observations",
https://doi.org/10.15468/ab3s5x). The photographs are not included in this repository: their copyright
stays with the iNaturalist observers (CC0 or Creative Commons licences, some non-commercial), and each one
can be downloaded from the URL in `metadata.csv`.

## Licence

Code: MIT (see `LICENSE`). Result tables and figures in `results/` and `paper/`: CC BY 4.0.
