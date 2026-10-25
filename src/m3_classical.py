"""M3: features from the fine-tuned M1 backbone + SVM or k-NN (Section 5.1).

The classifier (SVM or k-NN) and its hyperparameters are chosen by validation
macro-F1. The output has the same format as the M1/M2 prediction files.

Usage:  python -m src.m3_classical --seed 0
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
from src.train import CLASS_TO_IDX, EVAL_TF, TreeImages, build_model  # noqa: E402


@torch.no_grad()
def features(model, df, device):
    model.eval()
    dl = DataLoader(TreeImages(df, EVAL_TF), batch_size=C.BATCH_SIZE, num_workers=2)
    return np.concatenate([model.forward_head(model.forward_features(x.to(device)), pre_logits=True)
                           .cpu().numpy() for x, _ in dl])


def main(seed, split):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run = f"M1_seed{seed}_{split}"
    model = build_model("M1").to(device)
    model.load_state_dict(torch.load(C.OUT_DIR / f"{run}.pt", map_location=device))
    df = pd.read_csv(C.DATA_DIR / f"split_{split}.csv")
    feats = {s: features(model, df[df.split == s], device) for s in df.split.unique()}
    y = {s: df[df.split == s].species.map(CLASS_TO_IDX).to_numpy() for s in df.split.unique()}

    candidates = [("svm", c, make_pipeline(StandardScaler(), SVC(C=c, probability=True, random_state=seed)))
                  for c in (0.1, 1, 10)]
    candidates += [("knn", k, make_pipeline(StandardScaler(), KNeighborsClassifier(k, weights="distance")))
                   for k in (5, 15, 31)]
    best = None
    for name, hp, clf in candidates:
        clf.fit(feats["train"], y["train"])
        f1 = f1_score(y["val"], clf.predict(feats["val"]), average="macro")
        print(f"M3 {name} {hp}: val macro-F1 {f1:.4f}")
        if best is None or f1 > best[0]:
            best = (f1, name, hp, clf)

    f1, name, hp, clf = best
    ev = df[df.split != "train"].copy()
    p = np.zeros((len(ev), len(C.SPECIES_NAMES)))
    for s in ev.split.unique():
        rows = np.where((ev.split == s).to_numpy())[0]
        p[rows[:, None], clf.classes_[None, :]] = clf.predict_proba(feats[s])
    for k, sp in enumerate(C.SPECIES_NAMES):
        ev[f"p_{sp}"] = p[:, k]
    ev.to_csv(C.OUT_DIR / f"preds_M3_seed{seed}_{split}.csv", index=False)
    with open(C.OUT_DIR / "runs.log", "a") as f:
        f.write(f"M3_seed{seed}_{split}\tclassifier={name}({hp})\tbest_val_macroF1={f1:.4f}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", choices=["grouped", "image_level"], default="grouped")
    a = ap.parse_args()
    main(a.seed, a.split)
