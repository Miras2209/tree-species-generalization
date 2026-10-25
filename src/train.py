"""Fine-tune M1 (EfficientNetV2-S) or M2 (ViT-B/16) with the shared protocol.

Protocol (Section 5.1): 224x224 input, random crop, horizontal flip, small
rotation, colour jitter (no vertical flip), class-weighted cross-entropy, AdamW,
up to 30 epochs, early stopping on validation macro-F1, three seeds.

Usage:
    python -m src.train --model M1 --seed 0
    python -m src.train --model M1 --seed 0 --split image_level    # ablation
"""
import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

CLASS_TO_IDX = {s: i for i, s in enumerate(C.SPECIES_NAMES)}
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

TRAIN_TF = T.Compose([
    T.RandomResizedCrop(C.IMG_SIZE, scale=(0.5, 1.0)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(10),
    T.ColorJitter(0.2, 0.2, 0.2, 0.02),
    T.ToTensor(), T.Normalize(MEAN, STD),
])
EVAL_TF = T.Compose([T.Resize(256), T.CenterCrop(C.IMG_SIZE), T.ToTensor(), T.Normalize(MEAN, STD)])


class TreeImages(Dataset):
    def __init__(self, df, tf):
        self.df, self.tf = df.reset_index(drop=True), tf

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(C.IMG_DIR / f"{r.image_id}.jpg").convert("RGB")
        return self.tf(img), CLASS_TO_IDX[r.species]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(name):
    return timm.create_model(C.MODELS[name], pretrained=True, num_classes=len(C.SPECIES_NAMES))


@torch.no_grad()
def predict(model, df, device):
    model.eval()
    dl = DataLoader(TreeImages(df, EVAL_TF), batch_size=C.BATCH_SIZE, num_workers=2)
    probs = []
    for x, _ in dl:
        probs.append(torch.softmax(model(x.to(device)), 1).cpu().numpy())
    return np.concatenate(probs)


def main(model_name, seed, split, lr):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(C.DATA_DIR / f"split_{split}.csv")
    tr, va = df[df.split == "train"], df[df.split == "val"]

    counts = tr.species.value_counts().reindex(C.SPECIES_NAMES).to_numpy(float)
    weights = torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32, device=device)

    model = build_model(model_name).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=C.MAX_EPOCHS)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    g = torch.Generator().manual_seed(seed)
    dl = DataLoader(TreeImages(tr, TRAIN_TF), batch_size=C.BATCH_SIZE, shuffle=True,
                    num_workers=2, generator=g, drop_last=True)

    run = f"{model_name}_seed{seed}_{split}"
    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    best, wait, t0 = -1.0, 0, time.time()
    y_val = va.species.map(CLASS_TO_IDX).to_numpy()
    for epoch in range(C.MAX_EPOCHS):
        model.train()
        for x, y in dl:
            opt.zero_grad()
            loss = loss_fn(model(x.to(device)), y.to(device))
            loss.backward()
            opt.step()
        sched.step()
        f1 = f1_score(y_val, predict(model, va, device).argmax(1), average="macro")
        print(f"{run} epoch {epoch + 1}: val macro-F1 {f1:.4f}", flush=True)
        if f1 > best:
            best, wait = f1, 0
            torch.save(model.state_dict(), C.OUT_DIR / f"{run}.pt")
        else:
            wait += 1
            if wait >= C.PATIENCE:
                break

    model.load_state_dict(torch.load(C.OUT_DIR / f"{run}.pt", map_location=device))
    ev = df[df.split != "train"].copy()
    p = predict(model, ev, device)
    for k, sp in enumerate(C.SPECIES_NAMES):
        ev[f"p_{sp}"] = p[:, k]
    ev.to_csv(C.OUT_DIR / f"preds_{run}.csv", index=False)
    with open(C.OUT_DIR / "runs.log", "a") as f:
        f.write(f"{run}\tbest_val_macroF1={best:.4f}\tminutes={(time.time() - t0) / 60:.1f}\tlr={lr}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["M1", "M2"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", choices=["grouped", "image_level"], default="grouped")
    ap.add_argument("--lr", type=float, default=3e-4)
    a = ap.parse_args()
    main(a.model, a.seed, a.split, a.lr)
