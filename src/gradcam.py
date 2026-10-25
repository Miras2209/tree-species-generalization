"""Grad-CAM attention share on whole-tree images (Section 5.3, Explainability).

For each of the ~100 whole-tree images with a hand-drawn tree mask
(data/masks/<image_id>.png, white = tree), we compute
    heat_share = heatmap mass inside the mask / total heatmap mass
    area_share = mask area / image area
    ratio      = heat_share / area_share
A model with ratio < 1 (median over images) is classed as background-dependent.

Usage:  python -m src.gradcam --model M1 --seed 0
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
from src.train import CLASS_TO_IDX, EVAL_TF, build_model  # noqa: E402

MASK_DIR = C.DATA_DIR / "masks"


def vit_reshape(t, h=14, w=14):
    t = t[:, 1:, :].reshape(t.size(0), h, w, t.size(2))
    return t.permute(0, 3, 1, 2)


def attention_share(cam_map, mask):
    heat = cam_map / (cam_map.sum() + 1e-12)
    heat_share = float(heat[mask].sum())
    area_share = float(mask.mean())
    return heat_share, area_share, heat_share / max(area_share, 1e-12)


def mask_224(path):
    m = Image.open(path).convert("L")
    m = m.resize((round(m.width * 256 / min(m.size)), round(m.height * 256 / min(m.size))), Image.NEAREST)
    left, top = (m.width - C.IMG_SIZE) // 2, (m.height - C.IMG_SIZE) // 2
    return np.array(m.crop((left, top, left + C.IMG_SIZE, top + C.IMG_SIZE))) > 127


def main(model_name, seed):
    model = build_model(model_name)
    model.load_state_dict(torch.load(C.OUT_DIR / f"{model_name}_seed{seed}_grouped.pt", map_location="cpu"))
    model.eval()
    if model_name == "M2":
        cam = GradCAM(model, target_layers=[model.blocks[-1].norm1], reshape_transform=vit_reshape)
    else:
        cam = GradCAM(model, target_layers=[model.conv_head])
    meta = pd.read_csv(C.DATA_DIR / "split_grouped.csv").set_index("image_id")
    rows = []
    for mpath in sorted(MASK_DIR.glob("*.png")):
        iid = mpath.stem
        x = EVAL_TF(Image.open(C.IMG_DIR / f"{iid}.jpg").convert("RGB")).unsqueeze(0)
        target = CLASS_TO_IDX[meta.loc[iid, "species"]]
        cmap = cam(input_tensor=x, targets=[ClassifierOutputTarget(target)])[0]
        hs, ar, ratio = attention_share(cmap, mask_224(mpath))
        rows.append({"image_id": iid, "species": meta.loc[iid, "species"], "split": meta.loc[iid, "split"],
                     "heat_share": hs, "area_share": ar, "ratio": ratio})
    out = pd.DataFrame(rows)
    out.to_csv(C.OUT_DIR / f"attention_{model_name}_seed{seed}.csv", index=False)
    print(out.groupby("split")[["heat_share", "area_share", "ratio"]].median())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["M1", "M2"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.model, a.seed)
