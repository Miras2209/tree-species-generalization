"""Figures of the article, made from the pilot result tables in results/tables.

Usage:  python paper/make_paper_figures.py
Output: paper/fig_accuracy_season.pdf, paper/fig_confusion_leafon.pdf
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"
OUT = Path(__file__).resolve().parent

plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#888", "axes.linewidth": 0.6,
                     "xtick.color": "#333", "ytick.color": "#333", "pdf.fonttype": 42})
MODELS = ["M1", "M2", "M3"]
COL = {"M1": "#2a78d6", "M2": "#eb6834", "M3": "#1baf7a"}
HATCH = {"M1": "", "M2": "////", "M3": "...."}
LAB = {"M1": "M1 EfficientNetV2-S", "M2": "M2 ViT-B/16", "M3": "M3 M1 features + SVM"}

# ---------------- Figure 1 ----------------
seed = pd.read_csv(TAB / "per_seed_metrics.csv")
sets = ["test_id", "test_leafon", "test_leafoff"]
xl = ["In-distribution", "Central Asia\nleaf-on", "Central Asia\nleaf-off"]
fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw={"width_ratios": [1.35, 1]})
w = 0.25
for i, m in enumerate(MODELS):
    for j, s in enumerate(sets):
        v = seed[(seed.model == m) & (seed.set == s)].top1
        x = j + (i - 1) * (w + 0.02)
        a.bar(x, v.mean(), w, yerr=v.std(), color=COL[m], hatch=HATCH[m], edgecolor="white", linewidth=0.8,
              error_kw={"elinewidth": 0.9, "capsize": 2.5, "ecolor": "#333"})
        a.text(x, v.mean() + v.std() + 1.5, f"{v.mean():.1f}", ha="center", va="bottom", fontsize=6.3, color="#333")
a.set_xticks(range(3), xl, fontsize=7.2)
a.set_ylabel("Top-1 accuracy (%)")
a.set_ylim(0, 100)
a.yaxis.grid(True, color="#e6e6e6", linewidth=0.6)
a.set_axisbelow(True)
a.set_title("(a) Top-1 accuracy, mean and SD over 3 seeds", fontsize=8.5, loc="left")

# panel b: recall of M2 (median-seed run) for the four species present in both Central Asian sets
shared = ["Betula pendula", "Pinus sylvestris", "Acer negundo", "Ulmus pumila"]
rec = {}
for s in ["test_leafon", "test_leafoff"]:
    cm = pd.read_csv(TAB / f"confusion_M2_{s}.csv", index_col=0)
    rec[s] = [100 * cm.loc[sp, sp] / cm.loc[sp].sum() for sp in shared]
y = np.arange(len(shared))
for k in range(len(shared)):
    b.plot([rec["test_leafon"][k], rec["test_leafoff"][k]], [y[k], y[k]], color="#bbb", linewidth=1.2, zorder=1)
b.scatter(rec["test_leafon"], y, s=34, color="#eb6834", edgecolor="white", linewidth=0.6, zorder=2, label="Leaf-on (Jun-Sep)")
b.scatter(rec["test_leafoff"], y, s=34, color="#5a2a12", marker="s", edgecolor="white", linewidth=0.6, zorder=2, label="Leaf-off (Dec-Feb)")
b.set_yticks(y, shared, style="italic", fontsize=7.5)
b.invert_yaxis()
b.set_xlim(0, 100)
b.set_xlabel("Recall of M2 (%)")
b.xaxis.grid(True, color="#e6e6e6", linewidth=0.6)
b.set_axisbelow(True)
b.set_title("(b) Same species, two seasons (M2)", fontsize=8.5, loc="left")
b.legend(loc="upper center", bbox_to_anchor=(0.4, -0.27), ncol=2, frameon=False, fontsize=7)
fig.legend(handles=[Patch(facecolor=COL[m], hatch=HATCH[m], edgecolor="white", label=LAB[m]) for m in MODELS],
           loc="lower left", bbox_to_anchor=(0.03, 0.0), ncol=3, frameon=False, fontsize=7)
fig.tight_layout(rect=(0, 0.1, 1, 1))
fig.savefig(OUT / "fig_accuracy_season.pdf", bbox_inches="tight")
plt.close(fig)

# ---------------- Figure 2 ----------------
cm = pd.read_csv(TAB / "confusion_M2_test_leafon.csv", index_col=0)
names = list(cm.columns)
fam = {"Betula pendula": "Betulaceae", "Fraxinus pennsylvanica": "Oleaceae", "Larix sibirica": "Pinaceae",
       "Picea pungens": "Pinaceae", "Pinus sylvestris": "Pinaceae", "Prunus padus": "Rosaceae",
       "Prunus virginiana": "Rosaceae", "Sorbus aucuparia": "Rosaceae", "Populus nigra": "Salicaceae",
       "Populus tremula": "Salicaceae", "Salix alba": "Salicaceae", "Acer negundo": "Sapindaceae",
       "Acer platanoides": "Sapindaceae", "Ulmus laevis": "Ulmaceae", "Ulmus pumila": "Ulmaceae"}
rows = [n for n in names if cm.loc[n].sum() > 0]
m = cm.loc[rows].to_numpy(float)
m = 100 * m / m.sum(1, keepdims=True)
fig, ax = plt.subplots(figsize=(6.2, 4.6))
im = ax.imshow(m, cmap="Blues", vmin=0, vmax=100, aspect="auto")
for i in range(m.shape[0]):
    for j in range(m.shape[1]):
        if m[i, j] >= 5:
            ax.text(j, i, f"{m[i, j]:.0f}", ha="center", va="center", fontsize=6.3,
                    color="white" if m[i, j] > 55 else "#222")
ab = lambda n: n.split()[0][0] + ". " + n.split()[1]
ax.set_xticks(range(len(names)), [ab(n) for n in names], rotation=60, ha="right", fontsize=7, style="italic")
ax.set_yticks(range(len(rows)), [ab(n) for n in rows], fontsize=7, style="italic")
for j in range(1, len(names)):
    if fam[names[j]] != fam[names[j - 1]]:
        ax.axvline(j - 0.5, color="#555", linewidth=0.7)
for i in range(1, len(rows)):
    if fam[rows[i]] != fam[rows[i - 1]]:
        ax.axhline(i - 0.5, color="#555", linewidth=0.7)
ax.set_xlabel("Predicted species")
ax.set_ylabel("True species")
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cb.set_label("Share of the true species' images (%)", fontsize=7.5)
cb.ax.tick_params(labelsize=7)
fig.tight_layout()
fig.savefig(OUT / "fig_confusion_leafon.pdf", bbox_inches="tight")
plt.close(fig)
print("saved figures to", OUT)
