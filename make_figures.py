"""Figures for Section 6 (expected results). All plotted values are EXPECTED ranges
derived from the literature, not measurements."""
import sys
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle

out = sys.argv[1]
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#888", "axes.linewidth": 0.6,
                     "xtick.color": "#444", "ytick.color": "#444", "pdf.fonttype": 42})
COL = {"M1": "#2a78d6", "M2": "#eb6834", "M3": "#1baf7a"}
HATCH = {"M1": "", "M2": "////", "M3": "...."}
LAB = {"M1": "M1 EfficientNetV2-S", "M2": "M2 ViT-B/16", "M3": "M3 M1 features + SVM/k-NN"}
TXT = "#222"

# ---------------- Figure 1: expected accuracy and gaps ----------------
top1 = {"M1": [(86, 91), (72, 82), (55, 70)],
        "M2": [(88, 93), (74, 85), (57, 72)],
        "M3": [(86, 92), (70, 82), (53, 70)]}
gaps = {"M1": [(7, 15), (10, 20)], "M2": [(6, 14), (10, 20)], "M3": [(8, 16), (10, 20)]}
fig, (a, b) = plt.subplots(1, 2, figsize=(6.3, 2.9), gridspec_kw={"width_ratios": [1.45, 1]})
conds = ["In-distribution\ntest", "New region\n(leaf-on)", "New region\n+ season (leaf-off)"]
w = 0.24
for i, m in enumerate(COL):
    for j, (lo, hi) in enumerate(top1[m]):
        x = j + (i - 1) * (w + 0.03)
        a.bar(x, hi - lo, w, bottom=lo, color=COL[m], hatch=HATCH[m], edgecolor="white", linewidth=0.8)
a.set_xticks(range(3), conds, fontsize=7.5)
a.set_ylabel("Expected Top-1 accuracy (%)")
a.set_ylim(50, 100)
a.yaxis.grid(True, color="#e6e6e6", linewidth=0.6)
a.set_axisbelow(True)
a.set_title("(a) Expected Top-1 accuracy range", fontsize=9, loc="left", color=TXT)
for i, m in enumerate(COL):
    for j, (lo, hi) in enumerate(gaps[m]):
        x = j + (i - 1) * (w + 0.03)
        b.bar(x, hi - lo, w, bottom=lo, color=COL[m], hatch=HATCH[m], edgecolor="white", linewidth=0.8)
b.axhline(10, color="#444", linestyle="--", linewidth=0.9)
b.set_xticks(range(2), ["Region gap", "Season gap"], fontsize=7.5)
b.set_ylabel("Expected accuracy drop (pp)")
b.set_ylim(0, 25)
b.yaxis.grid(True, color="#e6e6e6", linewidth=0.6)
b.set_axisbelow(True)
b.set_title("(b) Expected gap", fontsize=9, loc="left", color=TXT)
fig.legend(handles=[Patch(facecolor=COL[m], hatch=HATCH[m], edgecolor="white", label=LAB[m]) for m in COL]
           + [plt.Line2D([], [], color="#444", linestyle="--", linewidth=0.9, label="10 pp success limit")],
           loc="lower center", ncol=2, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.13, 1, 1))
fig.savefig(f"{out}/fig_expected_accuracy.pdf", bbox_inches="tight")

# ---------------- Figure 2: genus-grouped confusion matrix layout ----------------
sp = [("Betula pendula", "Betula", "Betulaceae"), ("Fraxinus pennsylvanica", "Fraxinus", "Oleaceae"),
      ("Larix sibirica", "Larix", "Pinaceae"), ("Picea pungens", "Picea", "Pinaceae"),
      ("Pinus sylvestris", "Pinus", "Pinaceae"), ("Prunus padus", "Prunus", "Rosaceae"),
      ("Prunus virginiana", "Prunus", "Rosaceae"), ("Sorbus aucuparia", "Sorbus", "Rosaceae"),
      ("Populus nigra", "Populus", "Salicaceae"), ("Populus tremula", "Populus", "Salicaceae"),
      ("Salix alba", "Salix", "Salicaceae"), ("Acer negundo", "Acer", "Sapindaceae"),
      ("Acer platanoides", "Acer", "Sapindaceae"), ("Ulmus laevis", "Ulmus", "Ulmaceae"),
      ("Ulmus pumila", "Ulmus", "Ulmaceae")]
n = len(sp)
shade = {"correct": "#184f95", "genus": "#5598e7", "family": "#b7d3f6", "other": "#fcfcfb"}
fig, ax = plt.subplots(figsize=(5.6, 5.0))
for i, (si, gi, fi) in enumerate(sp):
    for j, (sj, gj, fj) in enumerate(sp):
        k = "correct" if i == j else "genus" if gi == gj else "family" if fi == fj else "other"
        ax.add_patch(Rectangle((j, i), 1, 1, facecolor=shade[k], edgecolor="#d9d9d9", linewidth=0.4))
expected_pairs = [("Ulmus laevis", "Ulmus pumila"), ("Prunus padus", "Prunus virginiana"),
                  ("Populus nigra", "Populus tremula"), ("Acer negundo", "Fraxinus pennsylvanica"),
                  ("Larix sibirica", "Populus tremula"), ("Larix sibirica", "Betula pendula"),
                  ("Picea pungens", "Pinus sylvestris")]
idx = {s[0]: i for i, s in enumerate(sp)}
one_way = {("Larix sibirica", "Populus tremula"), ("Larix sibirica", "Betula pendula")}
for p, q in expected_pairs:
    cells = [(idx[p], idx[q])] if (p, q) in one_way else [(idx[p], idx[q]), (idx[q], idx[p])]
    for r, c in cells:
        ax.plot(c + 0.5, r + 0.5, marker="o", markersize=7, markerfacecolor="none",
                markeredgecolor="#111", markeredgewidth=1.1)
fam_edges = [i for i in range(1, n) if sp[i][2] != sp[i - 1][2]]
for e in fam_edges:
    ax.axhline(e, color="#555", linewidth=0.9)
    ax.axvline(e, color="#555", linewidth=0.9)
names = [s[0].replace(" ", " ") for s in sp]
ax.set_xlim(0, n)
ax.set_ylim(n, 0)
ax.set_xticks(np.arange(n) + 0.5, names, rotation=60, ha="right", fontsize=7, style="italic")
ax.set_yticks(np.arange(n) + 0.5, names, fontsize=7, style="italic")
ax.set_xlabel("Predicted species")
ax.set_ylabel("True species")
for s in ax.spines.values():
    s.set_visible(False)
ax.tick_params(length=0)
leg = [Patch(facecolor=shade["correct"], label="Correct (diagonal)"),
       Patch(facecolor=shade["genus"], label="Intra-genus error"),
       Patch(facecolor=shade["family"], label="Intra-family error"),
       Patch(facecolor=shade["other"], edgecolor="#bbb", label="Inter-family error"),
       plt.Line2D([], [], marker="o", linestyle="none", markerfacecolor="none", markeredgecolor="#111",
                  markersize=7, label="Expected frequent confusion")]
ax.legend(handles=leg, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=7.5)
fig.tight_layout()
fig.savefig(f"{out}/fig_confusion_layout.pdf", bbox_inches="tight")
print("ok")
