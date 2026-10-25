"""Evaluation for all research questions (Section 5.3).

Image level: Top-1, Top-5, macro-F1, genus and family accuracy, error types.
Observation level: probabilities of all images of one observation are averaged;
McNemar tests and bootstrap confidence intervals use this level because images
of one tree are not independent.

Usage:  python -m src.evaluate            (reads outputs/preds_*.csv)
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2
from sklearn.metrics import confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

SPECIES = np.array(C.SPECIES_NAMES)
GEN = np.array([C.GENUS[s] for s in SPECIES])
FAM = np.array([C.FAMILY[s] for s in SPECIES])
TEST_SETS = ["test_id", "test_region_leafon", "test_region_leafoff"]


def prob_matrix(df):
    return df[[f"p_{s}" for s in SPECIES]].to_numpy()


def to_observations(df):
    """Average image probabilities within an observation."""
    pcols = [f"p_{s}" for s in SPECIES]
    return df.groupby(["split", "obs_id", "species"], as_index=False)[pcols].mean()


def metrics(df):
    p = prob_matrix(df)
    y = np.array([C.SPECIES_NAMES.index(s) for s in df["species"]])
    pred = p.argmax(1)
    top5 = np.argsort(-p, 1)[:, :5]
    wrong = pred != y
    same_gen = GEN[pred] == GEN[y]
    same_fam = FAM[pred] == FAM[y]
    n_err = max(wrong.sum(), 1)
    return {
        "n": len(y),
        "top1": 100 * (~wrong).mean(),
        "top5": 100 * (top5 == y[:, None]).any(1).mean(),
        "macro_f1": f1_score(y, pred, average="macro", labels=np.unique(y)),
        "genus_acc": 100 * same_gen.mean(),
        "family_acc": 100 * same_fam.mean(),
        "err_intra_genus": 100 * (wrong & same_gen).sum() / n_err,
        "err_intra_family": 100 * (wrong & ~same_gen & same_fam).sum() / n_err,
        "err_inter_family": 100 * (wrong & ~same_fam).sum() / n_err,
    }


def mcnemar(correct_a, correct_b):
    """Paired test on the same observations. Returns b, c, statistic, p-value."""
    b = int((correct_a & ~correct_b).sum())
    c = int((~correct_a & correct_b).sum())
    if b + c == 0:
        return b, c, 0.0, 1.0
    if b + c < 25:  # exact binomial version for few discordant pairs
        return b, c, np.nan, binomtest(b, b + c, 0.5).pvalue
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    return b, c, stat, chi2.sf(stat, 1)


def bootstrap_ci(obs_df, key="top1", n=C.BOOTSTRAP_RESAMPLES, seed=C.BOOTSTRAP_SEED):
    rng = np.random.default_rng(seed)
    vals = [metrics(obs_df.iloc[rng.integers(0, len(obs_df), len(obs_df))])[key] for _ in range(n)]
    return np.percentile(vals, [2.5, 97.5])


def load_runs(split="grouped"):
    runs = {}
    for f in sorted(C.OUT_DIR.glob(f"preds_M*_seed*_{split}.csv")):
        m, s = re.match(r"preds_(M\d)_seed(\d+)_", f.name).groups()
        runs[(m, int(s))] = pd.read_csv(f)
    return runs


def median_seed(runs, model):
    """Seed whose validation macro-F1 is the median of the three runs."""
    scores = {s: metrics(df[df.split == "val"])["macro_f1"] for (m, s), df in runs.items() if m == model}
    order = sorted(scores, key=scores.get)
    return order[len(order) // 2]


def main():
    runs = load_runs("grouped")
    models = sorted({m for m, _ in runs})

    # Table 3: main results, mean and SD over seeds (image level)
    rows = []
    for (m, s), df in runs.items():
        for t in TEST_SETS:
            rows.append({"model": m, "seed": s, "test_set": t, **metrics(df[df.split == t])})
    per_seed = pd.DataFrame(rows)
    main_tab = per_seed.groupby(["model", "test_set"]).agg(["mean", "std"]).drop(columns="seed")
    main_tab.round(2).to_csv(C.OUT_DIR / "table_main_results.csv")

    # Generalization gaps
    acc = per_seed.pivot_table(index=["model", "seed"], columns="test_set", values="top1")
    acc["gap_region"] = acc["test_id"] - acc["test_region_leafon"]
    acc["gap_season"] = acc["test_region_leafon"] - acc["test_region_leafoff"]
    acc.groupby("model")[["gap_region", "gap_season"]].agg(["mean", "std"]).round(2) \
        .to_csv(C.OUT_DIR / "table_gaps.csv")

    # Observation level: bootstrap CIs and McNemar (median-seed run per model)
    med = {m: median_seed(runs, m) for m in models}
    obs = {m: to_observations(runs[(m, med[m])]) for m in models}
    ci_rows, mc_rows = [], []
    for t in TEST_SETS:
        for m in models:
            o = obs[m][obs[m].split == t]
            lo, hi = bootstrap_ci(o)
            ci_rows.append({"model": m, "test_set": t, "n_obs": len(o),
                            "top1_obs": metrics(o)["top1"], "ci_low": lo, "ci_high": hi})
        for i, a in enumerate(models):
            for b in models[i + 1:]:
                ids = np.intersect1d(obs[a][obs[a].split == t].obs_id, obs[b][obs[b].split == t].obs_id)
                oa = obs[a][obs[a].obs_id.isin(ids)].sort_values("obs_id")
                ob = obs[b][obs[b].obs_id.isin(ids)].sort_values("obs_id")
                ya = np.array([C.SPECIES_NAMES.index(s) for s in oa.species])
                ca = prob_matrix(oa).argmax(1) == ya
                cb = prob_matrix(ob).argmax(1) == ya
                bb, cc, stat, p = mcnemar(ca, cb)
                mc_rows.append({"pair": f"{a} vs {b}", "test_set": t, "n_obs": len(oa),
                                "b": bb, "c": cc, "chi2": stat, "p": p, "significant": p < C.ALPHA})
    pd.DataFrame(ci_rows).round(2).to_csv(C.OUT_DIR / "table_bootstrap_ci.csv", index=False)
    pd.DataFrame(mc_rows).round(4).to_csv(C.OUT_DIR / "table_mcnemar.csv", index=False)

    # Per group (deciduous vs evergreen) and per species, median-seed run
    grp_rows = []
    for m in models:
        df = runs[(m, med[m])]
        for t in TEST_SETS:
            d = df[df.split == t]
            habit = d.species.map(C.HABIT).str.contains("evergreen").map({True: "evergreen", False: "deciduous"})
            for h in ("deciduous", "evergreen"):
                if (habit == h).any():
                    grp_rows.append({"model": m, "test_set": t, "group": h, **metrics(d[habit == h])})
            for sp in d.species.unique():
                grp_rows.append({"model": m, "test_set": t, "group": sp, **metrics(d[d.species == sp])})
            y = [C.SPECIES_NAMES.index(s) for s in d.species]
            cm = confusion_matrix(y, prob_matrix(d).argmax(1), labels=range(len(SPECIES)))
            order = np.lexsort((SPECIES, GEN, FAM))
            pd.DataFrame(cm[np.ix_(order, order)], index=SPECIES[order], columns=SPECIES[order]) \
                .to_csv(C.OUT_DIR / f"confusion_{m}_{t}.csv")
    pd.DataFrame(grp_rows).round(2).to_csv(C.OUT_DIR / "table_per_group.csv", index=False)

    # Ablation: image-level split versus grouped split (M1)
    abl = load_runs("image_level")
    if abl:
        rows = [{"split": "image_level", "seed": s, "test_set": t, "top1": metrics(df[df.split == t])["top1"]}
                for (m, s), df in abl.items() if m == "M1" for t in TEST_SETS]
        rows += [{"split": "grouped", "seed": s, "test_set": t, "top1": metrics(df[df.split == t])["top1"]}
                 for (m, s), df in runs.items() if m == "M1" for t in TEST_SETS]
        pd.DataFrame(rows).groupby(["split", "test_set"])["top1"].agg(["mean", "std"]).round(2) \
            .to_csv(C.OUT_DIR / "table_split_ablation.csv")
    print(main_tab.round(2))


if __name__ == "__main__":
    main()
