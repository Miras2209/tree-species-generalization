"""Checks of the splitting and evaluation logic on synthetic data (no GPU needed).

Run:  python -m pytest -q tests
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
from src.evaluate import bootstrap_ci, mcnemar, metrics, to_observations  # noqa: E402
from src.split import central_asia_sets, grouped_split, tree_groups  # noqa: E402


def fake_source(n_obs=600, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_obs):
        sp = C.SPECIES_NAMES[i % len(C.SPECIES_NAMES)]
        lat, lon = 50 + rng.normal(0, 2), 10 + rng.normal(0, 2)
        for k in range(rng.integers(1, 4)):
            rows.append({"obs_id": i, "image_id": f"{i}_{k}", "species": sp, "observer": f"u{i % 40}",
                         "lat": lat, "lon": lon, "month": int(rng.integers(1, 13))})
    return pd.DataFrame(rows)


def test_same_tree_grouping():
    obs = pd.DataFrame({"species": ["A", "A", "A", "B"], "observer": ["x", "x", "x", "x"],
                        "lat": [51.0, 51.0005, 52.0, 51.0], "lon": [71.0, 71.0, 71.0, 71.0]})
    g = tree_groups(obs)
    assert g[0] == g[1]          # about 55 m apart: same tree
    assert g[0] != g[2]          # 111 km apart
    assert g[0] != g[3]          # different species


def test_grouped_split_has_no_leakage():
    df = grouped_split(fake_source())
    assert set(df.split) == {"train", "val", "test_id"}
    assert df.groupby("group").split.nunique().max() == 1
    assert df.groupby("obs_id").split.nunique().max() == 1
    share = df.split.value_counts(normalize=True)
    assert 0.6 < share["train"] < 0.8


def test_central_asia_season_sets():
    ca = fake_source(400, seed=1)
    sets, removed = central_asia_sets(ca, min_obs=5)
    assert set(sets.split) <= {"test_region_leafon", "test_region_leafoff"}
    assert sets[sets.split == "test_region_leafon"].month.isin(C.LEAF_ON_MONTHS).all()


def fake_preds(n=300, acc=0.8, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, len(C.SPECIES_NAMES), n)
    logits = rng.normal(0, 1, (n, len(C.SPECIES_NAMES)))
    hit = rng.random(n) < acc
    logits[np.arange(n)[hit], y[hit]] += 10
    p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    df = pd.DataFrame(p, columns=[f"p_{s}" for s in C.SPECIES_NAMES])
    df["species"] = np.array(C.SPECIES_NAMES)[y]
    df["obs_id"] = np.arange(n) // 2
    df["split"] = "test_id"
    return df


def test_metrics_ranges_and_error_shares():
    m = metrics(fake_preds())
    assert 70 < m["top1"] < 90
    assert m["top5"] >= m["top1"]
    assert m["genus_acc"] >= m["top1"] and m["family_acc"] >= m["genus_acc"]
    assert abs(m["err_intra_genus"] + m["err_intra_family"] + m["err_inter_family"] - 100) < 1e-6


def test_mcnemar_and_bootstrap():
    a = np.array([True] * 80 + [False] * 20)
    b = np.array([True] * 60 + [False] * 40)
    bb, cc, _, p = mcnemar(a, b)
    assert (bb, cc) == (20, 0) and p < 0.001
    assert mcnemar(a, a)[3] == 1.0
    obs = to_observations(fake_preds())
    lo, hi = bootstrap_ci(obs, n=200)
    assert lo < metrics(obs)["top1"] < hi
