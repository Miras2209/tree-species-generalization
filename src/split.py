"""Build the data splits.

Main split (Section 5.2): images of one observation stay together, and
observations of the same species by the same observer within 100 m are merged
into one group (probably the same tree). Groups are split 70/15/15 per species.

Ablation split (Section 6.2): a plain image-level random split, used only to
measure how much leakage inflates in-distribution accuracy.

Shifted test sets: Central Asian observations split by month into
leaf-on (June-September) and leaf-off (December-February).

Usage:  python -m src.split
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import BallTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

EARTH_R = 6_371_000.0


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def tree_groups(obs, radius_m=C.SAME_TREE_RADIUS_M):
    """Return a group id per observation (one row per observation)."""
    obs = obs.reset_index(drop=True)
    uf = UnionFind(len(obs))
    for _, idx in obs.groupby(["species", "observer"]).indices.items():
        if len(idx) < 2:
            continue
        coords = np.radians(obs.loc[idx, ["lat", "lon"]].to_numpy(float))
        tree = BallTree(coords, metric="haversine")
        for i, nbrs in zip(idx, tree.query_radius(coords, r=radius_m / EARTH_R)):
            for j in nbrs:
                uf.union(i, idx[j])
    return pd.Series([uf.find(i) for i in range(len(obs))], index=obs.index)


def grouped_split(df, seed=C.SPLIT_SEED, fractions=C.SPLIT_FRACTIONS):
    """Assign train/val/test per tree group, stratified by species."""
    obs = df.drop_duplicates("obs_id")[["obs_id", "species", "observer", "lat", "lon"]]
    obs = obs.reset_index(drop=True)
    obs["group"] = tree_groups(obs)
    rng = np.random.default_rng(seed)
    split_of_group = {}
    for _, g in obs.groupby("species"):
        groups = g["group"].unique()
        rng.shuffle(groups)
        sizes = g.groupby("group").size().loc[groups].to_numpy()
        cum = np.cumsum(sizes) / sizes.sum()
        for grp, c in zip(groups, cum):
            if c <= fractions[0]:
                split_of_group[grp] = "train"
            elif c <= fractions[0] + fractions[1]:
                split_of_group[grp] = "val"
            else:
                split_of_group[grp] = "test_id"
    obs["split"] = obs["group"].map(split_of_group)
    return df.merge(obs[["obs_id", "group", "split"]], on="obs_id", how="left")


def image_level_split(df, seed=C.SPLIT_SEED):
    """Ablation: random split of individual images (leakage allowed)."""
    tr, rest = train_test_split(df.index, test_size=0.30, stratify=df["species"], random_state=seed)
    va, te = train_test_split(rest, test_size=0.50, stratify=df.loc[rest, "species"], random_state=seed)
    out = df.copy()
    out.loc[tr, "split"], out.loc[va, "split"], out.loc[te, "split"] = "train", "val", "test_id"
    return out


def central_asia_sets(df, min_obs=C.MIN_TEST_OBS_PER_SPECIES):
    """Leaf-on and leaf-off test sets; species below min_obs are dropped from that set."""
    df = df.copy()
    df["split"] = np.where(df["month"].isin(C.LEAF_ON_MONTHS), "test_region_leafon",
                  np.where(df["month"].isin(C.LEAF_OFF_MONTHS), "test_region_leafoff", "unused"))
    removed = []
    for s in ("test_region_leafon", "test_region_leafoff"):
        counts = df[df["split"] == s].groupby("species")["obs_id"].nunique()
        for sp, n in counts.items():
            if n < min_obs:
                removed.append((s, sp, n))
                df.loc[(df["split"] == s) & (df["species"] == sp), "split"] = "unused"
    return df[df["split"] != "unused"], pd.DataFrame(removed, columns=["set", "species", "n_obs"])


def main():
    src = pd.read_csv(C.DATA_DIR / "meta_source.csv")
    ca = pd.read_csv(C.DATA_DIR / "meta_central_asia.csv")
    src, ca = src[src["downloaded"]], ca[ca["downloaded"]]

    main_split = grouped_split(src)
    ca_sets, removed = central_asia_sets(ca)
    ca_sets["group"] = ca_sets["obs_id"]
    pd.concat([main_split, ca_sets]).to_csv(C.DATA_DIR / "split_grouped.csv", index=False)
    pd.concat([image_level_split(src), ca_sets]).to_csv(C.DATA_DIR / "split_image_level.csv", index=False)
    removed.to_csv(C.DATA_DIR / "removed_species_from_test_sets.csv", index=False)

    summary = pd.concat([main_split, ca_sets]).groupby("split").agg(
        images=("image_id", "count"), observations=("obs_id", "nunique"), species=("species", "nunique"))
    print(summary)


if __name__ == "__main__":
    main()
