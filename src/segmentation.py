"""
src/segmentation.py

two separate things live here: (1) does the 21-item psychological
block actually measure 5 things like it was designed to, or fewer,
and (2) can we group respondents into meaningful profiles.

spoiler on (1) - PCA says more like 2 factors, not 5. we kept the
5-scale scoring anyway for comparability with the source instruments,
but it's worth being upfront that the dimensionality is questionable.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import config as cfg

PSYCH_ITEMS = ["Q094", "Q095", "Q096", "Q097", "Q099", "Q100", "Q101", "Q102", "Q103",
               "Q104", "Q105", "Q106", "Q107", "Q108", "Q109", "Q110", "Q111",
               "Q112", "Q113", "Q114", "Q115"]

CLUSTER_VARS = ["adm_know_pre", "fin_lit_pre", "anx_pre", "seff_pre", "iu_pre",
                "econ_pre", "trust_pre", "selfrate_know_pre", "funnel_pre"]


def run_pca(pre: pd.DataFrame) -> dict:
    X = pre[PSYCH_ITEMS].dropna()
    Xs = StandardScaler().fit_transform(X)
    pca = PCA().fit(Xs)
    n_kaiser = int((pca.explained_variance_ > 1).sum())
    return {
        "eigenvalues": pca.explained_variance_[:8].tolist(),
        "n_factors_kaiser": n_kaiser,
        "var_explained_5": PCA(n_components=5).fit(Xs).explained_variance_ratio_.sum(),
    }


def silhouette_by_k(D: pd.DataFrame, k_range=range(2, 7)) -> dict:
    Xc = D[CLUSTER_VARS].dropna()
    Xcs = StandardScaler().fit_transform(Xc)
    out = {}
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=cfg.RANDOM_SEED).fit(Xcs)
        out[k] = silhouette_score(Xcs, km.labels_)
    return out


def fit_clusters(D: pd.DataFrame, k: int = 4) -> pd.DataFrame:
    Xc = D[CLUSTER_VARS].dropna()
    Xcs = StandardScaler().fit_transform(Xc)
    km = KMeans(n_clusters=k, n_init=10, random_state=cfg.RANDOM_SEED).fit(Xcs)

    D = D.copy()
    D.loc[Xc.index, "cluster"] = km.labels_

    profile = D.groupby("cluster")[CLUSTER_VARS].mean()
    profile_z = (profile - D[CLUSTER_VARS].mean()) / D[CLUSTER_VARS].std()
    return profile_z


if __name__ == "__main__":
    pre = pd.read_excel(cfg.PRE_PATH).apply(pd.to_numeric, errors="coerce")
    D = pd.read_pickle(cfg.ANALYTIC_PATH)

    pca_result = run_pca(pre)
    print(f"eigenvalues: {[round(x,2) for x in pca_result['eigenvalues']]}")
    print(f"kaiser criterion says {pca_result['n_factors_kaiser']} factors "
          f"(instrument was designed with 5 sub-scales...)")

    print("\nsilhouette by k:")
    for k, s in silhouette_by_k(D).items():
        print(f"  k={k}: {s:.3f}")

    print("\ncluster profiles (k=4, standardized):")
    print(fit_clusters(D).round(2))
