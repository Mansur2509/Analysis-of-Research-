"""
src/data_prep.py

loads the three raw excel files and builds the analytic dataset used
by everything else in this repo. run this first, everything downstream
imports ANALYTIC.pkl.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import config as cfg


def load_raw():
    """just reads the three excel files, nothing fancy."""
    pre = pd.read_excel(cfg.PRE_PATH)
    post = pd.read_excel(cfg.POST_PATH)
    rct = pd.read_excel(cfg.RCT_PATH)
    return pre, post, rct


def numify(df: pd.DataFrame) -> pd.DataFrame:
    """
    coerce every column to numeric. multi-select columns (stored as
    "1,3" strings) will turn into NaN here on purpose - we handle those
    separately where needed, this function is only for the plain items.
    """
    return df.apply(lambda s: pd.to_numeric(s, errors="coerce"))


def score_knowledge(df_n: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """admissions knowledge, financial literacy, and the combined total."""
    out = pd.DataFrame(index=df_n.index)
    out["adm_know" + suffix] = sum((df_n[q] == v).astype(int) for q, v in cfg.ADM_KEY.items())
    out["fin_lit" + suffix] = sum((df_n[q] == v).astype(int) for q, v in cfg.FIN_KEY.items())
    out["compound" + suffix] = (df_n["Q078"] == 2).astype(int)
    out["total_know" + suffix] = out["adm_know" + suffix] + out["fin_lit" + suffix] + out["compound" + suffix]
    return out


def build_wave(df_n: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """
    builds one wave (pre or post) of every derived variable we use in
    analysis. suffix is "_pre" or "_post" so columns from both waves
    don't collide when we concat them later.
    """
    d = score_knowledge(df_n, suffix)

    for name, cols in cfg.SCALES.items():
        d[name + suffix] = df_n[cols].mean(axis=1)

    d["selfrate_know" + suffix] = df_n["Q088"]
    d["selfrate_aid" + suffix] = df_n["Q089"]
    d["selfrate_risk" + suffix] = df_n["Q090"]

    d["funnel" + suffix] = df_n["Q009"]
    d["intent" + suffix] = df_n["Q002"]
    d["aspire" + suffix] = df_n["Q001"]

    d["revealD_pre" + suffix] = df_n["Q136"]
    d["revealD_post" + suffix] = df_n["Q137"]

    # choice experiments - just the raw 1/2 codes for now
    d["expA" + suffix] = df_n["Q133"]
    d["expB" + suffix] = df_n["Q134"]
    d["expC" + suffix] = df_n["Q135"]
    d["expE" + suffix] = df_n["Q138"]

    d["friction" + suffix] = (df_n["Q092"] == 1).astype(int)
    d["price_drop" + suffix] = (df_n["Q091"] == 1).astype(int)
    d["visa_est" + suffix] = df_n["Q043"]
    d["stay" + suffix] = df_n["Q039"]

    d["bridge_pct" + suffix] = df_n["Q116"]
    d["fin_worry" + suffix] = df_n["Q118"]

    return d


def add_calibration(D: pd.DataFrame) -> pd.DataFrame:
    """
    overconfidence gap = self-rated knowledge (rescaled to a 0-100
    percentage) minus objective % correct. positive = overconfident.
    """
    D["obj_pct_pre"] = D["adm_know_pre"] / 10 * 100
    D["obj_pct_post"] = D["adm_know_post"] / 10 * 100
    # Q088 is a 1-5 scale, rescale to 0-100 so it's comparable to obj_pct
    D["subj_pct_pre"] = (D["selfrate_know_pre"] - 1) / 4 * 100
    D["subj_pct_post"] = (D["selfrate_know_post"] - 1) / 4 * 100
    D["calib_gap_pre"] = D["subj_pct_pre"] - D["obj_pct_pre"]
    D["calib_gap_post"] = D["subj_pct_post"] - D["obj_pct_post"]
    D["calib_gap_d"] = D["calib_gap_post"] - D["calib_gap_pre"]
    return D


def add_change_scores(D: pd.DataFrame) -> pd.DataFrame:
    change_vars = ["adm_know", "fin_lit", "total_know", "stress", "anx", "seff", "iu",
                   "mood", "trust", "econ", "mobil", "selfrate_know", "selfrate_aid",
                   "selfrate_risk", "funnel", "intent", "aspire", "bridge_pct",
                   "fin_worry", "visa_est"]
    for v in change_vars:
        D[v + "_d"] = D[v + "_post"] - D[v + "_pre"]
    return D


def build_analytic() -> pd.DataFrame:
    pre, post, rct = load_raw()
    area_col = [c for c in pre.columns if "Type of area" in str(c)][0]

    pre_n, post_n = numify(pre), numify(post)
    rct["Id"] = pd.to_numeric(rct["Id"], errors="coerce")

    A = build_wave(pre_n, "_pre")
    B = build_wave(post_n, "_post")

    D = pd.concat([A.reset_index(drop=True), B.reset_index(drop=True)], axis=1)
    D["Id"] = pre_n["Id"].values
    D["Gender"] = pre_n["Gender"].values
    D["Country"] = pre_n["Country"].values
    D["Area"] = pre_n[area_col].values
    D["Age"] = pre_n["Q013"].values
    D["Grade"] = pre_n["Q015"].values
    D["School"] = pre_n["Q016"].values
    D["GPA"] = pre_n["Q018"].values

    D = D.merge(rct[["Id", "Arm", "ArmCode"]], on="Id", how="left")

    D = add_change_scores(D)
    D = add_calibration(D)

    # reveal-effect shift (post-reveal minus pre-reveal likelihood)
    D["revealD_shift_pre"] = D["revealD_post_pre"] - D["revealD_pre_pre"]
    D["revealD_shift_post"] = D["revealD_post_post"] - D["revealD_pre_post"]

    return D


if __name__ == "__main__":
    D = build_analytic()
    D.to_pickle(cfg.ANALYTIC_PATH)
    print(f"built analytic dataset: {D.shape[0]} rows x {D.shape[1]} cols -> {cfg.ANALYTIC_PATH}")
