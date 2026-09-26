"""
src/calibration.py

people think they know more than they do, and this is where we
measure exactly how much. also checks whether the course fixed it.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from scipy import stats

import config as cfg
from stats_utils import tost_two_sample


def calibration_summary(D: pd.DataFrame) -> dict:
    return {
        "mean_objective_pct": D["obj_pct_pre"].mean(),
        "mean_subjective_pct": D["subj_pct_pre"].mean(),
        "mean_gap": D["calib_gap_pre"].mean(),
        "obj_subj_r": D[["adm_know_pre", "selfrate_know_pre"]].corr().iloc[0, 1],
        "pct_overconfident": (D["calib_gap_pre"] > 0).mean() * 100,
    }


def gender_gap(D: pd.DataFrame) -> dict:
    """H3a - turns out this one doesn't hold up, see tost check below too."""
    male = D[D.Gender == 1]["calib_gap_pre"].dropna()
    female = D[D.Gender == 2]["calib_gap_pre"].dropna()
    t, p = stats.ttest_ind(male, female)
    tost = tost_two_sample(male, female)
    return {"male_gap": male.mean(), "female_gap": female.mean(), "t": t, "p": p,
            "tost": tost}


def dunning_kruger_check(D: pd.DataFrame) -> pd.Series:
    """overconfidence should be worst in the bottom knowledge quartile."""
    D = D.copy()
    D["_kq"] = pd.qcut(D["adm_know_pre"], 4, labels=["Q1_lowest", "Q2", "Q3", "Q4_highest"],
                        duplicates="drop")
    return D.groupby("_kq")["calib_gap_pre"].mean()


def calibration_by_arm(D: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm in cfg.ARMS:
        s = D[D.Arm == arm]
        rows.append({"arm": arm, "gap_pre": s["calib_gap_pre"].mean(),
                     "gap_post": s["calib_gap_post"].mean(),
                     "change": s["calib_gap_d"].mean()})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    D = pd.read_pickle(cfg.ANALYTIC_PATH)

    print("overall:", calibration_summary(D))

    g = gender_gap(D)
    print(f"\ngender gap: male={g['male_gap']:.1f} female={g['female_gap']:.1f} "
          f"t={g['t']:.2f} p={g['p']:.3f}")
    print(f"  tost: d={g['tost']['d']:.3f} p_tost={g['tost']['p_tost']:.4f} "
          f"equivalent={g['tost']['equivalent']}  (this is the actually informative part)")

    print("\ndunning-kruger check:")
    print(dunning_kruger_check(D))

    print("\ncalibration change by arm:")
    print(calibration_by_arm(D).to_string(index=False))
