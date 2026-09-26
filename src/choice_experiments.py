"""
src/choice_experiments.py

the five forced-choice tasks. mostly just binomial tests against a
50/50 split, plus the reveal experiment which is its own thing.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

import config as cfg

EXPERIMENTS = {
    "expA_pre": "Country vs Quality",
    "expB_pre": "Prestige vs Fit",
    "expC_pre": "Scholarship Risk",
    "expE_pre": "Education vs Migration",
}


def binomial_summary(D: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, name in EXPERIMENTS.items():
        v = D[col].dropna()
        n = len(v)
        n_opt1 = int((v == 1).sum())
        bt = stats.binomtest(n_opt1, n, 0.5)
        rows.append({"experiment": name, "pct_option1": n_opt1 / n * 100,
                     "p_binomial": bt.pvalue, "n": n})
    return pd.DataFrame(rows)


def risk_logit(D: pd.DataFrame):
    """what predicts choosing the gamble in Experiment C?"""
    D = D.copy()
    D["expC_bin"] = (D["expC_pre"] == 2).astype(int)
    return smf.logit("expC_bin ~ selfrate_risk_pre + anx_pre + fin_lit_pre + C(Gender) + C(Country)",
                      data=D).fit(disp=0)


def reveal_effect(D: pd.DataFrame) -> dict:
    """experiment D - sticker price then net price reveal."""
    m = D[["revealD_pre_pre", "revealD_post_pre"]].dropna()
    t, p = stats.ttest_rel(m["revealD_post_pre"], m["revealD_pre_pre"])
    shift = (m["revealD_post_pre"] - m["revealD_pre_pre"])
    return {
        "pre_reveal_mean": m["revealD_pre_pre"].mean(),
        "post_reveal_mean": m["revealD_post_pre"].mean(),
        "shift": shift.mean(),
        "t": t, "p": p,
        "pct_increased": (shift > 0).mean() * 100,
    }


def reveal_attenuation_by_arm(D: pd.DataFrame) -> pd.DataFrame:
    """manipulation check - treated respondents should be less surprised
    at post-test than at baseline, since the course teaches this exact thing."""
    rows = []
    for arm in cfg.ARMS:
        s = D[D.Arm == arm]
        rows.append({"arm": arm, "shift_pre": s["revealD_shift_pre"].mean(),
                     "shift_post": s["revealD_shift_post"].mean()})
    df = pd.DataFrame(rows)
    df["attenuation"] = df["shift_post"] - df["shift_pre"]
    return df


if __name__ == "__main__":
    D = pd.read_pickle(cfg.ANALYTIC_PATH)

    print(binomial_summary(D).to_string(index=False))

    print("\nbreakdown by country, experiment A:")
    print(D.groupby("Country")["expA_pre"].apply(lambda x: (x == 1).mean() * 100))

    print("\nrisk logit:")
    m = risk_logit(D)
    print(f"  pseudo-R2={m.prsquared:.3f}")
    print(f"  selfrate_risk OR={np.exp(m.params['selfrate_risk_pre']):.2f} "
          f"p={m.pvalues['selfrate_risk_pre']:.4f}")

    print("\nreveal effect:")
    r = reveal_effect(D)
    print(f"  {r['pre_reveal_mean']:.2f} -> {r['post_reveal_mean']:.2f}  "
          f"shift={r['shift']:+.2f}  t={r['t']:.2f}  {r['pct_increased']:.1f}% increased")

    print("\nattenuation by arm (this is the manipulation check):")
    print(reveal_attenuation_by_arm(D).to_string(index=False))
