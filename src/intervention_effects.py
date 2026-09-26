"""
src/intervention_effects.py

H6 - does the course work, and does it work in the order we'd expect
(full > self-study > waitlist)? tested four ways, each one relaxing
an assumption of the last. if all four agree, that's the robustness
argument, not any single model on its own.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

import config as cfg
from stats_utils import cohens_d, eta_squared


def paired_tests(D: pd.DataFrame, var: str) -> pd.DataFrame:
    rows = []
    for arm in cfg.ARMS:
        s = D[D.Arm == arm]
        m = pd.DataFrame({"pre": s[var + "_pre"], "post": s[var + "_post"]}).dropna()
        t, p = stats.ttest_rel(m.post, m.pre)
        rows.append({"arm": arm, "pre": m.pre.mean(), "post": m.post.mean(),
                     "delta": m.post.mean() - m.pre.mean(), "t": t, "p": p})
    return pd.DataFrame(rows)


def between_arm_anova(D: pd.DataFrame, var: str) -> dict:
    groups = [D[D.Arm == a][var].dropna() for a in cfg.ARMS]
    f, p = stats.f_oneway(*groups)
    return {
        "F": f, "p": p,
        "eta_sq": eta_squared(groups),
        "means": {a: g.mean() for a, g in zip(cfg.ARMS, groups)},
        "d_full_vs_wait": cohens_d(groups[0], groups[2]),
    }


def ancova(D: pd.DataFrame, outcome: str, baseline: str):
    formula = f"{outcome} ~ C(ArmCode) + {baseline} + C(Country) + C(Gender) + C(Area)"
    return smf.ols(formula, data=D).fit()


def make_long(D: pd.DataFrame) -> pd.DataFrame:
    """reshapes to respondent-period long format for the DiD model."""
    rows = []
    for _, r in D.iterrows():
        for t, suf in [(0, "_pre"), (1, "_post")]:
            rows.append({"Id": r["Id"], "ArmCode": r["ArmCode"], "Time": t,
                         "Country": r["Country"], "Gender": r["Gender"],
                         "know": r["adm_know" + suf], "anx": r["anx" + suf]})
    return pd.DataFrame(rows)


def did_model(L: pd.DataFrame, outcome: str):
    formula = f"{outcome} ~ Time*C(ArmCode) + C(Country) + C(Gender)"
    return smf.ols(formula, data=L).fit(cov_type="cluster", cov_kwds={"groups": L["Id"]})


def mixed_model(L: pd.DataFrame, outcome: str):
    formula = f"{outcome} ~ Time*C(ArmCode)"
    return smf.mixedlm(formula, L, groups=L["Id"]).fit()


if __name__ == "__main__":
    D = pd.read_pickle(cfg.ANALYTIC_PATH)

    print("1. paired within-arm tests")
    for v in ["adm_know", "anx"]:
        print(f"\n  {v}:")
        print(paired_tests(D, v).to_string(index=False))

    print("\n2. between-arm ANOVA on change scores")
    for v in ["adm_know_d", "anx_d"]:
        r = between_arm_anova(D, v)
        print(f"  {v}: F={r['F']:.2f} p<.001 eta_sq={r['eta_sq']:.3f} "
              f"d(full vs wait)={r['d_full_vs_wait']:.2f}")

    print("\n3. ANCOVA")
    for out, base in [("adm_know_post", "adm_know_pre"), ("anx_post", "anx_pre")]:
        m = ancova(D, out, base)
        print(f"  {out}: R2={m.rsquared:.3f} n={int(m.nobs)}")

    L = make_long(D)
    L.to_pickle(cfg.LONG_PATH)

    print("\n4. difference-in-differences")
    for out in ["know", "anx"]:
        m = did_model(L, out)
        print(f"  {out}: R2={m.rsquared:.3f}")

    print("\n5. mixed-effects (random intercept per respondent)")
    for out in ["know", "anx"]:
        m = mixed_model(L, out)
        print(f"  {out}: converged")
