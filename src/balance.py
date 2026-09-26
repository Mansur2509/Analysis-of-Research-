"""
src/balance.py

checks that the three arms don't differ on baseline covariates. they
shouldn't, since assignment was random - if they do, something's wrong
with either the randomization or the data.
"""
import pandas as pd
from scipy import stats


def balance_continuous(D: pd.DataFrame, cols: list, arm_col: str = "ArmCode") -> pd.DataFrame:
    rows = []
    for c in cols:
        groups = [D[D[arm_col] == a][c].dropna() for a in sorted(D[arm_col].dropna().unique())]
        f, p = stats.f_oneway(*groups)
        rows.append({"variable": c, "F": f, "p": p, "balanced": p > 0.05})
    return pd.DataFrame(rows)


def balance_categorical(D: pd.DataFrame, cols: list, arm_col: str = "ArmCode") -> pd.DataFrame:
    rows = []
    for c in cols:
        ct = pd.crosstab(D[c], D[arm_col])
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        rows.append({"variable": c, "chi2": chi2, "p": p, "dof": dof, "balanced": p > 0.05})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import config as cfg

    D = pd.read_pickle(cfg.ANALYTIC_PATH)

    cont = balance_continuous(D, ["adm_know_pre", "anx_pre", "funnel_pre", "seff_pre",
                                   "iu_pre", "GPA", "Age"])
    print("continuous covariates:")
    print(cont.to_string(index=False))

    cat = balance_categorical(D, ["Gender", "Country", "Area"])
    print("\ncategorical covariates:")
    print(cat.to_string(index=False))

    print("\narm sizes:")
    print(D["Arm"].value_counts())
