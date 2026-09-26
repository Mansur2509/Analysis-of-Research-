"""
src/mediation.py

this is the part of the analysis that actually matters most for the
paper. the course reduces anxiety, sure, but does it do that through
the knowledge it teaches? short answer: not really. this file is
where that gets tested.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import config as cfg


def bootstrap_indirect(data: pd.DataFrame, treat_col: str, mediator: str, outcome: str,
                        n_boot: int = cfg.N_BOOT, seed: int = cfg.RANDOM_SEED) -> dict:
    """
    standard a*b mediation decomposition with a bootstrap CI on the
    indirect effect, because the sampling distribution of a product of
    two coefficients isn't normal and a delta-method SE would lie to you.
    """
    a_model = smf.ols(f"{mediator} ~ {treat_col}", data=data).fit()
    a = a_model.params[treat_col]

    b_model = smf.ols(f"{outcome} ~ {mediator} + {treat_col}", data=data).fit()
    b = b_model.params[mediator]
    c_prime = b_model.params[treat_col]

    c_total = smf.ols(f"{outcome} ~ {treat_col}", data=data).fit().params[treat_col]

    rng = np.random.default_rng(seed)
    boot_indirects = []
    for _ in range(n_boot):
        resampled = data.sample(len(data), replace=True, random_state=int(rng.integers(1_000_000_000)))
        try:
            aa = smf.ols(f"{mediator} ~ {treat_col}", data=resampled).fit().params[treat_col]
            bb = smf.ols(f"{outcome} ~ {mediator} + {treat_col}", data=resampled).fit().params[mediator]
            boot_indirects.append(aa * bb)
        except Exception:
            # occasionally a resample gives a singular matrix, just skip it
            continue

    ci_low, ci_high = np.percentile(boot_indirects, [2.5, 97.5])
    indirect = a * b

    return {
        "a": a, "b": b, "c_total": c_total, "c_prime": c_prime,
        "indirect": indirect,
        "prop_mediated": indirect / c_total if c_total != 0 else None,
        "boot_ci": [ci_low, ci_high],
        "n_boot_used": len(boot_indirects),
    }


def horse_race(D: pd.DataFrame):
    """all four candidate channels in one model - whichever survives
    with the treatment indicator still significant is doing the real work."""
    return smf.ols("anx_d ~ adm_know_d + seff_d + iu_d + calib_gap_d + treatF + C(Country)",
                    data=D).fit()


if __name__ == "__main__":
    D = pd.read_pickle(cfg.ANALYTIC_PATH)
    D["treatF"] = (D["ArmCode"] == 1).astype(int)  # full course vs everyone else

    for mediator, label in [("adm_know_d", "knowledge gain"),
                             ("seff_d", "self-efficacy gain"),
                             ("iu_d", "intolerance-of-uncertainty reduction")]:
        r = bootstrap_indirect(D, "treatF", mediator, "anx_d")
        pct = r["prop_mediated"] * 100 if r["prop_mediated"] else float("nan")
        print(f"{label}: a={r['a']:.3f} b={r['b']:.4f} indirect={r['indirect']:.4f} "
              f"prop_mediated={pct:.1f}% ci={[round(x,4) for x in r['boot_ci']]}")

    print("\nmulti-mediator horse race:")
    m = horse_race(D)
    for term in ["adm_know_d", "seff_d", "iu_d", "calib_gap_d", "treatF"]:
        print(f"  {term:14s} beta={m.params[term]:+.4f}  p={m.pvalues[term]:.4f}")
    print(f"  R2 = {m.rsquared:.3f}")
    print("\n  treatF should be way bigger than everything else here - that's the point")
