"""
src/hypothesis_tests.py

H1, H2, H3, H5. H4 lives in choice_experiments.py since it's really
a choice-experiment result, and H6 is in intervention_effects.py.
"""
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


def test_h3_knowledge_anxiety(D: pd.DataFrame) -> dict:
    """does objective knowledge predict anxiety, net of demographics?"""
    basic = smf.ols("anx_pre ~ adm_know_pre + C(Gender) + Age + C(Country) + GPA", data=D).fit()

    # adding IU and self-efficacy - this is the model that shows the
    # knowledge coefficient basically disappears, see paper section 7.18
    full = smf.ols("anx_pre ~ adm_know_pre + C(Gender) + Age + C(Country) + iu_pre + seff_pre",
                    data=D).fit()

    return {
        "basic_beta": basic.params["adm_know_pre"],
        "basic_p": basic.pvalues["adm_know_pre"],
        "basic_r2": basic.rsquared,
        "full_beta_know": full.params["adm_know_pre"],
        "full_p_know": full.pvalues["adm_know_pre"],
        "full_beta_iu": full.params["iu_pre"],
        "full_beta_seff": full.params["seff_pre"],
        "full_r2": full.rsquared,
    }


def test_h5_friction(D: pd.DataFrame) -> dict:
    """administrative friction should predict lower funnel progress even
    controlling for knowledge and aspiration."""
    with_friction = D[D.friction_pre == 1]["funnel_pre"].dropna()
    without = D[D.friction_pre == 0]["funnel_pre"].dropna()
    t, p = stats.ttest_ind(without, with_friction)

    model = smf.ols("funnel_pre ~ friction_pre + adm_know_pre + aspire_pre + "
                     "C(Country) + C(Gender) + Age", data=D).fit()

    return {
        "mean_no_friction": without.mean(),
        "mean_friction": with_friction.mean(),
        "t": t, "p": p,
        "controlled_beta": model.params["friction_pre"],
        "controlled_p": model.pvalues["friction_pre"],
    }


def test_h1_information(D: pd.DataFrame) -> dict:
    """objective vs self-rated knowledge as predictors of funnel progress."""
    model = smf.ols("funnel_pre ~ adm_know_pre + selfrate_know_pre + C(Country) + C(Area)",
                     data=D).fit()
    return {
        "beta_objective": model.params["adm_know_pre"],
        "p_objective": model.pvalues["adm_know_pre"],
        "beta_selfrated": model.params["selfrate_know_pre"],
        "p_selfrated": model.pvalues["selfrate_know_pre"],
    }


def test_h2_literacy_migration(D: pd.DataFrame) -> dict:
    groups = [D[D.stay_pre == g]["fin_lit_pre"].dropna() for g in [1, 2, 3]]
    f, p = stats.f_oneway(*groups)
    return {
        "mean_stay": groups[0].mean(),
        "mean_return": groups[1].mean(),
        "mean_undecided": groups[2].mean(),
        "F": f, "p": p,
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import config as cfg

    D = pd.read_pickle(cfg.ANALYTIC_PATH)

    print("H3 - knowledge -> anxiety")
    r = test_h3_knowledge_anxiety(D)
    print(f"  basic model: beta={r['basic_beta']:.4f} p={r['basic_p']:.4f} R2={r['basic_r2']:.3f}")
    print(f"  full model:  beta_know={r['full_beta_know']:.4f} p={r['full_p_know']:.4f} "
          f"(this is where it falls apart once IU/self-efficacy are in)")

    print("\nH5 - friction -> funnel stage")
    r = test_h5_friction(D)
    print(f"  no friction={r['mean_no_friction']:.2f}  friction={r['mean_friction']:.2f}  "
          f"t={r['t']:.2f}  p={r['p']:.4f}")

    print("\nH1 - information access -> funnel progress")
    r = test_h1_information(D)
    print(f"  objective know beta={r['beta_objective']:.4f} p={r['p_objective']:.4f}")
    print(f"  self-rated know beta={r['beta_selfrated']:.4f} p={r['p_selfrated']:.4f}")

    print("\nH2 - financial literacy -> migration intent (this one's null)")
    r = test_h2_literacy_migration(D)
    print(f"  F={r['F']:.3f} p={r['p']:.3f}")
