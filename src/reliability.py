"""
src/reliability.py

alpha and omega for every multi-item scale. omega does most of the
work here because half our scales are only 2-3 items and alpha just
isn't fair to those - see the docstring below for why.
"""
import numpy as np
import pandas as pd
from numpy.linalg import eig


def cronbach_alpha(items: pd.DataFrame) -> float:
    """
    standard Cronbach's alpha.

    alpha = (k / (k-1)) * (1 - sum(item variances) / variance(total score))

    Parameters
    ----------
    items : DataFrame
        one column per item, same scale, rows = respondents.

    Returns
    -------
    float
    """
    items = items.dropna()
    k = items.shape[1]
    item_var_sum = items.var(ddof=1).sum()
    total_var = items.sum(axis=1).var(ddof=1)
    return (k / (k - 1)) * (1 - item_var_sum / total_var)


def mcdonald_omega(items: pd.DataFrame) -> float:
    """
    single-factor omega via eigendecomposition of the correlation matrix.

    alpha assumes every item loads on the factor equally (tau-equivalence).
    that's basically never true, and it's especially not true for short
    scales, where alpha ends up underestimating reliability pretty badly.
    omega doesn't make that assumption, so we use it as the main number
    for anything under ~4 items.

    this isn't a full CFA, it's the quick version - first-eigenvector
    loadings, not iteratively fit factor loadings. good enough here.
    """
    items = items.dropna()
    corr = items.corr().values
    eigvals, eigvecs = eig(corr)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order].real, eigvecs[:, order].real

    loadings = np.abs(eigvecs[:, 0] * np.sqrt(eigvals[0]))
    sum_sq_loadings = loadings.sum() ** 2
    sum_uniqueness = (1 - loadings ** 2).sum()
    sum_uniqueness = max(sum_uniqueness, 0.01)  # guard divide-by-zero on weird data

    return sum_sq_loadings / (sum_sq_loadings + sum_uniqueness)


def item_total_correlations(items: pd.DataFrame) -> dict:
    """
    corrected item-total r and alpha-if-deleted for every item.
    useful for spotting a bad item before you trust the composite.
    """
    items = items.dropna()
    result = {}
    for col in items.columns:
        rest = items.drop(columns=col).sum(axis=1)
        r = np.corrcoef(items[col], rest)[0, 1]

        remaining_cols = [c for c in items.columns if c != col]
        if len(remaining_cols) > 1:
            a_deleted = cronbach_alpha(items[remaining_cols])
        else:
            a_deleted = np.nan

        result[col] = {"item_total_r": r, "alpha_if_deleted": a_deleted}
    return result


def reliability_report(items: pd.DataFrame) -> dict:
    """convenience wrapper - alpha, omega, and n in one call."""
    return {
        "alpha": cronbach_alpha(items),
        "omega": mcdonald_omega(items),
        "n": len(items.dropna()),
        "k": items.shape[1],
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import config as cfg

    pre = pd.read_excel(cfg.PRE_PATH).apply(pd.to_numeric, errors="coerce")
    post = pd.read_excel(cfg.POST_PATH).apply(pd.to_numeric, errors="coerce")

    all_scales = dict(cfg.SCALES)
    all_scales["fin_lit_big3"] = ["Q073", "Q074", "Q075"]
    all_scales["perceived_lit"] = ["Q088", "Q089", "Q090"]
    all_scales["adm_knowledge"] = ["Q058", "Q059", "Q060", "Q061", "Q062",
                                    "Q063", "Q064", "Q065", "Q066", "Q072"]

    for name, cols in all_scales.items():
        rep_pre = reliability_report(pre[cols])
        rep_post = reliability_report(post[cols])
        print(f"{name:16s} k={rep_pre['k']}  "
              f"alpha_pre={rep_pre['alpha']:.3f}  omega_pre={rep_pre['omega']:.3f}  "
              f"alpha_post={rep_post['alpha']:.3f}  omega_post={rep_post['omega']:.3f}")
