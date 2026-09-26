"""
src/stats_utils.py

small helper functions that a few different scripts need. cohens_d
especially gets used everywhere so it lives here once instead of
copy-pasted five times.
"""
import numpy as np
import pandas as pd
from scipy import stats


def cohens_d(a: pd.Series, b: pd.Series) -> float:
    """pooled-sd cohen's d between two independent samples."""
    a, b = a.dropna(), b.dropna()
    n1, n2 = len(a), len(b)
    pooled_sd = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    return (a.mean() - b.mean()) / pooled_sd


def eta_squared(groups: list) -> float:
    grand = pd.concat(groups)
    ss_between = sum(len(g) * (g.mean() - grand.mean()) ** 2 for g in groups)
    ss_total = ((grand - grand.mean()) ** 2).sum()
    return ss_between / ss_total


def fmt_p(p: float) -> str:
    """p<.001 style formatting, because APA wants it that way."""
    return "<.001" if p < .001 else f"{p:.3f}".lstrip("0") if p < 1 else f"{p:.3f}"


def tost_two_sample(a: pd.Series, b: pd.Series, bound_d: float = 0.20) -> dict:
    """
    two one-sided tests for equivalence, bound expressed as cohen's d.
    a significant TOST means the effect is smaller than the bound -
    i.e. we can say the groups are equivalent, not just "not different."
    """
    a, b = a.dropna(), b.dropna()
    n1, n2 = len(a), len(b)
    pooled_sd = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    diff = a.mean() - b.mean()
    se = pooled_sd * np.sqrt(1 / n1 + 1 / n2)
    bound = bound_d * pooled_sd
    df = n1 + n2 - 2

    p_lower = 1 - stats.t.cdf((diff - (-bound)) / se, df)
    p_upper = stats.t.cdf((diff - bound) / se, df)
    p_tost = max(p_lower, p_upper)

    return {"d": diff / pooled_sd, "p_tost": p_tost, "equivalent": p_tost < 0.05}


def tost_corr(x: pd.Series, y: pd.Series, bound_r: float = 0.10) -> dict:
    """same idea as tost_two_sample but for a correlation, via fisher z."""
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    n = len(df)
    r = np.corrcoef(df.x, df.y)[0, 1]
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    zb = np.arctanh(bound_r)

    p_lower = 1 - stats.norm.cdf((z - (-zb)) / se)
    p_upper = stats.norm.cdf((z - zb) / se)
    p_tost = max(p_lower, p_upper)

    return {"r": r, "p_tost": p_tost, "equivalent": p_tost < 0.05, "n": n}
