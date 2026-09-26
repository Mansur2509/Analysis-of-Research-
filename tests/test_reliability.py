"""
tests/test_reliability.py

sanity checks for the reliability functions. the main idea: build a
fake dataset where we already know what alpha/omega SHOULD be, then
check the function gets close. if these break, don't trust anything
that used cronbach_alpha or mcdonald_omega downstream.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest

from reliability import cronbach_alpha, mcdonald_omega, item_total_correlations


def make_correlated_items(n=1000, k=5, true_corr=0.6, seed=1):
    """
    items that share a common factor at strength true_corr. higher
    true_corr should give higher alpha - this is the easiest sanity
    check there is.
    """
    rng = np.random.default_rng(seed)
    factor = rng.normal(size=n)
    items = {}
    for i in range(k):
        noise = rng.normal(size=n)
        items[f"item{i}"] = true_corr * factor + np.sqrt(1 - true_corr**2) * noise
    return pd.DataFrame(items)


def make_independent_items(n=1000, k=5, seed=2):
    """items with no shared factor at all - alpha should come out near zero."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({f"item{i}": rng.normal(size=n) for i in range(k)})


class TestCronbachAlpha:

    def test_high_correlation_gives_high_alpha(self):
        items = make_correlated_items(true_corr=0.8)
        alpha = cronbach_alpha(items)
        assert alpha > 0.85

    def test_independent_items_give_low_alpha(self):
        items = make_independent_items()
        alpha = cronbach_alpha(items)
        # not exactly 0 because of sampling noise, but should be low
        assert alpha < 0.15

    def test_alpha_increases_with_more_items(self):
        # spearman-brown says more items = higher alpha at fixed avg correlation
        items_3 = make_correlated_items(k=3, true_corr=0.5, seed=10)
        items_6 = make_correlated_items(k=6, true_corr=0.5, seed=10)
        assert cronbach_alpha(items_6) > cronbach_alpha(items_3)

    def test_handles_missing_data(self):
        items = make_correlated_items()
        items.iloc[0:20, 0] = np.nan
        # shouldn't crash, and should still be reasonable
        alpha = cronbach_alpha(items)
        assert 0 <= alpha <= 1


class TestMcDonaldOmega:

    def test_omega_reasonable_range(self):
        items = make_correlated_items(true_corr=0.7)
        omega = mcdonald_omega(items)
        assert 0 <= omega <= 1
        assert omega > 0.7  # should track the true factor loading roughly

    def test_omega_gte_alpha_when_loadings_unequal(self):
        # this is the actual reason we use omega for short scales -
        # when items don't load equally, omega should be >= alpha
        rng = np.random.default_rng(5)
        n = 2000
        factor = rng.normal(size=n)
        # deliberately unequal loadings: .3, .5, .9
        items = pd.DataFrame({
            "a": 0.3 * factor + rng.normal(size=n) * 0.95,
            "b": 0.5 * factor + rng.normal(size=n) * 0.87,
            "c": 0.9 * factor + rng.normal(size=n) * 0.44,
        })
        a = cronbach_alpha(items)
        o = mcdonald_omega(items)
        # not a strict mathematical guarantee in every finite sample, but
        # should hold here given how unequal the loadings are
        assert o >= a - 0.05


class TestItemTotalCorrelations:

    def test_bad_item_gets_flagged(self):
        # 4 good items + 1 that's pure noise, unrelated to the others
        items = make_correlated_items(k=4, true_corr=0.7, seed=3)
        rng = np.random.default_rng(99)
        items["junk_item"] = rng.normal(size=len(items))

        result = item_total_correlations(items)
        junk_r = result["junk_item"]["item_total_r"]
        good_r = result["item0"]["item_total_r"]

        assert junk_r < good_r
        # removing the junk item should raise alpha, not lower it
        assert result["junk_item"]["alpha_if_deleted"] > cronbach_alpha(items)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
