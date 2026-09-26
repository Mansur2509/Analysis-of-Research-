"""
tests/test_stats_utils.py

checks the shared helper functions. cohens_d gets tested against a
hand-calculable case since it's used in basically every results
section of the paper - if this is wrong everything downstream is wrong.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest

from stats_utils import cohens_d, eta_squared, tost_two_sample, tost_corr


class TestCohensD:

    def test_known_case(self):
        # two groups, same sd=1, means differ by exactly 1 -> d should be 1
        a = pd.Series(np.random.default_rng(1).normal(1, 1, 5000))
        b = pd.Series(np.random.default_rng(2).normal(0, 1, 5000))
        d = cohens_d(a, b)
        assert abs(d - 1.0) < 0.1

    def test_zero_when_means_equal(self):
        rng = np.random.default_rng(3)
        a = pd.Series(rng.normal(0, 1, 3000))
        b = pd.Series(rng.normal(0, 1, 3000))
        d = cohens_d(a, b)
        assert abs(d) < 0.1

    def test_sign_flips_with_order(self):
        a = pd.Series([5, 6, 7, 8])
        b = pd.Series([1, 2, 3, 4])
        assert cohens_d(a, b) > 0
        assert cohens_d(b, a) < 0


class TestEtaSquared:

    def test_no_group_difference_near_zero(self):
        rng = np.random.default_rng(4)
        groups = [pd.Series(rng.normal(0, 1, 500)) for _ in range(3)]
        eta = eta_squared(groups)
        assert eta < 0.02

    def test_large_group_difference_high_eta(self):
        groups = [pd.Series([0]*100), pd.Series([10]*100), pd.Series([20]*100)]
        eta = eta_squared(groups)
        assert eta > 0.9

    def test_bounded_zero_to_one(self):
        rng = np.random.default_rng(6)
        groups = [pd.Series(rng.normal(i, 2, 200)) for i in range(3)]
        eta = eta_squared(groups)
        assert 0 <= eta <= 1


class TestTost:

    def test_truly_equivalent_groups_pass(self):
        # same distribution, should come out equivalent at a generous bound
        rng = np.random.default_rng(7)
        a = pd.Series(rng.normal(0, 1, 2000))
        b = pd.Series(rng.normal(0, 1, 2000))
        r = tost_two_sample(a, b, bound_d=0.3)
        assert bool(r["equivalent"])

    def test_clearly_different_groups_fail(self):
        rng = np.random.default_rng(8)
        a = pd.Series(rng.normal(0, 1, 2000))
        b = pd.Series(rng.normal(2, 1, 2000))  # huge gap
        r = tost_two_sample(a, b, bound_d=0.2)
        assert not r["equivalent"]

    def test_corr_equivalence_near_zero(self):
        rng = np.random.default_rng(9)
        x = pd.Series(rng.normal(size=3000))
        y = pd.Series(rng.normal(size=3000))  # unrelated
        r = tost_corr(x, y, bound_r=0.1)
        assert bool(r["equivalent"])

    def test_corr_not_equivalent_when_real_relationship(self):
        rng = np.random.default_rng(10)
        x = pd.Series(rng.normal(size=2000))
        y = x * 0.5 + pd.Series(rng.normal(size=2000)) * 0.5  # real correlation
        r = tost_corr(x, y, bound_r=0.1)
        assert not r["equivalent"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
