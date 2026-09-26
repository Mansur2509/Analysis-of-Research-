"""
tests/test_mediation.py

builds a fake dataset where we already know the true a, b, and indirect
effect, then checks bootstrap_indirect actually recovers something close.
also a separate case where the mediator does nothing, to make sure the
function doesn't just always find a mediation effect out of nowhere.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest

from mediation import bootstrap_indirect


def make_mediation_data(n=3000, a_true=0.5, b_true=0.4, c_prime_true=0.1, seed=1):
    """
    treat -> mediator (strength a_true) -> outcome (strength b_true),
    plus a direct treat -> outcome path (c_prime_true). true indirect
    effect is a_true * b_true, true total is c_prime_true + a_true*b_true.
    """
    rng = np.random.default_rng(seed)
    treat = rng.integers(0, 2, n)
    mediator = a_true * treat + rng.normal(size=n) * 0.5
    outcome = b_true * mediator + c_prime_true * treat + rng.normal(size=n) * 0.5
    return pd.DataFrame({"treat": treat, "mediator": mediator, "outcome": outcome})


class TestBootstrapIndirect:

    def test_recovers_known_indirect_effect(self):
        data = make_mediation_data(a_true=0.5, b_true=0.4, c_prime_true=0.1)
        r = bootstrap_indirect(data, "treat", "mediator", "outcome", n_boot=500)

        true_indirect = 0.5 * 0.4
        # bootstrap is noisy, give it some room
        assert abs(r["indirect"] - true_indirect) < 0.05

    def test_ci_does_not_contain_zero_for_real_effect(self):
        data = make_mediation_data(a_true=0.6, b_true=0.5)
        r = bootstrap_indirect(data, "treat", "mediator", "outcome", n_boot=500)
        lo, hi = r["boot_ci"]
        assert lo > 0  # a real mediation effect shouldn't straddle zero

    def test_no_mediation_when_b_path_is_zero(self):
        # mediator is affected by treatment but doesn't affect the outcome at all
        data = make_mediation_data(a_true=0.6, b_true=0.0, c_prime_true=0.3)
        r = bootstrap_indirect(data, "treat", "mediator", "outcome", n_boot=500)
        lo, hi = r["boot_ci"]
        # should straddle zero since there's genuinely no b path
        assert lo < 0 < hi

    def test_prop_mediated_between_reasonable_bounds(self):
        # this is basically our real finding - small mediated proportion,
        # most of the effect is direct
        data = make_mediation_data(a_true=0.4, b_true=0.05, c_prime_true=0.5)
        r = bootstrap_indirect(data, "treat", "mediator", "outcome", n_boot=500)
        assert r["prop_mediated"] < 0.15  # should be small, not the whole effect


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
