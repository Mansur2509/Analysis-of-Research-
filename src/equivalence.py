"""
src/equivalence.py

a null p-value doesn't tell you whether an effect is actually zero or
just underpowered. TOST does. we ran this on every null result in the
study - a couple of the "clean" negative controls turned out to be
inconclusive once we actually checked, which was a little annoying
but better to know.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

import config as cfg
from stats_utils import tost_two_sample, tost_corr


def run_all_tost(D: pd.DataFrame) -> pd.DataFrame:
    results = []

    r = tost_two_sample(D[D.stay_pre == 1]["fin_lit_pre"], D[D.stay_pre == 2]["fin_lit_pre"])
    results.append({"test": "H2: literacy x migration intent", **r})

    r = tost_two_sample(D[D.Gender == 1]["calib_gap_pre"], D[D.Gender == 2]["calib_gap_pre"])
    results.append({"test": "H3a: gender x overconfidence", **r})

    # this one is NOT expected to be equivalent - it's checking whether
    # the mediator actually relates to the outcome at all, and it does
    r = tost_corr(D["adm_know_d"], D["anx_d"])
    results.append({"test": "mediation b-path: knowledge->anxiety", **r})

    r = tost_two_sample(D[D.Arm == "Full Course"]["trust_d"], D[D.Arm == "Waitlist"]["trust_d"])
    results.append({"test": "negative control: institutional trust", **r})

    r = tost_two_sample(D[D.Arm == "Full Course"]["mobil_d"], D[D.Arm == "Waitlist"]["mobil_d"])
    results.append({"test": "negative control: social mobility", **r})

    return pd.DataFrame(results)


if __name__ == "__main__":
    D = pd.read_pickle(cfg.ANALYTIC_PATH)
    df = run_all_tost(D)
    print(df.to_string(index=False))
