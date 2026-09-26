"""
run_all.py

runs the whole thing start to finish. order matters - data_prep has
to go first, intervention_effects has to go before make_figures since
it's what builds LONG.pkl.

if you just want one piece, you can run any src/*.py file on its own,
they all work standalone as long as ANALYTIC.pkl already exists.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

PIPELINE = [
    "data_prep.py",
    "data_quality.py",
    "reliability.py",
    "balance.py",
    "hypothesis_tests.py",
    "intervention_effects.py",   # builds LONG.pkl - make_figures needs this
    "mediation.py",
    "calibration.py",
    "choice_experiments.py",
    "equivalence.py",
    "segmentation.py",
    "make_figures.py",
]

if __name__ == "__main__":
    for script in PIPELINE:
        path = SRC / script
        print(f"\n{'='*60}\n{script}\n{'='*60}")
        result = subprocess.run([sys.executable, str(path)])
        if result.returncode != 0:
            print(f"stopped at {script}, something broke")
            sys.exit(1)

    print("\ndone. run 'pytest tests/' separately if you want to check the code itself.")
