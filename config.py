"""
config.py

paths and constants used across the whole analysis. change this if the
data lives somewhere else, don't go hunting through every script.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "output"
FIG_DIR = OUT_DIR / "figures"

PRE_PATH = DATA_DIR / "Pre-Test.xlsx"
POST_PATH = DATA_DIR / "Post-Test.xlsx"
RCT_PATH = DATA_DIR / "RCT_Assignment.xlsx"

ANALYTIC_PATH = OUT_DIR / "ANALYTIC.pkl"
LONG_PATH = OUT_DIR / "LONG.pkl"

RANDOM_SEED = 42
N_BOOT = 2000

# scoring keys - correct answer code per item. don't touch unless the
# instrument itself changes
ADM_KEY = {"Q058": 2, "Q059": 1, "Q060": 2, "Q061": 3, "Q062": 3, "Q063": 4,
           "Q064": 1, "Q065": 3, "Q066": 2, "Q072": 3}
FIN_KEY = {"Q073": 1, "Q074": 3, "Q075": 2}

ARMS = ["Full Course", "Guided Self-Study", "Waitlist"]

SCALES = {
    "stress": ["Q094", "Q095", "Q096", "Q097"],
    "anx": ["Q099", "Q100", "Q101", "Q102", "Q103"],
    "seff": ["Q104", "Q105", "Q106", "Q107"],
    "iu": ["Q108", "Q109", "Q110", "Q111"],
    "mood": ["Q112", "Q113", "Q114", "Q115"],
    "trust": ["Q122", "Q123", "Q124"],
    "econ": ["Q119", "Q120"],
    "mobil": ["Q125", "Q126"],
}

for d in (OUT_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)
