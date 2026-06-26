import os
import sys
import subprocess
from pathlib import Path

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
# PROJECT_ROOT = the "assignment 2" folder (one level up from this src file).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
print(PROJECT_ROOT)

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"

# the two data files we care about
FULL_DATA = DATA_DIR / "flight_data_2024.csv"          # the big 1.3 GB file
SAMPLE_DATA = DATA_DIR / "flight_data_2024_sample.csv"  # small 10k rows for quick tests

# make sure the output folders exist (does nothing if already there)
for _d in (OUTPUT_DIR, FIG_DIR, TABLE_DIR):
    _d.mkdir(parents=True, exist_ok=True)