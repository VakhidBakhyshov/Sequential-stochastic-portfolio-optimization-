import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path

INPUT_PATH = Path.cwd() / "datasets" / "csv"
OUTPUT = INPUT_PATH.parent / "excel"


# PURE_SP500_BENCHMARK = ["SPY"]
PURE_SP500_BENCHMARK = ["SPY", "VSPY", "ZSPY"]

SP500_VARIANTS = [
    'SPYT', 'GSPY', 'KSPY', 'SPYM',
    'SPYG', 'SPYD', 'SPYV', 'SPY',
    'SPYX', 'SPYC', 'SSPY', 'DSPY',
    'ZSPY', 'VSPY', 'YSPY','TSPY'
]

EXCLUDE_OR_CHECK = ["VSPY", "ZSPY"]
