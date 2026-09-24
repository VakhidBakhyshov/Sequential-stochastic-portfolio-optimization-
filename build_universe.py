"""Build the point-in-time eligibility workbook from the raw panels (Section 2.2).

    python build_universe.py
Reads datasets/csv/NewClosePrice.csv, OpenPrice.csv, Volume.csv and Meta.csv and writes
datasets/excel/last_filtered_weights.xlsx (one sheet per rebalance date, columns Key and Value), the
workbook the experiment arms read. Takes about half a minute.
"""
import runpy, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
runpy.run_path(str(ROOT / "scripts" / "dataloader" / "liquidity_screen.py"), run_name="__main__")
