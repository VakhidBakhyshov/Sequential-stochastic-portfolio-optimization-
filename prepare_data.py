"""Rebuild the derived daily-return panels from the raw close file (optional; the shipped files are identical).

    python prepare_data.py
Reads datasets/csv/NewClosePrice.csv and writes datasets/excel/new_etf_returns.csv, new_etf_ewma.csv,
new.csv and business_dates.xlsx.
"""
import runpy, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
runpy.run_path(str(ROOT / "scripts" / "dataloader" / "parse_close.py"), run_name="__main__")
