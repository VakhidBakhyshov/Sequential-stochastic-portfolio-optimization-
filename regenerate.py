"""Regenerate the two BASE model arms from scratch, from the raw inputs in datasets/.

Runs the model engine (scripts/) for main_dyn_off and main_dyn_strong using the YAML configs in
scripts/configs/, reading datasets/csv/{NewClosePrice,Volume}.csv and
datasets/excel/{business_dates,new_etf_returns,new_etf_ewma,last_filtered_weights}, then moves the
outputs into data/. After this, re-run FINAL_results.ipynb to rebuild every table and figure (the
overlay arms, benchmarks, metrics and charts are all derived inside the notebook).

Usage:  python regenerate.py        (run from the FINAL/ folder; takes a few minutes)
"""
import os, sys, shutil, subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIGS = ["main_dyn_off.yaml", "main_dyn_strong.yaml"]   # Static and Dynamic-base
env = os.environ.copy(); env["PYTHONPATH"] = str(HERE)

for cfg in CONFIGS:
    print(f"\n=== regenerating {cfg} (this runs the full walk-forward backtest) ===", flush=True)
    subprocess.run([sys.executable, "-m", "scripts.runs.run", "-c", cfg],
                   cwd=str(HERE), env=env, check=True)

import time
need = {"main_dyn_off": ["pnl.csv", "weights.xlsx", "real.csv"],
        "main_dyn_strong": ["pnl.csv", "weights.xlsx", "real.csv", "forecast_risk.csv", "dynamic_parameter_history.csv"]}
for arm, files in need.items():                       # copy-with-retry (Windows-safe)
    (HERE/"data"/arm).mkdir(parents=True, exist_ok=True)
    for f in files:
        for _ in range(6):
            try: shutil.copy2(HERE/"results"/arm/f, HERE/"data"/arm/f); break
            except PermissionError: time.sleep(1.0)
    print(f"wrote data/{arm}")
shutil.rmtree(HERE/"results", ignore_errors=True)
print("\nDONE. The base arms are regenerated; re-run FINAL_results.ipynb to rebuild all results.")
