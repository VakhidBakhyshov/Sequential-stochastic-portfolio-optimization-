"""Regenerate the tables and figures of the paper from the run folders, in dependency order.

Usage (from the package root, after the arms have been run with run_arms.py):
    python reproduce.py [--only name1,name2] [--skip name1,...]
Each step writes its console output to logs/<name>.log; exit codes and timings go to logs/manifest.txt.
"""
from __future__ import annotations
import argparse, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
ENV = dict(os.environ, PYTHONIOENCODING="utf-8")

STEPS = [  # (name, argv); overlays before verification, verification before figures
    ("fast_voltarget",         ["fast_voltarget.py"]),
    ("cvar_target_overlay",    ["cvar_target_overlay.py"]),
    ("verify_all",             ["verify_all.py"]),
    ("benchmarks",             ["benchmarks.py"]),
    ("tail_bootstrap",         ["tail_bootstrap.py"]),
    ("vix_experiment",         ["vix_experiment.py"]),
    ("controls",               ["controls.py"]),
    ("paper_measurements",     ["paper_measurements.py"]),
    ("paper_subperiod",        ["paper_measurements.py", "--subperiod"]),
    ("universe_compare",       ["universe_compare.py"]),
    ("compare_fhs_arm",        ["compare_fhs_arm.py"]),
    ("fhs_experiment",         ["fhs_experiment.py"]),
    ("experiment5",            ["experiment5.py"]),
    ("block_bootstrap",        ["block_bootstrap_experiment.py"]),
    ("extension_comparison",   ["extension_comparison.py"]),
    ("build_results",          ["build_results.py"]),
    ("build_drawdown",         ["build_drawdown.py"]),
    ("plot_dynamic_params",    ["plot_dynamic_params.py"]),
    ("fig_universe_decorr",    ["fig_universe_and_decorr.py"]),
    ("make_figures",           ["make_figures.py"]),
    ("overlay_arms",           ["overlay_arms.py", "main_dyn_strong", "--out", "arms_main_dyn_strong.csv"]),
    ("overlay_arms_execbook",  ["overlay_arms.py", "main_dyn_strong", "--weights-col", "executed_weights", "--out", "arms_main_dyn_strong_execbook.csv"]),
    ("overlay_arms_uncentred", ["overlay_arms.py", "main_dyn_strong", "--cvar-col", "cvar_model_raw", "--out", "arms_main_dyn_strong_uncentred.csv"]),
    ("mech_summary",           ["mech_summary.py"]),
    ("ew_universe_ablation",   ["ew_universe_ablation.py"]),
    ("signal_validity",        ["signal_validity.py"]),
    ("subperiod_table",        ["subperiod_table.py"]),
    ("seed_table",             ["seed_table.py"]),
    ("grid_table",             ["grid_table.py"]),
    ("grid_composition",       ["grid_composition.py"]),
    ("extra_ratios",           ["extra_ratios.py"]),
    ("complementarity",        ["complementarity_diagnostics.py"]),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--skip", default="")
    a = ap.parse_args()
    only = {s for s in a.only.split(",") if s}
    skip = {s for s in a.skip.split(",") if s}
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    manifest = logs / "manifest.txt"
    for name, argv in STEPS:
        if (only and name not in only) or name in skip:
            continue
        t0 = time.time()
        with (logs / f"{name}.log").open("w", encoding="utf-8") as lf:
            rc = subprocess.call([PY] + argv, cwd=str(ROOT), stdout=lf, stderr=subprocess.STDOUT, env=ENV)
        line = f"{name:24s} exit={rc} {time.time() - t0:7.1f}s"
        print(line)
        with manifest.open("a", encoding="utf-8") as m:
            m.write(line + "\n")


if __name__ == "__main__":
    main()
