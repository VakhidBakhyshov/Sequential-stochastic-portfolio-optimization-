"""Companion-engine chain behind Sections 7.9-7.12, Table 7, Figures 6-19 and L.6-L.8 of the paper.

    python reproduce_companion.py                 analysis only, on the shipped run folders (a few minutes)
    python reproduce_companion.py --run-arms      first rerun the four companion arms (about 3.5 min, 40 min and
                                                  2 x 2.5 h on a 16-core machine; they run in parallel)

Steps: (1) copy datasets/ into companion/datasets if missing (the companion engine reads data relative to its root);
(2) optionally run bayessian_cvar_paired, bayessian_cdar, bayessian_cdar_two_signal and advanced_smart_bayesian;
(3) signal research on the multi-signal arm (Figures 9-12); (4) matched CVaR-CDaR comparison (Table 7);
(5) companion figures (Figures 6-8, 14, L.6-L.8); (6) calibration tables and figures from the trial table of the shipped search
(Section 7.12, Figures 17-19); (7) the extension comparison against the main study (Figure 13).
Console output of every step goes to logs/companion/.
"""
from __future__ import annotations
import argparse, os, shutil, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COMP = ROOT / "companion"
PY = sys.executable
ENV = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONPATH=str(COMP))
ARMS = ["bayessian_cvar_paired", "bayessian_cdar", "bayessian_cdar_two_signal", "advanced_smart_bayesian"]
LOGS = ROOT / "logs" / "companion"


def step(name: str, argv: list[str], cwd: Path) -> int:
    LOGS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with (LOGS / f"{name}.log").open("w", encoding="utf-8") as lf:
        rc = subprocess.call(argv, cwd=str(cwd), stdout=lf, stderr=subprocess.STDOUT, env=ENV)
    print(f"{name:34s} exit={rc} {time.time() - t0:7.1f}s", flush=True)
    return rc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-arms", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if not (COMP / "datasets").exists():
        shutil.copytree(ROOT / "datasets", COMP / "datasets")
        print("copied datasets/ into companion/datasets")
    if a.run_arms:
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            list(ex.map(lambda c: step(f"run_{c}", [PY, "-m", "scripts.runs.run", "-c", f"{c}.yaml"], COMP), ARMS))
    res = COMP / "results"
    step("signal_research", [PY, "-m", "scripts.validation.signal_research", "--results-dir", str(res / "advanced_smart_bayesian"),
                             "--output-dir", str(res / "advanced_smart_bayesian" / "signal_research")], COMP)
    step("compare_cvar_cdar", [PY, "-m", "scripts.runs.compare_cvar_cdar", "--cvar-dir", str(res / "bayessian_cvar_paired_walk_forward"),
                               "--cdar-dir", str(res / "bayessian_cdar_walk_forward"), "--output-dir", str(res / "cvar_cdar_comparison")], COMP)
    step("companion_figures", [PY, "make_companion_figures.py", "--results", str(res), "--out", str(res / "companion_figures")], COMP)
    step("calibration_figures", [PY, "calibration_figures.py"], COMP)
    if (ROOT / "results" / "main_combined" / "pnl.csv").exists() and (ROOT / "results" / "main_dyn_off" / "real.csv").exists():
        step("extension_comparison", [PY, "extension_comparison.py"], ROOT)
    else:
        print("extension_comparison skipped: run the main engine first (run_arms.py, reproduce.py) to create results/main_dyn_off and results/main_combined")


if __name__ == "__main__":
    main()
