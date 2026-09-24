"""Run the monthly policy for one or more configurations (from the package root).

    python run_arms.py                      runs the ten configurations reported in the paper
    python run_arms.py main_dyn_strong      runs one configuration
    python run_arms.py --grid               adds the scenario-count x history-rule x seed grid
Each run reads scripts/configs/<name>.yaml and writes results/<output_folder>/.
"""
from __future__ import annotations
import os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PAPER_ARMS = ["main_dyn_strong", "main_dyn_off", "main_dyn_strong_fhs", "main_strong_m1", "main_off_m1",
              "main_strong_m2", "main_off_m2", "mech_execstate_main_dyn_strong", "main_dyn_strong_fixeduniv",
              "main_dyn_strong_eta1"]
GRID = ["main_dyn_strong_seed7", "main_dyn_strong_seed2024", "main_dyn_strong_S1000"] + \
       [f"grid_S{s}_H{h}_s{seed}" for s in (1000, 5000) for h in ("none", 36, 48, 60) for seed in (42, 7, 2024)]


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    names = args or PAPER_ARMS
    if "--grid" in sys.argv:
        names = names + [g for g in GRID if (ROOT / "scripts" / "configs" / f"{g}.yaml").exists()]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    for name in names:
        cfg = ROOT / "scripts" / "configs" / f"{name}.yaml"
        if not cfg.exists():
            print(f"{name}: no such configuration"); continue
        t0 = time.time()
        rc = subprocess.call([sys.executable, str(ROOT / "scripts" / "runs" / "run.py"), "-c", f"{name}.yaml"], cwd=str(ROOT), env=env)
        print(f"{name}: exit={rc} {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
