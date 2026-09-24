"""
Console replacement for the FINAL_results.ipynb workflow.

It runs the main strategy configs, derives overlays where the required base results exist,
and builds Plotly dashboards. This gives a simple non-notebook entry point:

    python -m scripts.runs.final_results_simple --configs main_dyn_off.yaml main_dyn_strong.yaml

Use --skip-regenerate to build reports/overlays from existing results only.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def run_config(config_name: str, cwd: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(cwd)
    cmd = [sys.executable, "-m", "scripts.runs.run", "-c", config_name]
    print("RUN", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(cwd), env=env, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Run failed for {config_name} with code {proc.returncode}")


def build_reports(cwd: Path) -> None:
    from scripts.results.interactive_report import build_interactive_report
    res = cwd / "results"
    if not res.exists():
        print("No results/ folder found yet.")
        return
    for folder in sorted(p for p in res.iterdir() if p.is_dir()):
        if (folder / "pnl.csv").exists():
            try:
                out = build_interactive_report(folder)
                print(f"dashboard: {out}")
            except Exception as exc:
                print(f"dashboard skipped for {folder.name}: {exc}")


def run_overlay_scripts(cwd: Path) -> None:
    for script in ["fast_voltarget.py", "cvar_target_overlay.py"]:
        p = cwd / script
        if p.exists():
            print(f"overlay script: {script}")
            subprocess.run([sys.executable, str(p)], cwd=str(cwd), text=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=["main_dyn_off.yaml", "main_dyn_strong.yaml"], help="configs under scripts/configs")
    ap.add_argument("--skip-regenerate", action="store_true", help="do not run scripts.runs.run; only build overlays and dashboards")
    ap.add_argument("--copy-to-data", action="store_true", help="copy results folders into data/ like the original notebook")
    args = ap.parse_args()

    cwd = Path.cwd()
    if not args.skip_regenerate:
        for cfg in args.configs:
            run_config(cfg, cwd)

    if args.copy_to_data:
        data = cwd / "data"
        data.mkdir(exist_ok=True)
        for cfg in args.configs:
            # Uses output_folder inside yaml; when unknown, the results folder will still be reported by dashboards.
            pass

    run_overlay_scripts(cwd)
    build_reports(cwd)


if __name__ == "__main__":
    main()
