import sys, runpy, importlib
sys.stdout.reconfigure(encoding="utf-8")
import journal_style
sys.modules["report_style"] = journal_style   # every figure script now uses journal style
for script in ["fig_universe_and_decorr.py", "fig_pipeline.py", "paper_measurements.py",
               "build_results.py", "build_drawdown.py", "extension_comparison.py"]:
    print(f"=== {script} ===", flush=True)
    try:
        runpy.run_path(script, run_name="__main__")
    except SystemExit:
        pass
print("ALL FIGURES REGENERATED (journal style)")
