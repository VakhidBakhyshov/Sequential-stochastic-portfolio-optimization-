"""Final analysis of the shared calibration study after all workers have finished.

Loads the sqlite study, rebuilds the trial table and writes trial_metrics.csv, pareto_front.csv, policy_comparison.csv,
parameter_range_summary.csv, parameter_importance.csv, best_configs/ and the three figures of Section 7.12 into the
study folder. Then prints the numbers the manuscript quotes.

    python finalize_calibration.py [--config scripts/configs/optuna_calibration.yaml]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import optuna, pandas as pd, yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from scripts.optimization.optuna_calibration import _study_dataframe, analyze_trials  # noqa: E402


def pct(x):
    return f"{100 * float(x):.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="scripts/configs/optuna_calibration.yaml")
    a = ap.parse_args()
    cfg = yaml.safe_load((ROOT / a.config).read_text(encoding="utf-8"))
    name = cfg["study_name"]
    out = ROOT / cfg["output_folder"]
    study = optuna.load_study(study_name=name, storage=f"sqlite:///{(out / 'study.sqlite3').resolve()}")
    trials = _study_dataframe(study)
    summary = analyze_trials(trials, out, study=study, selection_config=cfg.get("selection", {}))
    print(json.dumps({k: v for k, v in summary.items() if k != "pair_heatmaps"}, indent=2))

    m = pd.read_csv(out / "trial_metrics.csv")
    done = m[m["status"].astype(str).str.upper().eq("COMPLETE")]
    pareto = pd.read_csv(out / "pareto_front.csv")
    pol = pd.read_csv(out / "policy_comparison.csv").set_index("policy")
    imp = pd.read_csv(out / "parameter_importance.csv")
    print(f"\ntrials {len(m)} (completed {len(done)}, pareto {len(pareto)}), statuses: {m['status'].value_counts().to_dict()}")
    print("runtime per trial (min): mean %.1f, max %.1f" % (done["runtime_seconds"].mean() / 60, done["runtime_seconds"].max() / 60))
    print("\nAll completed trials: return %s to %s, vol %s to %s, Sharpe %.2f to %.2f, maxDD %s to %s" % (
        pct(done["annualized_return"].min()), pct(done["annualized_return"].max()),
        pct(done["annualized_volatility"].min()), pct(done["annualized_volatility"].max()),
        done["sharpe_ratio"].min(), done["sharpe_ratio"].max(),
        pct(done["max_drawdown"].min()), pct(done["max_drawdown"].max())))
    print("Pareto set: return %s-%s, Sharpe %.2f-%.2f, maxDD %s to %s" % (
        pct(pareto["annualized_return"].min()), pct(pareto["annualized_return"].max()),
        pareto["sharpe_ratio"].min(), pareto["sharpe_ratio"].max(),
        pct(pareto["max_drawdown"].min()), pct(pareto["max_drawdown"].max())))
    print("\nParameter importance (top two per target, and the max of the rest):")
    for tgt, g in imp.groupby("target", sort=False):
        g = g.sort_values("importance", ascending=False)
        rest = g.iloc[2:]["importance"].max() if len(g) > 2 else float("nan")
        print(f"  {tgt:22s} " + ", ".join(f"{r.parameter}={r.importance:.2f}" for r in g.head(2).itertuples()) + f"; rest<= {rest:.2f}")
    print("\nSelected policies:")
    for p in ["max_return", "risk_control"]:
        r = pol.loc[p]
        params = json.loads(r["parameters"])
        print(f"  {p}: trial {int(r['trial_number'])}: return {pct(r['annualized_return'])}, vol {pct(r['annualized_volatility'])}, "
              f"Sharpe {r['sharpe_ratio']:.3f}, maxDD {pct(r['max_drawdown'])}, PSR {r['psr']:.3f}, DSR {r['dsr']:.3f}")
        print("     " + ", ".join(f"{k}={v}" for k, v in sorted(params.items())))


if __name__ == "__main__":
    main()
