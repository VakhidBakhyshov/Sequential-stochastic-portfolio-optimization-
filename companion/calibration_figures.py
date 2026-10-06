"""Calibration tables and figures of Section 7.12 from the saved trial table, without rerunning the search.

    python calibration_figures.py [--trials extensions/calibration/results/.../trial_metrics.csv]
                                  [--config scripts/configs/optuna_calibration.yaml] [--out results/calibration_reanalysis]

Writes pareto_front.csv, policy_comparison.csv, parameter_range_summary.csv, parameter_effect_heatmap.png (Figure 18),
metric_distribution_shift.png (Figure 19) and return_risk_frontier.png (Figure 17). Run from the companion folder.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd, yaml
from scripts.optimization.analysis import (add_policy_scores, choose_policy_trials, extract_pareto_front,
                                           build_parameter_range_summary, policy_comparison_table)
from scripts.optimization.plots import (save_parameter_effect_heatmap, save_metric_distribution_shift,
                                        save_return_risk_frontier)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", default="../extensions/calibration/results/advanced_smart_bayesian_risk_return/trial_metrics.csv")
    ap.add_argument("--config", default="scripts/configs/optuna_calibration.yaml")
    ap.add_argument("--out", default="results/calibration_reanalysis")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8")) or {}
    selection = dict(cfg.get("selection", {}) or {})
    trials = pd.read_csv(a.trials)
    if "status" in trials.columns:
        trials = trials[trials["status"].astype(str).str.upper().eq("COMPLETE")].copy()
    scored = add_policy_scores(trials, **{k: v for k, v in selection.items() if k in add_policy_scores.__code__.co_varnames})
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    selected = choose_policy_trials(scored, return_floor_quantile=float(selection.get("return_floor_quantile", 0.35)),
                                    minimum_annual_return=selection.get("minimum_annual_return"), minimum_dsr=selection.get("minimum_dsr"))
    policy_comparison_table(scored, selected).to_csv(out / "policy_comparison.csv", index=False)
    pareto = extract_pareto_front(scored); pareto.to_csv(out / "pareto_front.csv", index=False)
    build_parameter_range_summary(scored, max_bins=int(selection.get("parameter_bins", 6)),
                                  top_fraction=float(selection.get("top_fraction", 0.20))).to_csv(out / "parameter_range_summary.csv", index=False)
    save_parameter_effect_heatmap(scored, out / "parameter_effect_heatmap.png")
    save_metric_distribution_shift(scored, out / "metric_distribution_shift.png", top_fraction=float(selection.get("top_fraction", 0.20)))
    save_return_risk_frontier(scored, out / "return_risk_frontier.png")
    summary = {"n_completed": int(len(scored)), "n_pareto": int(len(pareto)),
               "max_return_trial": int(selected["max_return"].get("trial_number", selected["max_return"].name)),
               "risk_control_trial": int(selected["risk_control"].get("trial_number", selected["risk_control"].name))}
    (out / "calibration_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary); print("saved", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
