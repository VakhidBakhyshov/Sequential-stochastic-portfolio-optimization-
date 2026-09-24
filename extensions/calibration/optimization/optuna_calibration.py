"""Optuna calibration runner for the portfolio optimization pipeline.

This module executes the existing monthly walk-forward pipeline once per trial,
then analyses the same trial set under two explicit policies:

1. ``max_return``: choose the configuration with the highest annualized return.
2. ``risk_control``: preserve a return floor while maximizing robust risk-adjusted
   quality (Sharpe, Sortino, PSR, DSR) and reducing volatility, drawdown and CVaR.

Usage from the repository root::

    python -m scripts.optimization.optuna_calibration \
        --config scripts/configs/optuna_calibration.yaml

Expensive completed studies can be reanalysed without rerunning the pipeline::

    python -m scripts.optimization.optuna_calibration \
        --analyze-trials results/calibration/my_study/trial_metrics.csv \
        --output results/calibration/my_study/reanalysis
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import yaml
from optuna.importance import get_param_importances

from scripts.optimization.analysis import (
    add_policy_scores,
    build_parameter_range_summary,
    choose_policy_trials,
    extract_pareto_front,
    policy_comparison_table,
)
from scripts.optimization.plots import (
    save_metric_distribution_shift,
    save_numeric_parameter_pair_heatmaps,
    save_parameter_effect_heatmap,
    save_return_risk_frontier,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / "scripts" / "configs"

METRIC_ALIASES = {
    "Annualized Return": "annualized_return",
    "Annualized Volatility": "annualized_volatility",
    "Sharpe Ratio": "sharpe_ratio",
    "Sortino Ratio": "sortino_ratio",
    "Max Drawdown": "max_drawdown",
    "CVaR (95%)": "cvar_95",
    "CDaR (95%)": "cdar_95",
    "Downside Deviation": "downside_deviation",
    "PSR": "psr",
    "DSR": "dsr",
    "Cumulative Return": "cumulative_return",
    "Win Rate": "win_rate",
}


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected a YAML mapping in {path}")
    return data


def _write_yaml(path: str | Path, data: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True)
    return output


def set_dotted_value(config: Any, dotted_path: str, value: Any) -> None:
    """Set values in mixed dict/list YAML structures using ``a.0.b`` paths."""
    tokens = dotted_path.split(".")
    current = config
    for token in tokens[:-1]:
        if isinstance(current, list):
            current = current[int(token)]
        elif isinstance(current, dict):
            if token not in current:
                current[token] = {}
            current = current[token]
        else:
            raise TypeError(f"Cannot descend through {type(current).__name__} at {token!r}")
    last = tokens[-1]
    if isinstance(current, list):
        if last.isdigit():
            current[int(last)] = value
        else:
            for item in current:
                if isinstance(item, dict) and last in item:
                    item[last] = value
                    break
            else:
                current.append({last: value})
    elif isinstance(current, dict):
        current[last] = value
    else:
        raise TypeError(f"Cannot set {dotted_path!r} on {type(current).__name__}")


def suggest_parameter(trial: optuna.Trial, name: str, spec: dict[str, Any]) -> Any:
    kind = str(spec.get("type", "float")).lower()
    if kind == "float":
        return trial.suggest_float(
            name,
            float(spec["low"]),
            float(spec["high"]),
            step=float(spec["step"]) if spec.get("step") is not None else None,
            log=bool(spec.get("log", False)),
        )
    if kind == "int":
        return trial.suggest_int(
            name,
            int(spec["low"]),
            int(spec["high"]),
            step=int(spec.get("step", 1)),
            log=bool(spec.get("log", False)),
        )
    if kind in {"categorical", "choice"}:
        return trial.suggest_categorical(name, list(spec["choices"]))
    raise ValueError(f"Unsupported search-space type {kind!r} for {name!r}")




def _tail_risk_objective_key(config: dict[str, Any]) -> str:
    """Return the calibrated tail-risk metric, defaulting to historical CVaR behavior."""
    raw = str(config.get("tail_risk_objective", "cvar")).strip().lower()
    aliases = {
        "cvar": "abs_cvar_95", "cvar_95": "abs_cvar_95", "abs_cvar_95": "abs_cvar_95",
        "cdar": "abs_cdar_95", "cdar_95": "abs_cdar_95", "abs_cdar_95": "abs_cdar_95",
    }
    if raw not in aliases:
        raise ValueError(f"Unsupported tail_risk_objective={raw!r}; use 'cvar' or 'cdar'.")
    return aliases[raw]


def sync_nested_optimizer_grid(trial_config: dict[str, Any], changed_values: dict[str, Any]) -> None:
    """Keep outer calibration values active inside the nested selector.

    With ``use_backtest_engine: true`` the monthly selector expands
    ``optimizer_param_grid`` over ``optimizer``. If the grid contains a key also
    tuned by Optuna, an old fixed grid value would otherwise silently override
    the trial value. We pin those duplicated grid keys to the current trial.
    """
    if not bool(trial_config.get("use_backtest_engine", False)):
        return
    grid = trial_config.get("optimizer_param_grid")
    if not isinstance(grid, dict):
        return
    for dotted_name, value in changed_values.items():
        tokens = dotted_name.split(".")
        if len(tokens) >= 3 and tokens[0] == "optimizer" and tokens[1].isdigit():
            key = ".".join(tokens[2:])
            if key in grid:
                grid[key] = [value]

def _read_pipeline_metrics(result_folder: Path, preferred_sample: str = "test") -> dict[str, float]:
    metrics_path = result_folder / "train_test_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Pipeline did not create {metrics_path}")
    table = pd.read_csv(metrics_path)
    if "sample" in table.columns:
        selected = table[table["sample"].astype(str).str.lower() == preferred_sample.lower()]
        if selected.empty:
            selected = table[table["sample"].astype(str).str.lower() == "overall"]
        row = selected.iloc[0] if not selected.empty else table.iloc[0]
    else:
        row = table.iloc[0]
    metrics: dict[str, float] = {}
    for raw, canonical in METRIC_ALIASES.items():
        if raw in row.index:
            metrics[canonical] = float(pd.to_numeric(pd.Series([row[raw]]), errors="coerce").iloc[0])
    if "turnover" not in metrics:
        turnover_files = [result_folder / "dynamic_parameter_history.csv", result_folder / "selection_audit.csv"]
        for path in turnover_files:
            if not path.exists():
                continue
            frame = pd.read_csv(path)
            candidates = [c for c in frame if "turnover" in c.lower()]
            if candidates:
                metrics["turnover"] = float(pd.to_numeric(frame[candidates[0]], errors="coerce").mean())
                break
    metrics["abs_max_drawdown"] = abs(metrics.get("max_drawdown", np.nan))
    metrics["abs_cvar_95"] = abs(metrics.get("cvar_95", np.nan))
    metrics["abs_cdar_95"] = abs(metrics.get("cdar_95", np.nan))
    required = ["annualized_return", "annualized_volatility", "abs_max_drawdown"]
    missing = [m for m in required if not np.isfinite(metrics.get(m, np.nan))]
    if missing:
        raise ValueError(f"Missing finite objective metrics in {metrics_path}: {missing}")
    return metrics


def _trial_record_from_frozen(trial: optuna.trial.FrozenTrial) -> dict[str, Any]:
    record: dict[str, Any] = {
        "trial_number": int(trial.number),
        "status": trial.state.name,
        "datetime_start": trial.datetime_start,
        "datetime_complete": trial.datetime_complete,
        "duration_seconds": trial.duration.total_seconds() if trial.duration else np.nan,
    }
    record.update({f"param_{k}": v for k, v in trial.params.items()})
    record.update(trial.user_attrs)
    if trial.values is not None:
        for i, value in enumerate(trial.values):
            record[f"objective_{i}"] = value
    return record


def _study_dataframe(study: optuna.Study) -> pd.DataFrame:
    return pd.DataFrame([_trial_record_from_frozen(t) for t in study.trials])


def _save_parameter_importance(study: optuna.Study, output: Path) -> pd.DataFrame:
    targets = ["annualized_return", "annualized_volatility", "abs_max_drawdown", "abs_cvar_95", "abs_cdar_95", "primary_tail_risk", "dsr"]
    rows: list[dict[str, Any]] = []
    for target_name in targets:
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and np.isfinite(t.user_attrs.get(target_name, np.nan))]
        if len(completed) < 3:
            continue
        try:
            importance = get_param_importances(study, target=lambda t, name=target_name: float(t.user_attrs[name]))
        except Exception:
            continue
        for parameter, value in importance.items():
            rows.append({"target": target_name, "parameter": parameter, "importance": float(value)})
    result = pd.DataFrame(rows)
    if not result.empty:
        result.to_csv(output / "parameter_importance.csv", index=False)
    return result


def _copy_selected_config(row: pd.Series, policy: str, output: Path) -> Path:
    source = Path(str(row["generated_config_path"]))
    if not source.is_absolute():
        source = PROJECT_ROOT / source
    config = _read_yaml(source)
    config["calibration_metadata"] = {
        "policy": policy,
        "selected_trial_number": int(row.get("trial_number", row.name)),
        "annualized_return": float(row.get("annualized_return", np.nan)),
        "annualized_volatility": float(row.get("annualized_volatility", np.nan)),
        "sharpe_ratio": float(row.get("sharpe_ratio", np.nan)),
        "sortino_ratio": float(row.get("sortino_ratio", np.nan)),
        "max_drawdown": float(row.get("max_drawdown", np.nan)),
        "cvar_95": float(row.get("cvar_95", np.nan)),
        "psr": float(row.get("psr", np.nan)),
        "dsr": float(row.get("dsr", np.nan)),
    }
    return _write_yaml(output / "best_configs" / f"{policy}.yaml", config)


def analyze_trials(
    trials: pd.DataFrame,
    output_directory: str | Path,
    *,
    study: optuna.Study | None = None,
    selection_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    selection = selection_config or {}
    scored = add_policy_scores(
        trials,
        return_weights=selection.get("return_weights"),
        risk_weights=selection.get("risk_weights"),
    )
    scored.to_csv(output / "trial_metrics.csv", index=False)
    completed = scored[scored.get("status", "COMPLETE").astype(str).str.upper().eq("COMPLETE")].copy() if "status" in scored else scored.copy()
    if completed.empty:
        raise ValueError("The study has no completed trials to analyse.")

    selected = choose_policy_trials(
        completed,
        return_floor_quantile=float(selection.get("return_floor_quantile", 0.35)),
        minimum_annual_return=selection.get("minimum_annual_return"),
        minimum_dsr=selection.get("minimum_dsr"),
    )
    comparison = policy_comparison_table(completed, selected)
    comparison.to_csv(output / "policy_comparison.csv", index=False)
    pareto = extract_pareto_front(completed)
    pareto.to_csv(output / "pareto_front.csv", index=False)
    ranges = build_parameter_range_summary(
        completed,
        max_bins=int(selection.get("parameter_bins", 6)),
        top_fraction=float(selection.get("top_fraction", 0.20)),
    )
    ranges.to_csv(output / "parameter_range_summary.csv", index=False)

    if "generated_config_path" in completed.columns:
        for policy, row in selected.items():
            _copy_selected_config(row, policy, output)

    save_parameter_effect_heatmap(completed, output / "parameter_effect_heatmap.png")
    save_metric_distribution_shift(completed, output / "metric_distribution_shift.png", top_fraction=float(selection.get("top_fraction", 0.20)))
    save_return_risk_frontier(completed, output / "return_risk_frontier.png")
    pair_paths = save_numeric_parameter_pair_heatmaps(completed, output / "parameter_pair_heatmaps")

    importance = _save_parameter_importance(study, output) if study is not None else pd.DataFrame()
    summary = {
        "n_trials": int(len(scored)),
        "n_completed": int(len(completed)),
        "n_pareto": int(len(pareto)),
        "max_return_trial": int(selected["max_return"].get("trial_number", selected["max_return"].name)),
        "risk_control_trial": int(selected["risk_control"].get("trial_number", selected["risk_control"].name)),
        "pair_heatmaps": [str(p) for p in pair_paths],
        "parameter_importance_rows": int(len(importance)),
    }
    (output / "calibration_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def run_study(config_path: str | Path, n_trials_override: int | None = None) -> Path:
    cfg = _read_yaml(config_path)
    base_config_path = Path(cfg["base_config"])
    if not base_config_path.is_absolute():
        base_config_path = PROJECT_ROOT / base_config_path
    base_config = _read_yaml(base_config_path)
    study_name = str(cfg.get("study_name", f"portfolio_calibration_{int(time.time())}"))
    output = PROJECT_ROOT / str(cfg.get("output_folder", f"results/calibration/{study_name}"))
    output.mkdir(parents=True, exist_ok=True)
    generated_dir = CONFIG_ROOT / "_optuna_generated" / study_name
    generated_dir.mkdir(parents=True, exist_ok=True)

    sampler_cfg = cfg.get("sampler", {}) or {}
    sampler = optuna.samplers.NSGAIISampler(
        seed=int(sampler_cfg.get("seed", 42)),
        population_size=int(sampler_cfg.get("population_size", 20)),
    )
    storage = f"sqlite:///{(output / 'study.sqlite3').resolve()}"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        directions=["maximize", "minimize", "minimize", "minimize"],
        sampler=sampler,
    )
    search_space = cfg.get("search_space", {}) or {}
    if not search_space:
        raise ValueError("optuna_calibration.yaml has an empty search_space")
    entrypoint = str(cfg.get("entrypoint", "scripts.runs.run"))
    preferred_sample = str(cfg.get("evaluation_sample", "test"))
    tail_risk_key = _tail_risk_objective_key(cfg)
    timeout = cfg.get("trial_timeout_seconds")
    python_executable = str(cfg.get("python_executable", sys.executable))

    def objective(trial: optuna.Trial) -> tuple[float, float, float, float]:
        trial_config = copy.deepcopy(base_config)
        changed_values: dict[str, Any] = {}
        for dotted_name, spec in search_space.items():
            value = suggest_parameter(trial, dotted_name, dict(spec))
            set_dotted_value(trial_config, dotted_name, value)
            changed_values[dotted_name] = value
        if bool(cfg.get("sync_nested_optimizer_grid", True)):
            sync_nested_optimizer_grid(trial_config, changed_values)
        result_relative = Path(str(cfg.get("trial_results_root", "results/calibration_runs"))) / study_name / f"trial_{trial.number:05d}"
        trial_config["output_folder"] = str(result_relative.relative_to("results") if result_relative.parts and result_relative.parts[0] == "results" else result_relative)
        trial_config.setdefault("research_validation", {})["enabled"] = True
        generated_path = generated_dir / f"trial_{trial.number:05d}.yaml"
        _write_yaml(generated_path, trial_config)
        relative_config = generated_path.relative_to(CONFIG_ROOT)
        result_folder = PROJECT_ROOT / "results" / str(trial_config["output_folder"])
        log_path = output / "logs" / f"trial_{trial.number:05d}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = [python_executable, "-m", entrypoint, "--config-path", str(relative_config)]
        start = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=float(timeout) if timeout else None,
                env={**os.environ, "PYTHONHASHSEED": str(cfg.get("seed", 42))},
                check=False,
            )
        elapsed = time.perf_counter() - start
        trial.set_user_attr("generated_config_path", str(generated_path.relative_to(PROJECT_ROOT)))
        trial.set_user_attr("result_folder", str(result_folder.relative_to(PROJECT_ROOT)))
        trial.set_user_attr("log_path", str(log_path.relative_to(PROJECT_ROOT)))
        trial.set_user_attr("runtime_seconds", float(elapsed))
        trial.set_user_attr("return_code", int(completed.returncode))
        if completed.returncode != 0:
            trial.set_user_attr("failure_reason", f"pipeline_return_code_{completed.returncode}")
            raise optuna.TrialPruned(f"Pipeline failed; inspect {log_path}")
        metrics = _read_pipeline_metrics(result_folder, preferred_sample=preferred_sample)
        if not np.isfinite(metrics.get(tail_risk_key, np.nan)):
            raise ValueError(
                f"Calibration requested {tail_risk_key}, but the pipeline did not export a finite value "
                f"in {result_folder / 'train_test_metrics.csv'}"
            )
        metrics["primary_tail_risk"] = float(metrics[tail_risk_key])
        for key, value in metrics.items():
            trial.set_user_attr(key, float(value) if np.isfinite(value) else np.nan)
        trial.set_user_attr("primary_tail_risk_metric", tail_risk_key)
        return (
            metrics["annualized_return"],
            metrics["annualized_volatility"],
            metrics["abs_max_drawdown"],
            metrics[tail_risk_key],
        )

    n_trials = int(n_trials_override if n_trials_override is not None else cfg.get("n_trials", 50))
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True, show_progress_bar=bool(cfg.get("show_progress_bar", True)))
    trials = _study_dataframe(study)
    analyze_trials(trials, output, study=study, selection_config=cfg.get("selection", {}))
    shutil.copy2(config_path, output / "optuna_calibration_used.yaml")
    shutil.copy2(base_config_path, output / "base_config_used.yaml")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna calibration of portfolio return and risk")
    parser.add_argument("--config", type=str, default="scripts/configs/optuna_calibration.yaml")
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--analyze-trials", type=str, default=None, help="Reanalyse an existing trial_metrics.csv")
    parser.add_argument("--output", type=str, default=None, help="Output directory for --analyze-trials")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.analyze_trials:
        source = Path(args.analyze_trials)
        output = Path(args.output) if args.output else source.parent / "reanalysis"
        summary = analyze_trials(pd.read_csv(source), output)
        print(json.dumps(summary, indent=2))
        return
    output = run_study(args.config, n_trials_override=args.n_trials)
    print(f"Calibration artifacts saved to {output}")


if __name__ == "__main__":
    main()
