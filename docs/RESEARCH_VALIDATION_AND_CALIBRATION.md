# Research validation and Optuna calibration

## Pipeline integration

Both `scripts/runs/run.py` and `scripts/runs/strategy_run.py` now save every validation and internal-test candidate to `trial_audit.csv`. After the final out-of-sample month is processed, the run automatically executes:

1. PSR and DSR inference;
2. the False Strategy Theorem benchmark `E[max(SR)]`;
3. Bonferroni/Holm FWER and Benjamini–Hochberg/Benjamini–Yekutieli FDR corrections;
4. a nested validation/internal-test overfitting proxy;
5. a causal split audit to detect future-data use;
6. monthly and annual return visualizations and the false-strategy heatmap;
7. the interactive HTML report.

The generated publication artifacts are stored beside `pnl.csv`:

- `research_validation_summary.csv` and `.json`
- `multiple_testing_results.csv`
- `nested_pbo_proxy.csv`
- `causality_audit.csv`
- `false_strategy_surface.csv`
- `monthly_returns_matrix.csv`
- `annual_returns_comparison.csv`
- `monthly_returns_heatmap.png`
- `annual_returns_vs_spy.png`
- `false_strategy_heatmap.png`
- `interactive_report.html`

## Run the normal strategy

```bash
python -m scripts.runs.run --config-path advanced_smart_bayesian.yaml
```

or

```bash
python -m scripts.runs.strategy_run --config-path strategy_smart.yaml
```

## Run Optuna calibration

```bash
python -m scripts.optimization.optuna_calibration \
  --config scripts/configs/optuna_calibration.yaml
```

The study creates a separate output directory for every trial, a SQLite study, logs, `trial_metrics.csv`, a Pareto front, parameter-range summaries, parameter importance, pairwise heatmaps and two selected YAMLs:

- `best_configs/max_return.yaml`
- `best_configs/risk_control.yaml`

The risk-controller is subject to a return floor, which prevents the optimizer from selecting a trivial near-cash solution merely because it has low volatility.

## Reanalyse an existing study

```bash
python -m scripts.optimization.optuna_calibration \
  --analyze-trials results/calibration/advanced_smart_bayesian_risk_return/trial_metrics.csv \
  --output results/calibration/advanced_smart_bayesian_risk_return/reanalysis
```

Then open `notebooks/optuna_parameter_calibration_analysis.ipynb` for the final visual analysis.

## Important empirical rule

The calibration study must use only validation/internal-test periods for parameter decisions. The final untouched out-of-sample segment must not be used for selecting ranges, objectives or weights. DSR must count all attempted recipes, including failed or unattractive ones, rather than only the reported winners.
