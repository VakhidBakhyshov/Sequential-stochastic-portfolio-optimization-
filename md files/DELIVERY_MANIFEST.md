# Portfolio Optimization Pipeline - Modification Patch

This archive contains only files that were added or changed. Paths are preserved relative to the root of the original `portfolio_optimization_pipeline_v2` framework.

## Integration

1. Back up the original framework.
2. Extract this archive into the framework root and allow matching files to be replaced.
3. Install the updated requirements.
4. Run the tests and then the desired pipeline/configuration.

```bash
python -m pip install -r requirements.txt
python -m compileall -q scripts tests
python -m pytest -q tests/test_research_validation.py tests/test_optuna_calibration.py
```

## Changed existing files

- `requirements.txt`
- `scripts/backtesting/walk_forward_engine.py`
- `scripts/configs/advanced_smart_bayesian.yaml`
- `scripts/configs/strategy_smart.yaml`
- `scripts/results/interactive_report.py`
- `scripts/results/stats.py`
- `scripts/runs/common_postprocess.py`
- `scripts/runs/run.py`
- `scripts/runs/strategy_run.py`
- `scripts/runs/synthetic_smoke.py`

## New files

- `scripts/configs/optuna_calibration.yaml`
- `scripts/validation/__init__.py`
- `scripts/validation/false_strategy.py`
- `scripts/validation/research_validation.py`
- `scripts/optimization/__init__.py`
- `scripts/optimization/analysis.py`
- `scripts/optimization/optuna_calibration.py`
- `scripts/optimization/plots.py`
- `notebooks/optuna_parameter_calibration_analysis.ipynb`
- `docs/RESEARCH_VALIDATION_AND_CALIBRATION.md`
- `tests/test_research_validation.py`
- `tests/test_optuna_calibration.py`

## Main commands

Run the model pipeline:

```bash
python -m scripts.runs.run --config-path advanced_smart_bayesian.yaml
python -m scripts.runs.strategy_run --config-path strategy_smart.yaml
```

Run multi-objective Optuna calibration:

```bash
python -m scripts.optimization.optuna_calibration \
  --config scripts/configs/optuna_calibration.yaml
```

Reanalyse a completed trial table without rerunning models:

```bash
python -m scripts.optimization.optuna_calibration \
  --analyze-trials results/calibration/<study>/trial_metrics.csv \
  --output results/calibration/<study>/reanalysis
```

The separate QA archive contains synthetic outputs used only to verify execution and visualization. They are not empirical strategy results.

## 2026-09-09 known-distribution extension

New files:
- `scripts/theory/sequential_diffusion_analysis.py`
- `scripts/runs/sequential_diffusion_analysis.py`
- `tests/test_sequential_diffusion_analysis.py`

Changed files:
- `IMPLEMENTATION_NOTES.md`
- `CODE_AUDIT_AND_CHANGELOG.md`

Run:
```bash
python -m scripts.runs.sequential_diffusion_analysis --output results/sequential_diffusion_analysis
```
