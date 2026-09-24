# CDaR notebooks

Generated as CDaR-specific counterparts to the uploaded CVaR notebooks.

## 1. `CDAR_FINAL_results.ipynb`
For `bayessian_cdar.yaml` and `bayessian_cdar_two_signal.yaml`. Includes performance metrics, cumulative wealth, drawdowns, rolling risk, monthly heatmaps, predictive raw/centred CDaR, forward/backward timers, exposure, composition, concentration, turnover, execution alpha, dynamic controller state, model diagnostics, nested selection and signal-research previews. It also optionally adds the paired CVaR arm.

## 2. `CDAR_optuna_parameter_calibration_analysis.ipynb`
For `optuna_calibration_cdar.yaml`. Uses the CDaR-specific selection weights from that YAML, plots return-CDaR frontiers, parameter effects, alpha x budget interaction surfaces, Pareto configurations, trial progression and can write frozen selected YAMLs.

## 3. `CDAR_vs_CVAR_paired_analysis.ipynb`
For `bayessian_cvar_paired.yaml` versus `bayessian_cdar.yaml`. Includes matched metrics, cumulative/drawdown plots, monthly treatment effects, a moving-block paired bootstrap, forward-risk comparison, L1 weight distance, active-set Jaccard and turnover.

All notebooks default to analysis-only mode so they do not accidentally launch multi-hour runs.
