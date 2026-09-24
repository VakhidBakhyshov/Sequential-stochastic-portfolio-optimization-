# Portfolio optimization pipeline - updated implementation

This package starts from `scripts-3.zip`, overlays the implementation from `FINAL - копия.zip`, and adds a console-first research layer so the work in `FINAL_results.ipynb` can be run from `.py` files.

## Main entry points

```bash
# Bayesian-CVaR optimizer with nested validation/test, smart signals, and Plotly dashboard
python -m scripts.runs.run -c advanced_smart_bayesian.yaml

# Strategy-only run with smart signals
python -m scripts.runs.strategy_run -c strategy_smart.yaml

# Simple console replacement for FINAL_results.ipynb
python -m scripts.runs.final_results_simple --configs main_dyn_off.yaml main_dyn_strong.yaml

# Build/rebuild dashboard from any finished result folder
python -m scripts.results.interactive_report --result-folder results/advanced_smart_bayesian
```

## Added modules

- `scripts/risk_controls/smart_signals.py` - Kelly, risk-reward ratio, volatility clustering/GARCH-style forecast, efficiency ratio, ATR regime, liquidity score, breakout strength, normalized spike, pullback ratio, net move and a composite signal quality score.
- `scripts/results/interactive_report.py` - one-file Plotly dashboard with performance, drawdown, rolling Sharpe/Sortino, histograms, turnover, eligible vs selected ETFs heatmaps, covariance/correlation toggle, metrics and monthly filterable tables.
- `scripts/runs/common_postprocess.py` - shared saving of train/test selection audit, smart signal audit and dashboard generation.
- `scripts/runs/final_results_simple.py` - notebook workflow as a console script.
- `scripts/configs/advanced_smart_bayesian.yaml` and `scripts/configs/strategy_smart.yaml` - ready-to-run configs.

## Integrated from FINAL-copy

The root scripts `cvar_target_overlay.py`, `fast_voltarget.py`, `report_style.py`, and `regenerate.py` were preserved, and the corresponding scripts from `model_code/scripts` were overlaid into `scripts/`. This keeps the posterior-predictive scenario fan, model-implied CVaR forecast audit, dynamic parameter history, universe matrix helper and overlay code.

## Outputs after a run

Each result folder under `results/<output_folder>/` can include:

- `pnl.csv`, `real.csv`, `preds.csv`, `model.csv`, `weights.xlsx`
- `dynamic_parameter_history.csv`
- `forecast_risk.csv`
- `selection_audit.csv`
- `smart_rebalance_audit.csv`
- `train_test_metrics.csv`
- `interactive_report.html`

## Important note

The full historical ETF datasets required by the original backtest are not in this chat sandbox, so this package has been syntax-checked and dashboard-smoke-tested, but the full strategy backtest was not executed here.
