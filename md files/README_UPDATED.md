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

## 2026-09 analytical / publication-freeze extension

A new **known-distribution layer** now sits before the historical ETF experiment. It is deliberately
separate from the empirical backtest and must never be used as evidence that real ETF prices are GBM.
Its purpose is theorem verification under a data-generating process whose law is known exactly.

```bash
python -m scripts.runs.analytical_benchmark --output results/analytical_benchmark
```

The new package `scripts/theory/` provides:

- correlated multivariate GBM with an explicit Brownian correlation matrix;
- exact log-return moments for each asset and for a continuously rebalanced constant-mix portfolio;
- closed-form Gaussian VaR/CVaR for arithmetic/log-return losses;
- closed-form VaR/CVaR for the simple-return loss induced by lognormal wealth;
- a transparent return-maximizing CVaR LP with an L1 turnover penalty;
- deterministic top-K margin certificates and Gaussian rank-reversal probabilities;
- exact EMA impulse-decay and partial-execution turnover/contraction identities;
- a complete synthetic sequential policy with an endogenous feasible set, estimated scenario law,
  state-dependent CVaR/turnover parameters, fast and slow risk channels, a many-signal geometric
  aggregate, partial execution, and executed state carried forward.

The default deterministic run writes `results/analytical_benchmark/`, including theorem-verification
CSVs and four figures. Every artifact is labelled synthetic/known-DGP.

### Publication-freeze safeguards

`advanced_smart_bayesian.yaml` and `strategy_smart.yaml` now enable `strict_nested_engine: true`.
A publication run therefore aborts if the nested selector fails instead of mixing some months from the
nested engine with the legacy fallback. Research validation also uses `dsr_trial_count_mode: raw`, so the
primary DSR denominator is the full counted recipe ledger rather than the participation-ratio heuristic.
The primary percentile filter has an explicit orientation test (`tests/test_publication_freeze.py`).

The exact return/CVaR LP now records the CVaR constraint slack, binding flag, SciPy dual marginal, and
the corresponding shadow price for the original maximization problem. When the winning optimizer recipe
is refit on the full pre-rebalance history, these diagnostics are propagated into the monthly selection
audit so Proposition 2 / Proposition 9 can be checked date by date on the frozen historical re-run.
