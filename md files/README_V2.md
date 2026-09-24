# ETF Risk-Signal Diversification Pipeline — Research Edition

This package is a console-first version of the ETF portfolio research framework. It preserves the original folder structure, adds a causal walk-forward selection engine, horizon-consistent posterior scenarios, exact return-maximization subject to a CVaR budget, smart position sizing, train/test statistics, actual risk-matrix export, and a self-contained Plotly audit dashboard.

## Important empirical warning

The package does **not** hard-code the Advanced Smart Bayesian strategy as the winner. A strategy is promoted only when it improves validation and internal-test results and then survives the untouched next-month evaluation. The sample dashboard under `reports/sample_dashboard/` is synthetic and exists only to test file schemas and reporting.

## Main commands

```bash
python -m pip install -r requirements.txt
python -m pytest -q

# Primary EMA-smoothed ETF universe
python -m scripts.dataloader.monthly_etf_filtration_2 \
  --prices datasets/csv/NewClosePrice.csv \
  --volumes datasets/csv/Volume.csv \
  --business-dates datasets/excel/business_dates.xlsx \
  --top-n 100 --benchmark SPY --spy-filter smart \
  --lookback-days 90 --score-ema-alpha 0.5

# Rolling-score robustness universe
python -m scripts.dataloader.monthly_etf_filtration \
  --top-n 100 --rolling-score-months 3

# Advanced Bayesian-CVaR policy
python -m scripts.runs.run -c advanced_smart_bayesian.yaml

# Strategy-only champion/challenger experiment
python -m scripts.runs.strategy_run -c strategy_smart.yaml

# Notebook replacement
python -m scripts.runs.final_results_simple --configs advanced_smart_bayesian.yaml

# Rebuild dashboard from existing results
python -m scripts.results.interactive_report \
  --result-folder results/advanced_smart_bayesian

# Synthetic reporting smoke test — not a performance result
python -m scripts.runs.synthetic_smoke
```

## Core changes

1. **Holding-period posterior scenarios.** Daily simulated returns are aggregated over the configured horizon, so the optimizer and forward CVaR timer use the same monthly unit as execution.
2. **Model-consistent CVaR policy.** `return_cvar_constraint` maximizes posterior expected holding-period return subject to a scenario CVaR budget and turnover cost.
3. **Cash overlay preserved.** Exposure reductions are no longer renormalized back to 100% risky assets.
4. **Backward + forward risk timers.** The portfolio controller combines realized/GARCH volatility with posterior-predictive CVaR, then adds a bounded conviction timer from Kelly, payoff asymmetry and signal quality.
5. **Stable model selection.** Candidate recipes are ranked using validation and internal-test performance plus a degradation penalty. The next month is not used for selection.
6. **Executed weights are state.** When partial rebalancing uses `alpha < 1`, next month starts from the actual executed weights, not the unreachable target.
7. **Actual model matrices.** Every rebalance can save the covariance and correlation matrices used by the model to `risk_matrices.json`.
8. **ETF inception protection.** Universe metrics count observations before filling and never backward-fill pre-listing history.
9. **Research statistics.** Overall/train/test metrics, rolling Sharpe and Sortino, win rate, average drawdown, expectancy, profit factor, PSR and DSR are exported.
10. **Interactive audit.** The dashboard shows performance versus SPY/equal-weight proxy, drawdown, risk, turnover, risky exposure, eligible/selected universes, actual model matrices, smart signals and monthly min/max weight-filterable tables.

## Result folder contract

A normal run writes:

- `pnl.csv`, `preds.csv`, `real.csv`, `model.csv`
- `weights.xlsx` with target and executed weights
- `selection_audit.csv`, `smart_rebalance_audit.csv`
- `forecast_risk.csv`, `dynamic_parameter_history.csv`
- `risk_matrices.json`, `run_metadata.json`
- `train_test_metrics.csv`, `rolling_metrics.csv`
- `interactive_report.html`

## Legacy runners

`combined_run.py`, `combined_strategy_run.py`, `new_run.py`, and `new_run2.py` are retained for reproducibility of older experiments. They are not the recommended path for journal results because they contain legacy in-sample alpha/model comparison logic. New journal tables should be generated through `run.py` or `strategy_run.py` with `use_backtest_engine: true`.
