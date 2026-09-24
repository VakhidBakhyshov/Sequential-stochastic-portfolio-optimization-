# Signal Research Extension

This module implements the signal-dependence and multi-signal research requested by the HDRC methodology documents.
It is designed to run **after** the main monthly pipeline has produced its usual result folder. It does not require
future data to form the signals; future realized returns are used only as evaluation targets.

## Run

```bash
python3 -m scripts.validation.signal_research \
  --results-dir results/<your_run_folder> \
  --output-dir results/<your_run_folder>/signal_research
```

The loader looks for the standard artifacts `smart_rebalance_audit.csv`, `forecast_risk.csv`, `selection_audit.csv`,
and `pnl.csv`. It parses portfolio timers, model-risk diagnostics, dynamic state variables, and the per-ETF
`smart_signal_table` when available.

## Main outputs

- `signal_pearson.csv`, `signal_spearman.csv`, `signal_kendall.csv`, `signal_mutual_information.csv`
- `rolling_pairwise_correlations.csv`, `lead_lag_correlations.csv`, `best_lead_lag_matrix.csv`
- `defensive_event_jaccard.csv`
- `stationarity_tests.csv`, `engle_granger_cointegration.csv`, `cointegration_pvalue_matrix.csv`, `johansen_cointegration.json`
- `distribution_fits.csv`
- `signal_group_relationships.csv`
- `incremental_signal_order_future_risk.csv`, `incremental_signal_order_future_return.csv`
- `timer_subset_strategy_metrics.csv`
- publication-oriented heatmaps and response-surface PNG files
- `signal_research_summary.json`

## Statistical interpretation

Cointegration is deliberately **not** forced on clipped exposure timers. Each unbounded precursor is first tested
for integration order; Engle-Granger/Johansen tests are run only for series classified as plausibly I(1). For bounded
or stationary signals, correlation/rank correlation, mutual information, lead-lag structure, and defensive-event
overlap are the primary dependence diagnostics.

The greedy signal-addition tables use chronological expanding-block out-of-sample Ridge prediction and also report
a descriptive Frisch-Waugh-Lovell incremental-MSE quantity. They are a research-screening layer, not a confirmatory
winner selector. Any subset inspected and promoted after seeing these results belongs in the trial ledger used by
DSR/FDR/SPA/MCS-style inference.

`timer_subset_strategy_metrics.csv` reports raw and matched-average-exposure comparisons for geometric-mean and
product timer combinations. Matching exposure is required to separate timing quality from the mechanical effect of
holding more cash.

## False-strategy heatmap

The repository now distinguishes two versions:

- **standardized null**: mean Sharpe = 0, standard deviation of trial Sharpe = 1; useful for teaching/shape checks;
- **calibrated null**: mean and standard deviation estimated from the research trial ledger.

"Uninformed strategies" means no-skill/null strategies. It does **not** mean that Sharpe ratios are uniformly
distributed. Under the current False Strategy diagnostic the null trial Sharpe estimates are Gaussian, and the
maximum density is the Gaussian order-statistic density.

## Evidence boundary

The uploaded archive does not contain the raw ETF data panels required to reproduce the historical backtest. Therefore,
this delivery validates the code with unit tests and synthetic smoke artifacts only. Run the command above on the
frozen point-in-time empirical result directory before using any new table or figure in a publication.
