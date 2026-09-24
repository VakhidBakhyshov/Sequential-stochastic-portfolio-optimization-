# Implementation notes

## 1. Notebook-to-scripts integration

`FINAL_results.ipynb` re-ran `scripts.runs.run` for `main_dyn_off.yaml` and `main_dyn_strong.yaml`, then derived realized-vol, model-CVaR and combined overlays. The new script `scripts/runs/final_results_simple.py` gives the same workflow from console.

## 2. Smart signal extension

The original research design had a backward realized-volatility signal and a forward model-implied CVaR signal. The new `smart_signals.py` adds an intermediate layer:

- backward quality: efficiency ratio, ATR current/long-term regime, liquidity score, breakout strength, normalized spike, pullback ratio, net move;
- forward quality: scenario expected return, Kelly fraction, risk-reward ratio;
- regime quality: GARCH(1,1)-style volatility clustering forecast;
- portfolio exposure: multiplicative overlay from Kelly x risk-reward x volatility x composite quality.

The layer can either only audit signals or actively rescale target positions, controlled by the `smart_signals` config block.

## 3. Train/test performance split

`stats.py` now keeps old metrics but adds `calculate_train_test_metrics`, `split_pnl_train_test`, rolling Sharpe and rolling Sortino helpers. The Plotly dashboard writes `train_test_metrics.csv`.

## 4. Interactive dashboard

The dashboard is generated automatically after runs and can be rebuilt manually. It includes:

- strategy vs SPY/equal-weight benchmark performance;
- drawdowns;
- rolling Sharpe and Sortino;
- return histograms;
- turnover by rebalance;
- eligible universe vs selected non-zero ETF heatmap;
- latest selected ETF covariance/correlation heatmap with dropdown;
- filterable monthly tables by minimum weight;
- dynamic parameter and forecast-risk panels when those files exist.

## 5. How to disable new behavior

In config:

```yaml
smart_signals:
  enabled: false
```

Dashboard generation is post-processing only and does not affect returns.

## 6. Strengthened known-distribution sequential analysis (2026-09-09)

A new analytical layer now sits between the closed-form formulas and the existing known-DGP sequential simulation:

- `scripts/theory/sequential_diffusion_analysis.py` implements exact Gaussian CVaR for correlated GBM log returns, its weight gradient, the derivative of centered CVaR with respect to pairwise Brownian correlation, and an oracle Gaussian CVaR-constrained optimizer.
- The finite-scenario Rockafellar-Uryasev LP is compared directly with the exact-law oracle as scenario count grows.
- Plug-in drift/covariance/centered-CVaR errors are measured across estimation-window lengths under the known GBM law.
- A correlation comparative-static experiment verifies mathematically and numerically how dependence increases long-only portfolio risk.
- Mean-shift experiments verify that centering removes the return-location channel exactly, while uncentred CVaR shifts one-for-one.
- `scripts/runs/sequential_diffusion_analysis.py` exports reproducible CSV/JSON diagnostics and three publication-ready figures.

This block is analytical validation rather than an empirical performance claim.  It is designed to test the same sequential architecture under a law known to the experimenter before synthetic estimated-input experiments and the historical ETF backtest.


## 7. Theorem-to-code and distribution audit

Run the exact-law diffusion layer:

```bash
python -m scripts.runs.sequential_diffusion_analysis --output results/sequential_diffusion_analysis
```

Run the broader theorem/lemma validation layer:

```bash
python -m scripts.runs.theorem_empirical_bridge --output-dir results/theorem_empirical_bridge
```

The second command writes a machine-readable `theorem_coverage_register.csv`, theorem-specific CSV files and figures/heatmaps. The register separates (i) direct numerical identity/comparative-static checks, (ii) remarks covered by a parent theorem experiment, (iii) falsified hypotheses/counterexamples, and (iv) conjectures/open problems that must remain unclaimed.

A normal historical `scripts/runs/run.py` execution now emits the extra fields needed for the same diagnostics on ETF data and then calls `run_historical_theorem_audit`. Its output is placed under:

```text
results/<run>/theorem_empirical_audit/
```

The historical audit measures only what the completed run actually exported. It never substitutes manuscript numbers or synthetic values for missing real-data artifacts.

## 8. New audit fields written by the production run

`forecast_risk.csv` now includes the centered/raw model CVaR, scenario mean/volatility, reading and optimizer CVaR levels, scenario 5/50/95% return quantiles, scenario VaR/ES, the PIT position of the subsequently realised risky-sleeve return, VaR-violation indicator, and optimizer binding/slack/shadow-price fields. `risk_matrices.json` now adds `selected_dimension` and `estimation_observations`, which makes historical N/T and covariance-conditioning diagnostics reproducible.
