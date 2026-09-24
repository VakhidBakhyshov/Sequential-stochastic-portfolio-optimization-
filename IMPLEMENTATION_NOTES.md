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
