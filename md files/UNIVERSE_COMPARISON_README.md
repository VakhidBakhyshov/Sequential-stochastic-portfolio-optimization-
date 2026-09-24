# ETF Universe Comparison — Step-by-Step

This addition tests whether the richer ETF universe is useful **before** the Bayesian model, CVaR optimizer, dynamic parameters, or exposure overlay are allowed to influence the result.

The key identification idea is simple: hold the portfolio rule fixed at **monthly equal weight** and change only the universe. If the advanced filter still improves risk-adjusted and downside outcomes, the evidence is attributable to the action-set decision rather than to the optimizer.

## 1. Required framework files

The comparison uses the same market inputs as `scripts/runs/run.py` plus `OpenPrice.csv`, which the legacy filter needs:

```text
datasets/csv/NewClosePrice.csv
datasets/csv/OpenPrice.csv
datasets/csv/Volume.csv
datasets/excel/business_dates.xlsx
```

If `business_dates.xlsx` is absent, the comparison infers the first observed trading day of each month from `NewClosePrice.csv`.

`NewClosePrice.csv` should be the same adjusted/total-return-compatible close series used for the production run. If it is not adjusted for distributions/splits, no universe comparison should be interpreted as final evidence.

## 2. Why four arms are reported

```text
basic_legacy
    Legacy new_filter_data.py logic, including the old sum-based history test.
    It is kept for provenance, not as the strongest scientific baseline.

basic_repaired
    Same legacy score, but with a causal requirement for 36 completed months
    of positive traded value. This is the PRIMARY baseline.

advanced_matched
    monthly_etf_filtration_2.py with the SAME top-N as basic, SPY allowed,
    and no hard SPY gate. This isolates the richer score/EMA design.
    This is the PRIMARY treatment.

advanced_native
    Richer filter using its requested native top-N, hard SPY rule and
    benchmark exclusion. This evaluates the full deployed design.
```

The paper-quality comparison is therefore:

```text
advanced_matched  vs  basic_repaired
```

This prevents an apparent improvement from being caused merely by selecting a different number of ETFs or by comparing against a deliberately weak legacy bug.

## 3. Run the data preparation already used by the framework

From the project root:

```bash
python scripts/dataloader/parse_close.py
```

This is mainly needed to create/update `datasets/excel/business_dates.xlsx` and the return files used by the rest of your framework. The equal-weight comparison itself calculates holding-period returns from `NewClosePrice.csv` so that the return convention is explicit.

## 4. Run the equal-weight universe ablation

```bash
python scripts/runs/compare_etf_universes.py \
  --basic-top-n 80 \
  --advanced-native-top-n 100 \
  --advanced-native-spy-filter smart \
  --advanced-lookback-days 90 \
  --advanced-score-ema-alpha 0.5 \
  --warmup-years 3 \
  --cost-bps 10 \
  --bootstrap-reps 5000 \
  --bootstrap-block-length 6 \
  --output-dir results/universe_comparison
```

The script follows `run.py`'s default three-year warmup. Both basic and advanced filter states are constructed on the pre-evaluation months, but performance starts only after the warmup.

## 5. Read the outputs in this order

### A. `summary_metrics.csv`

Compare the equal-weight portfolios on:

- annualized return;
- annualized volatility;
- Sharpe ratio;
- Sortino ratio;
- maximum drawdown;
- 95% CVaR;
- Calmar ratio;
- annualized one-way turnover;
- cumulative transaction costs.

For a risk-control paper, the most persuasive pattern is **not** simply a higher return. A stronger result is higher Sharpe/Sortino together with shallower maximum drawdown/CVaR after costs.

### B. `paired_block_bootstrap.csv`

The same monthly blocks are resampled for both universes. This preserves the common market path and some serial dependence. Focus on:

```text
advanced_matched_vs_basic_repaired
```

The column `probability_advanced_better` answers how often the advanced universe is better in the paired bootstrap for each metric. A value such as 0.95 is much more useful than reporting only a point estimate.

### C. `universe_quality_summary.csv`

This asks *what kind of ETFs* each filter selected, using the same 90-day diagnostic definitions for every arm:

- median dollar volume: higher is preferable;
- median volatility: lower is preferable for a defensive screen;
- median downside volatility: lower is preferable;
- median absolute drawdown: lower is preferable;
- median risk-adjusted return: higher is preferable;
- median Amihud illiquidity: lower is preferable.

This table explains the mechanism behind any portfolio-level result.

### D. `universe_stability.csv`

Use Jaccard overlap and replacement fraction to show whether EMA smoothing actually stabilizes membership. This is a separate object from portfolio turnover.

### E. `cross_universe_overlap.csv`

This shows how different the selected ETF sets actually are. If advanced and basic have Jaccard overlap near one, a large performance difference should be treated skeptically. If overlap is materially below one, there is a genuine action-set change to explain.

### F. `advanced_vs_basic_scorecard.csv`

This is a descriptive checklist, **not** a statistical test. It prevents cherry-picking one favorable metric.

## 6. Required plots

The script creates `results/universe_comparison/plots/`:

```text
01_cumulative_wealth.png
02_drawdowns.png
03_rolling_sharpe.png
04_rolling_volatility.png
05_portfolio_turnover.png
06_universe_jaccard.png
07_median_dollar_volume.png
08_median_volatility.png
09_median_downside_volatility.png
10_median_drawdown_abs.png
11_median_risk_adjusted_return.png
12_median_amihud_illiquidity.png
```

For the article, the most useful figures are usually cumulative wealth + underwater drawdown, rolling Sharpe, universe Jaccard, and the selected-asset liquidity/downside-risk diagnostics.

## 7. Second-stage test through `scripts/runs/run.py`

The comparison script exports 0/1 eligibility matrices directly to:

```text
datasets/excel/universe_comparison/basic_legacy_matrix.csv
datasets/excel/universe_comparison/basic_repaired_matrix.csv
datasets/excel/universe_comparison/advanced_matched_matrix.csv
datasets/excel/universe_comparison/advanced_native_matrix.csv
```

`run.py` already supports `portfolio_matrix`. To test whether the universe result survives the **full Bayesian/CVaR pipeline**, copy your chosen base YAML twice and change only these keys.

Basic repaired:

```yaml
portfolio: filtered
portfolio_matrix: "universe_comparison/basic_repaired_matrix.csv"
universe_cap: null
output_folder: "universe_ablation_basic_repaired"
```

Advanced matched:

```yaml
portfolio: filtered
portfolio_matrix: "universe_comparison/advanced_matched_matrix.csv"
universe_cap: null
output_folder: "universe_ablation_advanced_matched"
```

Then run both with exactly the same model, optimizer, execution and seed:

```bash
python scripts/runs/run.py -c universe_ablation_basic_repaired.yaml
python scripts/runs/run.py -c universe_ablation_advanced_matched.yaml
```

Do **not** change CVaR confidence, posterior, seed, execution alpha, transaction costs or any other parameter between these two runs. The universe must be the only treatment variable.

The equal-weight experiment is the identification test; the full `run.py` experiment is the transfer test.

## 8. What would justify the sentence “the advanced universe is better”?

A defensible statement would require a consistent pattern such as:

1. `advanced_matched` has higher equal-weight Sharpe/Sortino than `basic_repaired`;
2. maximum drawdown and CVaR are less severe;
3. paired block-bootstrap probabilities support those improvements rather than showing a fragile point estimate;
4. selected ETFs are at least as liquid and preferably have lower downside risk;
5. the result survives transaction costs and is not explained by radically different breadth;
6. the same direction survives the second-stage `run.py` full-pipeline test.

If the advanced filter increases return but worsens drawdown/CVaR, or if the result disappears under matched breadth, the correct article conclusion is **mixed evidence**, not superiority.

## 9. Important research distinction

The old `new_filter_data.py` history line uses a rolling **sum of traded amounts** and compares it with the integer `36`. That does not literally test “36 months with at least one trading day.” The comparison therefore keeps `basic_legacy` to reproduce the old idea but adds `basic_repaired` so the new filter is not credited merely for beating an implementation defect.
