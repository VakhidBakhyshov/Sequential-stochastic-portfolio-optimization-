# Combined README
This file consolidates the complete contents of all Markdown (`.md`) documentation files from the supplied archive into one master README. The latest archive versions are used, and each source file is identified below.
## Included files
- `md files/README.md`
- `md files/README_UPDATED.md`
- `md files/README_V2.md`
- `md files/SIGNAL_RESEARCH_README.md`
- `md files/CODE_AUDIT_AND_CHANGELOG.md`
- `md files/CODE_AND_MANUSCRIPT_AUDIT.md`
- `md files/EXECUTION_CONVENTION_SUMMARY.md`
- `md files/CDAR_PATCH_MANIFEST.md`
- `md files/IMPLEMENTATION_NOTES.md`
- `md files/DELIVERY_MANIFEST.md`
- `md files/UNIVERSE_COMPARISON_README.md`
- `md files/CDAR_NOTEBOOKS_README.md`

---

<!-- BEGIN SOURCE FILE: md files/README.md -->

# Companion code for "Sequential Stochastic Optimization of ETF Portfolios with Horizon-Diversified Risk Control"

This package contains the code and data behind every table and figure of the paper: the monthly
allocation policy (scenario-based CVaR allocation with adaptive parameters, partial execution and the
two-signal exposure overlay), the analysis scripts that produce the reported statistics, the datasets,
and the companion engine used for Appendices A and B and for the known-law identification of
Sections 3.9–3.10 and Appendix C. Everything runs from the package root with plain Python.

## Contents

```
README.md, LICENSE, requirements.txt, paper_map.md
datasets/csv/        raw panels: NewClosePrice.csv (adjusted daily closes), OpenPrice.csv, Volume.csv (share
                     volumes) for about 3,100 US-listed ETFs, 2016-01-04 to 2025-12-31; Meta.csv (fund
                     metadata: ISIN, name, provider, asset class, sector, geography, expense ratio, exchange,
                     replication method); vixcls.csv (FRED VIXCLS series)
datasets/excel/      built from the raw panels by the two preparation scripts and shipped for convenience:
                     last_filtered_weights.xlsx (point-in-time eligibility, one sheet per rebalance date),
                     business_dates.xlsx (first trading day of each month), new_etf_returns.csv and
                     new_etf_ewma.csv (daily log returns and their EWMA); rebalance_universe_matrix1.csv and
                     matrix2.csv are the two alternative universes of Section 7.9
scripts/             the engine: dataloader/, models/ (scenario law), optimizers/ (CVaR linear program,
                     adaptive parameters), executions/ (partial execution), calculations/, results/,
                     backtesting/, runs/run.py (the monthly loop), configs/ (one YAML per experiment arm)
*.py at the root     analysis scripts (tables, figures, controls, robustness), see paper_map.md
build_universe.py    builds the eligibility workbook from the raw panels (the liquidity screen of Section 2.2)
prepare_data.py      builds the derived return panels from the raw closes
run_arms.py          runs the experiment arms; reproduce.py runs the analysis chain
companion/           the colleague's engine (multi-signal system of Appendix A, drawdown-constrained
                     allocation of Appendix B, known-law and theorem-bridge scripts) with the result
                     folders the paper cites and its own requirements.txt
extensions/          material not used in the paper: synthetic_engine/ (the production policy run on
                     synthetic markets with a known law), universe_comparison/ and calibration/ (from the
                     companion package; the calibration code is shipped as is and is not run)
reference/           the tables (CSV/MD) and figures (PNG) as they appear in the paper, and the known-law
                     diagnostics of Sections 3.9–3.10 and Appendix B
results/             created by the runs
```

## Requirements

Python 3.11 or later. Install the dependencies with

```
python -m venv .venv
.venv\Scripts\activate          (Windows)   or   source .venv/bin/activate
pip install -r requirements.txt
```

The reported results were produced with Python 3.14, numpy 2.5.2, pandas 3.0.5, scipy 1.18.1 (HiGHS solves
the linear programs), matplotlib 3.11.1 and openpyxl 3.1.5. The companion engine has its own
`companion/requirements.txt`.

## Reproducing the paper

All commands are run from the package root.

1. Data. The files in `datasets/excel/` are shipped, and each is rebuilt exactly from the raw panels by
   `python build_universe.py` (the eligibility workbook, about half a minute) and `python prepare_data.py`
   (`new_etf_returns.csv`, `new_etf_ewma.csv`, `business_dates.xlsx`, a few minutes). Running both is
   optional; the rebuilt files are identical to the shipped ones.

2. Experiment arms. `python run_arms.py` runs the ten configurations of the paper in sequence
   (`main_dyn_strong` is the policy of the paper; the others are the static, generator, universe, state and
   execution variants of Sections 7.5 and 7.9). One arm takes about three minutes; each writes
   `results/<name>/` with `pnl.csv`, `weights.xlsx`, `forecast_risk.csv` and `dynamic_parameter_history.csv`.
   `python run_arms.py --grid` adds the scenario-count, history-rule and seed grid of Table 12 (about
   one hour). A single arm: `python run_arms.py main_dyn_strong`.

3. Analysis. `python reproduce.py` runs the analysis scripts in dependency order (overlays, verification
   and bootstraps, benchmarks, controls, robustness tables, figures) and writes the tables and figures
   next to the scripts; console output goes to `logs/`. `universe_compare.py` takes about 25 minutes;
   skip it with `--skip universe_compare`. Individual scripts can be run directly, for example
   `python verify_all.py` (Tables 7 and 8) or `python overlay_arms.py main_dyn_strong`.

4. Companion engine (Appendices A and B, Sections 3.9–3.10, Appendix C). From `companion/`:
   `python -m scripts.runs.sequential_diffusion_analysis --output results/sequential_diffusion_analysis`
   and `python -m scripts.runs.analytical_benchmark --output results/analytical_benchmark --execution-eta 0.5`
   reproduce Tables 2 and 3; `python -m scripts.runs.cdar_research_validation --output-dir results/cdar_validation`
   reproduces Table 19; `python -m scripts.runs.theorem_empirical_bridge` reproduces Table 20;
   `python -m scripts.runs.compare_cvar_cdar` reproduces the comparison of Appendix B from the shipped
   walk-forward outputs. The walk-forward arms of the companion engine take several hours each; their
   outputs are shipped under `companion/results/`. To re-run them, copy `datasets/` into
   `companion/datasets/` (the engine resolves data relative to its working directory).

Every script sets its own random seed (42, re-seeded at every rebalance date) so that repeated runs are
identical on the same library versions.

## Notes on the data

Closes are adjusted for distributions and splits; delisted funds remain in the panel. The liquidity
screen (`scripts/dataloader/liquidity_screen.py`) computes each fund's daily traded value as the mid price
times share volume, the 36-month rolling share of aggregate traded value, and marks a fund eligible when
its lagged share exceeds 0.5 x 10^-4 and it has traded in each of the preceding 36 months. The experiments
read the resulting workbook. The two additional rules of the eligibility condition
(an observed close on the last trading day before the rebalance, at least 60 months of observed closes
with funds present since the panel start exempt) are applied by `scripts/runs/run.py` from the
configuration keys `require_recent_price_days` and `require_history_days`.

## Licence and citation

MIT licence (see LICENSE). Please cite the paper when using this code:

Vukovic, D., Bakhishov, V., Zinovyev, V., Dalal, A. (2026). Sequential Stochastic Optimization of ETF Portfolios with
Horizon-Diversified Risk Control. [Journal, volume, pages].

<!-- END SOURCE FILE: md files/README.md -->

---

<!-- BEGIN SOURCE FILE: md files/README_UPDATED.md -->

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

<!-- END SOURCE FILE: md files/README_UPDATED.md -->

---

<!-- BEGIN SOURCE FILE: md files/README_V2.md -->

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

<!-- END SOURCE FILE: md files/README_V2.md -->

---

<!-- BEGIN SOURCE FILE: md files/SIGNAL_RESEARCH_README.md -->

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

<!-- END SOURCE FILE: md files/SIGNAL_RESEARCH_README.md -->

---

<!-- BEGIN SOURCE FILE: md files/CODE_AUDIT_AND_CHANGELOG.md -->

# Code Audit and Change Log

## Status

- Python compilation: passed for the repository.
- Core unit tests: 4 passed.
- Synthetic reporting smoke test: passed; generated the dashboard and all reporting artifacts.
- Full historical ETF backtest: not run because the uploaded archive does not contain the raw point-in-time ETF dataset used by the original environment.

## High-impact defects found and corrected

| Severity | Original behavior | Why it mattered | Correction |
|---|---|---|---|
| Critical | Advanced config used pure `cvar_returns` minimization. | The optimizer could deliberately choose very low-return assets; this is consistent with the reported 0.71% annual return and negative Sharpe. | Primary task changed to expected-return maximization subject to a CVaR budget; a small candidate grid compares it with a mean-CVaR-Sharpe objective. |
| Critical | Smart cash exposure was normalized back to one. | `allow_cash_overlay` silently had no effect inside the nested engine. | `build_full_weight_series(... preserve_total_exposure=True)` preserves weights summing below one. |
| Critical | Monte Carlo scenarios were evaluated as an ordered time path. | Scenario rows have no temporal order; model metrics were statistically invalid. | Metrics now compare posterior cross-sectional distributions with the realized holding-period vector. |
| High | Scenario fan was effectively one-day while execution was next month. | Expected return, volatility and CVaR were in the wrong units. | Daily shocks are aggregated/compounded over `horizon`. |
| High | `w_current` was set to target even when partial execution used alpha below one. | Next-month turnover and transaction costs used an unattained state. | Actual executed weights are carried forward and saved separately. |
| High | Dashboard showed an ex-post matrix instead of the model matrix. | The audit could not verify what the optimizer actually saw. | Model snapshots now write `risk_matrices.json`; dashboard reads the exported covariance/correlation. |
| High | ETF prices used forward and backward fill before observation counts. | Newly listed ETFs could appear to have pre-inception history. | Observation counts use raw data; prices are forward-filled only after first availability. |
| Medium | Beta denominator used full benchmark variance rather than pairwise observations. | Missing-data patterns biased beta. | Beta uses pairwise benchmark variance. |
| Medium | Risk-adjusted return mixed annualized volatility and an extra square-root factor. | Formula units were inconsistent. | Annualized Sharpe-like score is mean daily return / daily volatility × √252. |
| Medium | “WAIC” was calculated without pointwise posterior log-likelihood draws. | The label overclaimed a Bayesian information criterion. | Renamed `predictive_deviance_proxy`; full WAIC requires likelihood draws. |
| Medium | Treynor assumed beta equal to one. | It was not a Treynor ratio. | Beta/Treynor are only reported when a benchmark return series exists. |
| Medium | Strategy runner ignored its `use_backtest_engine` flag. | Strategy parameters were selected through legacy in-sample logic. | Strategy runner now uses the same nested selection engine, with a causal fallback only when disabled/erroring. |
| Medium | Smart overlay multiplied many timers, often collapsing exposure. | This can explain severe CAGR loss without corresponding protection. | Default is a bounded geometric mean; product/minimum/convex blend remain explicit ablations. |

## Remaining limitations before journal submission

1. Point-in-time ETF membership, delistings, mergers, name changes and fund closures must be reconstructed to address survivorship bias.
2. Daily close/volume data cannot identify true spreads or market impact. Cost sensitivity is mandatory; better data should replace the proxy.
3. Fixed-parameter GARCH is a fast regime proxy, not a fitted likelihood model. Compare with rolling/EW volatility and, optionally, a fitted GARCH package.
4. Posterior mean and covariance uncertainty remain simplified. The paper should be precise about “empirical-Bayes shrinkage” rather than claiming a full MCMC posterior unless a full sampler is added.
5. DSR must use the real number of all tried variants, not only final saved winners. Freeze and log the complete search grid before the final test.
6. The paper should report matched-average-exposure comparisons so the combined overlay is not rewarded merely for holding more cash.
7. The Advanced Smart Bayesian policy cannot be called best until the fixed pipeline is rerun on the real dataset and passes the promotion criteria.

## 2026-09-09 analytical-diffusion extension

The known-distribution layer was strengthened beyond the existing BM/GBM path and scalar-CVaR checks.  The repository now contains an exact-law oracle for the same return/CVaR/turnover decision problem and compares it with the finite-scenario LP.  New diagnostics cover: (i) exact Gaussian CVaR and its weight gradient, (ii) CVaR sensitivity to Brownian correlation, (iii) finite-scenario convergence toward the exact-law oracle, (iv) plug-in drift/covariance/risk estimation error as the lookback grows, and (v) exact mean-shift identification of the centred signal.  These are verification results only and are not promoted to empirical ETF-performance claims.

<!-- END SOURCE FILE: md files/CODE_AUDIT_AND_CHANGELOG.md -->

---

<!-- BEGIN SOURCE FILE: md files/CODE_AND_MANUSCRIPT_AUDIT.md -->

# Code + manuscript audit and final-file decision

## Scope

I reviewed the uploaded framework as a codebase rather than treating `*-upd.py` as automatically newer/better. The archive contains **143 Python files, 7 notebooks, 122 YAML/YML files**, and seven direct `file.py` / `file-upd.py` pairs. I compared each pair, traced the main monthly pipeline, execution state, transaction costs, nested selection, predictive scenarios, CVaR/CDaR, result statistics and exposure overlays against the manuscript and `EXECUTION_CONVENTION_SUMMARY.md`.

The most important finding is that the final code should **not** be either `run.py` or `run-upd.py` unchanged. The strongest final version is a **merge based on the current non-`-upd` code**, with the Policy-C execution/accounting corrections from the markdown.

---

## 1. Which duplicate file should be final?

| Pair | Final choice | Reason |
|---|---|---|
| `scripts/runs/run.py` vs `run-upd.py` | **MERGE, starting from `run.py`** | Current `run.py` contains the richer final research pipeline: scenario-path handling, centered CVaR/CDaR diagnostics, trial ledger, risk-matrix audit, post-processing and current validation hooks. `run-upd.py` contains an older state-reference switch, but neither old version implements Policy C correctly because both effectively use one state for two different economic roles. The patched `run.py` separates previous-target optimizer anchor from actual executed holdings. |
| `scripts/models/base.py` vs `base-upd.py` | **Use `base.py`** | Current file treats scenario rows as a predictive distribution and compares them with the realized holding-period outcome. The `-upd` version contains older KDE/WAIC/R² logic and treats objects less consistently with the current scenario architecture. |
| `scripts/models/bayessian.py` vs `bayessian-upd.py` | **Use `bayessian.py`** | Current file supports horizon-aware scenario generation, daily covariance, reproducible posterior mean construction, ordered scenario paths for CDaR, risk-matrix snapshots and scenario-return-type handling. Those are required by the manuscript's CVaR/CDaR and structural diagnostics. |
| `scripts/optimizers/base.py` vs `base-upd.py` | **Use `base.py`** | Current file has the more precise sample-CVaR audit calculation (including fractional tail mass) and stronger LP/binding/shadow-price diagnostics, which are needed for the budget-degeneracy claims. |
| `scripts/backtesting/walk_forward_engine.py` vs `walk_forward_engine-upd.py` | **Use patched `walk_forward_engine.py`** | Current file preserves CDaR scenario paths, candidate-attempt/trial audit and robust selection diagnostics. Patch additionally splits optimizer reference from execution state and prevents post-execution re-normalization. |
| `scripts/backtesting/engine_integration.py` vs `engine_integration-upd.py` | **Use patched `engine_integration.py`** | Current file includes smart-signal integration and richer audit outputs. Patch passes the two Policy-C states separately. |
| `scripts/results/stats.py` vs `stats-upd.py` | **Use `stats.py`** | This is not close: current file is the full research version (PSR/DSR, extended metrics, train/test, rolling Sharpe/Sortino, annual/monthly plots, false-strategy diagnostics, artifact export). `stats-upd.py` is a much smaller older subset. |

### Practical cleanup rule

After integrating this patch and reproducing the final run, move all seven `*-upd.py` files to an archive folder or delete them from the publication branch. Keeping both names in the active source tree makes it too easy to run the wrong implementation and undermines reproducibility.

---

## 2. Main execution/accounting problem

The framework had three objects that must be distinct:

1. **previous target**: useful as the optimizer's regularization/turnover anchor;
2. **previous executed holdings**: the portfolio economically held at the start of the new trade;
3. **new target**: the current optimization output.

Using one `w_previous` variable for all three roles creates the exact inconsistency described in the markdown. The patched pipeline uses:

```text
optimizer anchor at t  = target_{t-1}
trade starting state   = executed_{t-1}
executed_t              = executed_{t-1} + eta_t * (target_t - executed_{t-1})
optimizer anchor at t+1 = target_t
trade starting state t+1= executed_t
```

This is Policy C.

### Why not Policy B (executed holdings everywhere)?

It is internally neat but changes the optimizer's turnover reference and, according to the supplied rerun summary, causes materially more turnover/oscillation. The state that determines *economic trading* does not have to be identical to the reference point used in a regularized optimization objective. The paper must be explicit about this distinction.

---

## 3. Transaction-cost convention

The execution code computes

```python
turnover = sum(abs(w_exec - w_prev_exec))
cost = c_bps * turnover
```

That quantity is **gross/two-way traded notional (L1 turnover)**: purchases and sales are both counted. If a fully invested portfolio sells 20% and buys 20%, L1 turnover is 40%, while conventional one-way turnover is 20%.

Therefore, with `c_bps = 0.001`:

- the code charges **10 bp per unit bought or sold**;
- equivalently, for a self-financing fully invested rebalance, it is **20 bp per unit of conventional one-way turnover**.

The manuscript should not call `sum(abs(delta w))` “one-way turnover.” The patch uses the unambiguous language **gross traded notional / L1 turnover**.

---

## 4. Missing-return bug for assets leaving the feasible set

This was a real accounting issue. Previously, next-month returns were loaded only for the *current eligible set*. With partial execution, an ETF that leaves the eligible set can still have a positive executed holding while it is being liquidated. Its return must therefore still enter realized PnL.

The patched `run.py` now builds:

- `future_returns`: current eligible names, used by the model/evaluation logic;
- `future_returns_execution`: the complete portfolio-state ticker list in the same order as the full weight vector, used for actual PnL.

`BaseExecution` detects the full-state frame and books all economically held positions. Required exits are also automatically included in turnover because the target weight for an ineligible name is zero while its previous executed holding may be positive.

---

## 5. Exposure-overlay trading cost

The original `overlay_arms.py` changes risky exposure but does not charge any incremental cost for changing the exposure multiplier. The manuscript currently says exposure changes generate trading, so code and text were not aligned.

The patched script adds an explicit option:

```bash
python overlay_arms.py <run_folder> --charge-exposure-costs --exposure-cost-bps 0.001
```

The separate exposure-channel turnover is

```text
|k_t - k_{t-1}|
```

and cost is `rate * |delta k|`.

This is intentionally reported as a **separate scalar exposure-channel cost**, not as an exact additive decomposition of full-vector L1 turnover when both composition and exposure change simultaneously.

### Recommended publication handling

Run and report **both**:

- Policy C with composition/execution costs only;
- Policy C + exposure-overlay cost at 10 bp.

Choose the first as baseline only if the paper explicitly states that overlay implementation costs are excluded from the baseline and the second is a robustness check. If the paper says exposure changes are charged in the baseline, then use the second as the baseline.

---

## 6. Additional harmful/important issue found: `strict_nested_engine` was ignored

Two publication YAML files declare:

```yaml
strict_nested_engine: true
```

but the original `run.py` caught every nested-engine exception and silently fell back to the legacy direct optimization path. That means a supposedly “strict” production run could contain a mixture of two selection procedures without stopping.

The patched `run.py` now raises immediately when `strict_nested_engine: true`. This is important for a publication run because otherwise the trial ledger, selection chronology and reported algorithm can differ across months.

---

## 7. Nested-engine partial-execution bug fixed

Inside `walk_forward_engine.py`, candidate weights were partially executed and then normalized back to one. That can silently undo the economic meaning of partial execution when total risky mass is below one (for example after required exits or when a cash sleeve exists).

The patch removes this post-execution renormalization. Target composition is normalized when it is constructed; the executed state is left as the affine state transition.

---

## 8. Manuscript/code inconsistencies that still require a new empirical rerun

The code patch makes the implementation internally coherent, but **old reported numerical results do not become valid automatically**. The manuscript must be regenerated from a fresh final-policy run.

The supplied markdown already shows that Policy C changes the production numbers. Therefore do not keep the old Table 5 / Section 7 production values and simply change the prose.

At minimum, regenerate:

- Table 5 main performance metrics;
- Table 6 exposure attribution;
- fast/slow/HDRC paths and drawdown plots;
- PSR/DSR from the exact final trial ledger;
- transaction-cost/turnover tables and break-even cost;
- fast/slow signal correlation diagnostics;
- VIX comparison numbers;
- execution mechanism section;
- all downstream slides/tables that use the production run.

The matched CVaR–CDaR experiment, known-law theory and purely structural mathematics only stay unchanged if they are genuinely produced by an independent matched branch that does not use the corrected production-run accounting outputs.

---

## 9. Manuscript wording that should change for Policy C

The current text says the executed portfolio is also the turnover reference passed into the next optimizer. Under the recommended split convention, that is no longer the definition.

Use this distinction throughout notation, Model §3.4/3.6, Theorem 2 discussion, Algorithm Appendix J and execution results:

```text
q_{t-1}: previous target composition used as the optimizer's turnover/reference anchor.
x_{t-1}^{exec}: actual risky holdings from which current trading, realized returns and transaction costs are computed.
x_t^{tar}: current total-wealth/risky target.
x_t^{exec}: actual holdings after partial execution.
```

Do **not** say “the optimizer starts from executed holdings” if using Policy C. It does not. The economic trade starts from executed holdings; the optimizer penalty is anchored on the previous target.

Also change transaction-cost wording from “10 bp per unit of one-way turnover” to something like:

> Transaction cost equals 10 bp times gross traded notional, where gross traded notional is the L1 change in implemented risky holdings and therefore counts buys and sells separately.

If using the optional overlay-cost implementation, add a separate sentence defining `|k_t-k_{t-1}|` and the applied rate.

---

## 10. Files in this patch

### Changed / new final-policy files

- `scripts/runs/run.py`
- `scripts/backtesting/walk_forward_engine.py`
- `scripts/backtesting/engine_integration.py`
- `scripts/executions/base.py`
- `scripts/executions/accounting.py` **(new)**
- `overlay_arms.py`
- `tests/test_execution_convention.py` **(new)**
- `notebooks/execution_transaction_cost_audit.ipynb` **(new)**

### Canonical files selected from duplicate pairs (kept as-is from the stronger current implementation)

- `scripts/models/base.py`
- `scripts/models/bayessian.py`
- `scripts/optimizers/base.py`
- `scripts/results/stats.py`

Do not replace these four with their `*-upd.py` versions.

---

## 11. Validation performed here

- All changed Python files compile successfully when merged into the uploaded framework.
- The whole `scripts/` tree compiles after the patch.
- New execution-convention regression tests: **5 passed**.
- The tests check exact partial-execution contraction/turnover identities, Policy-C state separation, the 10-bp/L1 cost convention, required liquidation, and the separate exposure-turnover channel.

A full 83-month numerical rerun was **not** treated as completed in this review. That rerun is required before updating empirical values in the manuscript, because the corrected state/cost convention is intentionally different from the archived production run.

---

## Final code-branch recommendation

Use the non-`-upd` implementation as the base branch, apply this patch, archive the `*-upd.py` files, set `strict_nested_engine: true` for the publication run, and regenerate all production results once under a single frozen convention. Do not mix archived Policy-A results, Policy-B state semantics, and Policy-C code in the final article.

<!-- END SOURCE FILE: md files/CODE_AND_MANUSCRIPT_AUDIT.md -->

---

<!-- BEGIN SOURCE FILE: md files/EXECUTION_CONVENTION_SUMMARY.md -->

# Execution and cost convention: summary

## Why we looked

Two points had to be checked before submission:

1. The final policy must carry executed holdings into the next decision, and any archived evidence in Section 7 must be labelled as archived.
2. The 10 bp cost convention in the manuscript must match the code exactly.

## What the code check found

- **Wrong state carried.** The reported run carried last month's target into the next decision, while the manuscript says executed holdings are carried. Both the optimizer's turnover penalty and the partial-execution step started from the previous target.
- **Costs under-charged.** As a result, costs were charged on about half of the actual trading: 0.50 turns per year (0.44% cumulative) were charged, against 1.02 turns (0.79%) actually traded.
- **Returns and exposure changes missing.** In six months, returns of held funds that had left the eligible set were not booked. Changes in the overlay's exposure are not charged.
- **Cost label wrong.** The code charges 10 bp on every unit bought and every unit sold. The manuscript's formula gives the same numbers, but its "one-way" label does not match. The code's rate equals 20 bp per unit of one-way turnover.

## What we changed and how

In a copy of the package (`hdrc_policyC`), we added one option to the run engine, called policy C:

- The optimizer's turnover penalty stays anchored on the previous target.
- Each month's trade starts from the executed holdings.
- Costs are 10 bp on every unit actually traded.
- The returns of all held funds are booked.

We reran all 31 paper configurations and the full analysis chain. The checks pass:

- Holdings and costs match the stated formulas to 10⁻¹⁶ in every month.
- The old setting still reproduces the reported results exactly.

## Options

| Option | What it means | HDRC Sharpe / max DD | Unmanaged base | Verdict |
|---|---|---|---|---|
| A, as reported | Previous target carried; half the trades charged | 0.859 / −6.55% | 0.660 | Contradicts the manuscript and under-charges costs. A reviewer can detect this from the weights. |
| A, corrected costs | Same decisions, with costs and returns on actual holdings | 0.849 / −6.92% | 0.645 | Accounting is right, but the paper would have to describe a strategy that tracks a target it never holds. |
| B, executed book everywhere | What the manuscript currently says | 0.627 / −10.79% | 0.625 | Worst option. The optimizer oscillates: turnover nearly doubles and 37 names are held. |
| **C, split (recommended)** | Optimizer anchored on the previous target; holdings and costs on the executed book | **0.811 / −7.07%** | 0.617 | Exact accounting, and executed holdings are carried. The manuscript only has to describe the anchor. |
| Full execution (η = 1) | Trade fully to the target every month | 0.874 / −6.80% | 0.664 | No gap between target and holdings, and the same trading as C. But it removes partial execution from the model and would be chosen after seeing results. |

A second, smaller choice concerns exposure changes. We can leave them uncharged and say so (C: 0.811), or charge them at 10 bp (C: 0.798 / −7.18%).

## Effects of switching to C

**Numbers.** Every result that comes from the production run changes:

- HDRC goes from 0.859 / −6.55% to 0.811 / −7.07%.
- The deflated Sharpe ratio goes from 96.0% to 94.6%.

These stay the same: data, the 1/N and Markowitz benchmarks, the theory, the known-law work and the CVaR–CDaR comparison.

**Conclusions.** The qualitative results all hold:

- The drawdown advantage over the fast channel, 1/N and the base is significant (P = 0.98–1.00).
- Timing accounts for about half of the drawdown cut.
- HDRC beats random timing.
- It matches the VIX rule.
- The product is the best combination.
- The two signals carry distinct information.

**Four claims need rewording:**

1. The unmanaged allocator is now below 1/N on Sharpe (0.617 vs 0.638).
2. The deflated Sharpe ratio (94.6%) is below the 95% threshold.
3. The slow signal has a small negative link to next-month return (−0.23, p = 0.04). The text can no longer say it has no return content.
4. Partial execution does not reduce trading, because full execution trades as much and earns more. The execution section can claim exact accounting, not a cost saving.

**Manuscript text beyond numbers:**

- **Turnover reference.** Define it as the previous target: notation, equations 23 and 33, Proposition 5, Appendix B.4.
- **Execution.** Write it on the risky composition, with exposure applied in full each month. This covers equations 56–61 and 81–86; restate Theorem 2.
- **Costs.** Say 10 bp per unit bought or sold, with turnover reported two-way. State whether exposure changes are charged.
- **Section 7.7.** Rewrite it so the B run becomes the mechanism test that justifies the anchor.
- **Remove the history.** Delete the "archived" and "development" wording.
- **Earlier items.** Fix S = 5,000, the 0.17 correlation sentence, the missing VIX results and the leftover image alt texts.

## Does this resolve the two points?

- **Point 1, executed holdings: yes.** Executed holdings are carried and booked. Every Section 7 result comes from the same final code, so nothing has to be labelled as archived. The manuscript must state that the optimizer's anchor is the previous target.
- **Point 2, cost convention: yes, in the code.** Costs equal 10 bp times the actual trades, checked in every month. The manuscript still needs the label fix and the exposure-cost statement.

## Next steps if we choose C

1. Switch the companion package to C and regenerate its reference tables and figures.
2. Update the manuscript numbers and text as listed above.
3. Update the slides.

Full results: `COMPARISON.md` and the `comparison/` folder.

<!-- END SOURCE FILE: md files/EXECUTION_CONVENTION_SUMMARY.md -->

---

<!-- BEGIN SOURCE FILE: md files/CDAR_PATCH_MANIFEST.md -->

# CDaR Research Completion Manifest

## Scope
This patch completes and verifies the CDaR research arm of the sequential stochastic portfolio framework without manufacturing historical ETF results that cannot be regenerated from the uploaded archive. The archive does not contain the immutable point-in-time ETF price/volume vintage required for a new historical CVaR-versus-CDaR backtest.

## Main implementation corrections

1. **Outer calibration now supports CDaR as a first-class tail-risk objective.** `scripts/optimization/optuna_calibration.py` and `analysis.py` recognize `CDaR (95%)`, `cdar_95`, and `abs_cdar_95`, and accept `tail_risk_objective: cdar`.
2. **Nested calibration parameter routing is corrected.** Trial-specific optimizer values are synchronized into `optimizer_param_grid` before the nested walk-forward engine is run. Previously, fixed values in that grid could override Optuna's trial-specific optimizer parameters. This affected the scientific interpretation of optimizer calibration generally, not only CDaR.
3. **True two-signal research arms are explicit.** `include_conviction_timer: false` makes the paired exposure engine exactly backward realized-volatility + one forward risk coordinate (CVaR or CDaR), instead of implicitly retaining the conviction timer.
4. **CDaR comparison inference is dependence-aware.** `compare_cvar_cdar.py` uses paired circular block bootstrap by default, adds constraint/slack/shadow-price mechanism summaries, forward-risk correlation, and optional target/executed-weight L1 distances.
5. **The exact-law CDaR validation layer is strengthened.** For zero-drift Brownian motion, terminal drawdown has the half-normal law. The code now checks closed-form VaR/CDaR against Monte Carlo, tests a CVaR-vs-CDaR ranking reversal, and measures finite-scenario convergence.
6. **A deterministic known-DGP CDaR calibration experiment is supplied.** `cdar_synthetic_calibration.py` provides a reproducible calibration/validation workflow when the historical point-in-time data vintage is unavailable.

## Modified files

- `scripts/optimization/optuna_calibration.py`
- `scripts/optimization/analysis.py`
- `scripts/optimization/plots.py`
- `scripts/risk_controls/smart_signals.py`
- `scripts/theory/cdar_benchmark.py`
- `scripts/runs/cdar_research_validation.py`
- `scripts/runs/compare_cvar_cdar.py`
- `tests/test_optuna_calibration.py`
- `tests/test_cdar_research.py`

## New files

- `scripts/configs/optuna_calibration_cdar.yaml`
- `scripts/configs/optuna_calibration_cvar_paired.yaml`
- `scripts/configs/bayessian_cdar_two_signal.yaml`
- `scripts/configs/bayessian_cvar_two_signal.yaml`
- `scripts/runs/cdar_synthetic_calibration.py`
- `CDAR_research_analysis.ipynb`
- `results/cdar_validation/*`
- `results/cdar_synthetic_calibration/*`
- `results/cdar_delivery/*`

## Executed validation

Repository regression suite: **61 passed**.

The sandbox did not contain the declared `beartype` package and package download is blocked. To exercise the repository logic without changing the deliverable dependency graph, the tests were run with a temporary no-op `beartype` compatibility shim outside the repository. That shim is **not** included in this patch.

### Exact-law / controlled CDaR diagnostics

- Zero-drift Brownian terminal CDaR, alpha=0.95: Monte Carlo `0.4636266`; exact half-normal value `0.4675606`; absolute error `0.0039340`.
- Zero-drift Brownian terminal VaR, alpha=0.95: Monte Carlo `0.3898301`; exact `0.3919928`; absolute error `0.0021627`.
- Same-terminal-return path example: CDaR difference `0.18`.
- Explicit risk-geometry ranking reversal: terminal-loss criterion prefers path A while CDaR prefers path B.
- Controlled CDaR LP: budget `0.0563400453`; optimized CDaR `0.0563400453`; zero numerical budget violation; fully invested; economic CDaR shadow price `0.304266`.

### Controlled sequential calibration

Risk-control selection:
- alpha `0.95`
- CDaR budget multiplier `1.00`
- turnover penalty `0.006`
- 18/18 sequential decisions solved
- annualized test return `0.4227%`
- annualized test volatility `1.2897%`
- zero-risk-free Sharpe `0.3278`
- mean out-of-sample CDaR95 `4.7061%`
- max out-of-sample CDaR95 `8.7533%`
- mean executed L1 turnover `0.1041`

Max-return selection:
- alpha `0.90`
- CDaR budget multiplier `1.15`
- turnover penalty `0.003`
- annualized test return `0.5089%`
- annualized test volatility `1.4165%`
- zero-risk-free Sharpe `0.3593`
- mean out-of-sample CDaR95 `5.2046%`
- max out-of-sample CDaR95 `9.2451%`
- mean executed L1 turnover `0.3159`

These calibration figures are controlled known-DGP evidence, **not historical ETF performance**.

## Historical paired rerun commands

Once the immutable ETF vintage is available in the expected data paths:

```bash
python3 -m scripts.runs.run --config-path bayessian_cvar_paired.yaml
python3 -m scripts.runs.run --config-path bayessian_cdar.yaml
python3 -m scripts.runs.compare_cvar_cdar \
  --cvar-run <CVaR_RUN_DIR> \
  --cdar-run <CDaR_RUN_DIR> \
  --output-dir results/cvar_cdar_comparison \
  --block-length 6
python3 -m scripts.runs.cdar_research_validation
python3 -m scripts.runs.cdar_synthetic_calibration
python3 -m tests.test_cdar_research

```

For matched true two-signal arms use `bayessian_cvar_two_signal.yaml` and `bayessian_cdar_two_signal.yaml`.

For outer calibration use `optuna_calibration_cvar_paired.yaml` and `optuna_calibration_cdar.yaml` with the project's existing calibration entry point.

## Claim boundary

The patch supports the mathematical formulation, sparse LP, exact-law validation, scenario convergence, calibrated known-DGP experiment, paired configuration/control design, nested-path causality, two-signal and multi-signal wiring, reporting, and statistical comparison infrastructure for CDaR. It does **not** claim a historical CDaR Sharpe ratio, maximum drawdown, selected ETF parameter vector, or superiority over CVaR until the frozen point-in-time ETF data are rerun.

<!-- END SOURCE FILE: md files/CDAR_PATCH_MANIFEST.md -->

---

<!-- BEGIN SOURCE FILE: md files/IMPLEMENTATION_NOTES.md -->

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

<!-- END SOURCE FILE: md files/IMPLEMENTATION_NOTES.md -->

---

<!-- BEGIN SOURCE FILE: md files/DELIVERY_MANIFEST.md -->

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

<!-- END SOURCE FILE: md files/DELIVERY_MANIFEST.md -->

---

<!-- BEGIN SOURCE FILE: md files/UNIVERSE_COMPARISON_README.md -->

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

<!-- END SOURCE FILE: md files/UNIVERSE_COMPARISON_README.md -->

---

<!-- BEGIN SOURCE FILE: md files/CDAR_NOTEBOOKS_README.md -->

# CDaR notebooks

Generated as CDaR-specific counterparts to the uploaded CVaR notebooks.

## 1. `CDAR_FINAL_results.ipynb`
For `bayessian_cdar.yaml` and `bayessian_cdar_two_signal.yaml`. Includes performance metrics, cumulative wealth, drawdowns, rolling risk, monthly heatmaps, predictive raw/centred CDaR, forward/backward timers, exposure, composition, concentration, turnover, execution alpha, dynamic controller state, model diagnostics, nested selection and signal-research previews. It also optionally adds the paired CVaR arm.

## 2. `CDAR_optuna_parameter_calibration_analysis.ipynb`
For `optuna_calibration_cdar.yaml`. Uses the CDaR-specific selection weights from that YAML, plots return-CDaR frontiers, parameter effects, alpha x budget interaction surfaces, Pareto configurations, trial progression and can write frozen selected YAMLs.

## 3. `CDAR_vs_CVAR_paired_analysis.ipynb`
For `bayessian_cvar_paired.yaml` versus `bayessian_cdar.yaml`. Includes matched metrics, cumulative/drawdown plots, monthly treatment effects, a moving-block paired bootstrap, forward-risk comparison, L1 weight distance, active-set Jaccard and turnover.

All notebooks default to analysis-only mode so they do not accidentally launch multi-hour runs.

<!-- END SOURCE FILE: md files/CDAR_NOTEBOOKS_README.md -->
