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
