# Code Assessment — Bayesian ETF Portfolio System

_Scope: `scripts/` (data pipeline, model, optimizers, strategies, execution, backtest loop, results). Notebooks were not audited in depth. Date of review: 2026-05-29._

---

## 1. What the system does

A monthly-rebalanced, walk-forward ETF backtest built around a Bayesian return simulator:

```
parse_close.py ─► returns + EWMA + business dates (CSV/XLSX)
filter_data.py ─► monthly eligibility matrix (ADV + history filters) ─► weights.xlsx
        │
        ▼
runs/run.py  (per month, expanding warm-up of 3y then rolling 1y window)
   1. BayessianModel.prediction()      → simulated forward returns (scenarios × assets)
   2. Optimizer (Bayesian-CVaR | Markowitz | Black-Litterman) → candidate weight sets
   3. Execution (alpha-blended transition + turnover cost)    → PnL on real future month
   4. BaseResults                       → preds/real/model/pnl CSVs + weights.xlsx
        │
        ▼
results/stats.py, multiple_stats.py, model_results.py → metrics tables + charts
```

Three entry points share the same loop:
- `runs/run.py` — single model + single optimizer family.
- `runs/combined_run.py` — iterates over **all** optimizer families and picks the best.
- `runs/strategy_run.py` — replaces optimizers with heuristic strategies (equal-weight, risk-parity, inverse-vol, liquidity, simple-long), used for benchmarks.

Design is **registry + config driven** (`*/registry.py` + `configs/*.yaml`), which is the strongest architectural decision in the repo — adding a model/optimizer/strategy is localized and clean.

---

## 2. Overall assessment

| Area | Rating | One-line |
|---|---|---|
| Architecture / extensibility | **Good** | Clean registry pattern, config-driven, clear layer separation. |
| Numerical / statistical correctness | **Weak** | Several real bugs in covariance cleaning, WAIC, R², metric design. |
| Backtest methodology | **At risk** | In-sample method/alpha selection each month → optimistic bias. |
| Engineering hygiene | **Weak** | No packaging, no tests, no deps file, fragile `sys.path` hacks, heavy dead code. |
| Reproducibility | **Partial** | Global seeds fix output, but data files & run config are implicit. |

The skeleton is good; the **inside of the math and the validation methodology are where the risk lives.** Most findings below are fixable without touching the architecture.

---

## 3. Correctness bugs (high priority)

### 3.1 Marchenko–Pastur noise threshold is inverted — `remove_noise_cov_matrix`
Duplicated identically in [markowitz.py:58](scripts/optimizers/markowitz.py:58), [black_litterman.py:62](scripts/optimizers/black_litterman.py:62), [bayessian.py:62](scripts/optimizers/bayessian.py:62):

```python
q = self.historical_returns.shape[1] / self.historical_returns.shape[0]   # q = N/T
lam_plus = sigma2 * (1 + 1/np.sqrt(q))**2                                   # ❌
```
The MP upper edge with `q = N/T` is `σ²(1 + √q)²`, **not** `σ²(1 + 1/√q)²`. Since `T > N` ⇒ `q < 1` ⇒ `1/√q > 1`, the threshold is inflated, so almost **all** eigenvalues are flagged as "noise" and averaged. The cleaned covariance collapses toward (near) equicorrelation, which silently pushes Markowitz/BL toward equal-weight and destroys the return signal. Note `calculations/covariance.py:mp_clip_correlation` gets this **right** (it defines `q = T/N` and uses `√(1/q)`) — the inline versions don't match it.
**Fix:** `lam_plus = sigma2 * (1 + np.sqrt(q))**2`, and ideally delete the three copies in favor of the correct shared util.

### 3.2 R² formula condition is backwards — [models/base.py:111](scripts/models/base.py:111)
```python
r_2 = 1 - ss_res/ss_tot if abs(ss_tot) <= 1e-5 else 1 - 1e+5 * ss_res
```
The branches are swapped: when `ss_tot ≈ 0` it divides by ~0 (blows up), and in the normal case it returns an arbitrary `1 - 1e5*ss_res`. The reported `r_squared` is meaningless. It also collapses every asset to a scalar (`pred_mean = np.mean(y_draws)`, `true_mean = np.mean(y_true)`), so even fixed it wouldn't be a per-asset R².

### 3.3 WAIC penalty term is not a posterior variance — [models/base.py:79](scripts/models/base.py:79)
```python
self.waic_var.append(np.sum(np.var(pointwise_loglik)))
```
WAIC's penalty is the variance of the log-likelihood **across posterior draws**, summed over data points. Here `pointwise_loglik` holds one KDE value per asset, so `np.var(...)` is the variance *across assets* of a single number each — not a posterior-draw variance. The WAIC values are not interpretable.

### 3.4 Predictive-density metric uses the cross-section as the "posterior" — [models/base.py:62‑69](scripts/models/base.py:62)
When `is_sampled_mean=False`, `pred_returns` rows are **future days**, not posterior draws. In `evaluate_metrics`, `y_draws = pred_returns[t]` is 1-D, so `samples = y_draws` (all assets on day _t_) is used as the predictive distribution for **every** asset _i_. The KDE is fit over the cross-section of assets and each asset's truth is scored against it. Log-score / coverage / WAIC therefore measure cross-asset dispersion, not a per-asset predictive distribution. The whole `evaluate_metrics` block needs its sample axis rethought.

### 3.5 `deviation_from_target` maximizes deviation — [optimizers/bayessian.py:106](scripts/optimizers/bayessian.py:106)
```python
deviation_from_target = -np.dot(weight_expression.T, self.cov_matrix @ weight_expression)
optimization_func = deviation_from_target          # then minimized
```
`cov` is PSD ⇒ the quadratic form is ≥ 0 ⇒ `optimization_func ≤ 0`. Minimizing a negative quantity **maximizes** the tracking-error term, i.e. pushes weights *away* from `w_previous + 0.05`. If the intent is "stay close to target," the sign is wrong (drop the leading `-`). This is the **active default** in `test.yaml` (`task_type: deviation_from_target`).

### 3.6 Clipping breaks the budget constraint (weights don't sum to 1)
- [strategies/base.py:40](scripts/strategies/base.py:40): `clip_weights` zeros small weights and caps large ones with **no renormalization** ⇒ `sum(w) ≠ 1`.
- [optimizers/base.py:102](scripts/optimizers/base.py:102): after `minimize` enforces `sum=1`, `weights = np.where(weights >= min_weight, weights, 0)` zeros small entries with no renormalization.

Downstream, execution computes `daily_returns = returns · w_exec` (see [executions/base.py:53](scripts/executions/base.py:53)), so a sub-1 weight sum silently models an uninvested cash sleeve at 0% return. This makes inverse-vol / liquidity / clipped-optimizer results not directly comparable to fully-invested benchmarks. Decide on the policy (renormalize vs. explicit cash) and apply it consistently.

### 3.7 Black-Litterman views are positional and can produce a singular Ω — [black_litterman.py:86](scripts/optimizers/black_litterman.py:86)
Views are hardcoded to column **indices** (`row[17]=1`, `row[62]=-1`, `row[:50]`, `row[50:100]`). The filtered ETF set and its column order change month-to-month, so the "views" attach to different ETFs each rebalance — effectively random. Worse, when the eligible set has < 63 assets, `row[50:100]` can be all zeros ⇒ a zero row in `P` ⇒ `Ω = τ P Σ Pᵀ` has a zero row/col ⇒ `np.linalg.inv(self.omega)` at line 141 raises. Views should be tied to tickers (or derived from a documented signal) and Ω regularized.

---

## 4. Methodological concerns (medium priority)

### 4.1 In-sample selection of method **and** alpha each month
In `results_by_fitting_model_alpha` ([run.py:151‑191](scripts/runs/run.py:151)), the optimizer emits several candidate weight sets (max_sharpe, min_variance, …). Each is scored by running execution on `past_month_returns = pred_returns` — i.e. the **same predicted returns the weights were optimized on** — and the best-scoring method + alpha is chosen. This is circular: you pick the method that best fits the in-sample objective, then report its out-of-sample PnL. `combined_run.py` compounds this by also choosing the best **optimizer family** the same way. Expect an optimistic bias in headline results. Method/alpha choice needs an out-of-sample or prior-period basis.

### 4.2 CVaR at 0.99 on ~21 scenarios
With `is_sampled_mean=False`, `len_returns = len(future_returns)` ≈ the number of trading days in the forward month (~21). So `pred_returns` has ~21 scenario rows. `np.quantile(losses, 0.99)` then `losses[losses >= q].mean()` ([optimizers/bayessian.py:97](scripts/optimizers/bayessian.py:97)) effectively averages the single worst point — CVaR99 is statistically meaningless at this sample size. Either raise the scenario count (the model already supports `n_samples` when `is_sampled_mean=True`) or lower the confidence level.

### 4.3 Lookback mismatch and high N/T
`find_start_time` uses a **1-year** window ([filter.py:24‑25](scripts/dataloader/filter.py:24)) though the comment says 3 years, and the warm-up offset in the loop uses 3 years. With ~252 daily rows and up to ~220 eligible assets, `q = N/T ≈ 0.87` — the sample covariance is extremely noisy, which is exactly why §3.1 matters. Reconcile the intended window and document it.

### 4.4 Transaction cost set to 100 bps — [configs/test.yaml:43](scripts/configs/test.yaml)
`c_bps: 1e-2` (= 1% per unit turnover) is the active value ("# high"); the realistic `0.0025` is commented out. At monthly turnover this dominates PnL. Make sure headline results aren't being judged at an unrealistic cost assumption.

### 4.5 Alpha heuristic saturates near 0.5
`exp_sum = 0.2*exp_sum + 0.8*return_value; alpha = sigmoid(exp_sum)` ([run.py:84](scripts/runs/run.py:84)). `return_value` is ~O(0.01), so `sigmoid(≈0) ≈ 0.5` almost always — the system trades ~half-way to target every month regardless. That may be a reasonable turnover damper, but it's an undocumented side effect rather than a tuned choice.

### 4.6 Future data lives on the model object
`BayessianModel` is constructed with `future_returns.values` ([run.py:312](scripts/runs/run.py:312)). It's currently used only by `evaluate_metrics` (not by `prediction()`), so there's no leakage today — but holding the realized future on the predictor is a leakage waiting to happen. Pass it to `evaluate_metrics(...)` explicitly instead of the constructor.

---

## 5. Engineering / maintainability (medium-low priority)

- **No packaging, no tests, no deps.** No `requirements.txt`/`pyproject.toml`, no `__init__.py`, no test files, empty `README.md`. Every module begins with a `sys.path.insert` hack, and import styles are inconsistent — `models/registry.py` and `models/bayessian.py` do `from models.base import ...` while optimizers do `from scripts.optimizers.base import ...`. This only works due to import-order side effects and will break under any normal packaging. Add a package layout (`pip install -e .`) and pin dependencies (numpy, pandas, scipy, scikit-learn, jax, beartype, loguru, tqdm, openpyxl, matplotlib).
- **Global RNG seed at import time** — `np.random.seed(42)` in [matrix.py:7](scripts/calculations/matrix.py:7) and [bayessian.py:9](scripts/models/bayessian.py:9). A library module setting a process-global seed is a side effect; it also means all "independent" Monte-Carlo draws share one stream. Use a local `np.random.default_rng(seed)` passed in.
- **`jax` is imported only for `jax.nn.sigmoid`** ([run.py:6](scripts/runs/run.py:6)) — a heavy dependency for one scalar function available in scipy/numpy. Drop it.
- **`covariance.py` and `covariance_cleaning.py` are unused by the live path.** They contain the *correct* covariance estimators (Ledoit-Wolf, OAS, RMT, detoning, correct MP) yet the optimizers reimplement a buggy MP inline. Wire the optimizers to these instead. `covariance_cleaning.py` also has Russian-language `ValueError` messages — fine internally, but inconsistent with the rest.
- **Heavy dead/experimental code.** Pervasive `# 1 way / 2 way / 3 way` commented alternatives and toggled code paths (e.g. `execution_process` switches between `run.py` and `strategy_run.py` via a comment at [executions/base.py:87](scripts/executions/base.py:87)). The "production" path is ambiguous. `features/base.py` is an unused stub whose `optimizer` has an inverted `if self.max_iter is None` guard. Prune or move to an `experiments/` area.
- **`evaluate_metrics` is very slow** — a `gaussian_kde` fit per (scenario × asset) per rebalance ([models/base.py:68](scripts/models/base.py:68)). With hundreds of rebalances this is the dominant cost and (per §3.4) measures the wrong thing.
- **Output folder is a hand-edited literal** in each `main()` (e.g. [run.py:407](scripts/runs/run.py:407)) — easy to overwrite the wrong results set. Promote to the YAML config or a CLI arg.
- **Mixed selection keys** — `info[:, 1]` (PnL) vs `info[:, -1]` (return) used interchangeably across `run.py`/`combined_run.py`. Harmless (monotonic) but confusing.

---

## 6. Strengths worth keeping

- Clean **registry + YAML** extensibility across all four component types.
- Sensible **walk-forward structure** with an explicit warm-up and a rolling/expanding switch.
- Good defensive numerics in places: `safe_cholesky` with jitter ([matrix.py:84](scripts/calculations/matrix.py:84)), PSD repair via eigenvalue clipping, `stds_safe` guards.
- A genuinely rich, **correct** covariance toolkit in `calculations/covariance*.py` (it just isn't connected).
- Broad, thoughtful **reporting** layer (Sharpe/Sortino/Calmar/Omega/VaR/CVaR/drawdown, normality tests, comparison charts).
- The eligibility filter in `filter_data.py` (rolling ADV share + minimum-history) is a reasonable, point-in-time investable-universe construction.

---

## 7. Prioritized fix list

1. **Fix the MP threshold** (§3.1) or route all optimizers through `calculations/covariance.py`. Highest impact on actual portfolio output.
2. **Fix the `deviation_from_target` sign** (§3.5) — it's the active objective in `test.yaml`.
3. **Resolve the budget-constraint leak** (§3.6): renormalize after clipping (or model cash explicitly), consistently across optimizers and strategies.
4. **Make method/alpha selection out-of-sample** (§4.1) — otherwise headline results are biased.
5. **Repair or replace the model metrics** (§3.2–3.4): R², WAIC, and the predictive-density sampling axis.
6. **Harden Black-Litterman views** (§3.7): ticker-based views + Ω regularization, guard the small-universe crash.
7. **Engineering baseline**: add packaging + `requirements.txt`, remove the `sys.path` hacks and `jax`, move seeds to local RNGs, and add a few unit tests (weights sum to 1, covariance PSD, CVaR sign, one-step PnL).
8. Right-size **CVaR scenario count** (§4.2) and confirm the **cost assumption** (§4.4) before trusting reported performance.

---

_No code was modified; this is an assessment only. Line references point to the live execution path used by `runs/run.py` with `configs/test.yaml`._
