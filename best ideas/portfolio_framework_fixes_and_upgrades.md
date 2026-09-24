
# Portfolio framework fixes and upgrade patch

This file contains practical drop-in code patches for the uploaded framework. It is written for the current `all_3.md` structure, where the live code is organized as `scripts/calculations`, `scripts/models`, `scripts/optimizers`, `scripts/strategies`, and `scripts/runs`.

## 0. Fix summary

Apply in this order:

1. Add `scripts/calculations/portfolio_utils.py`.
2. Replace `scripts/optimizers/base.py`.
3. Patch `scripts/optimizers/bayessian.py`.
4. Replace `BaseModel.evaluate_metrics` in `scripts/models/base.py`.
5. Patch `scripts/models/bayessian.py` so predictions are real scenarios, not only future-day rows.
6. Patch `scripts/strategies/base.py`.
7. Add new strategy file `scripts/strategies/momentum_quality.py` and register it.
8. Patch `runs/strategy_run.py` and `runs/combined_strategy_run.py` so heuristic strategies receive returns, not price levels.
9. Replace hard-coded Black-Litterman views with signal-based views.

---

## 1. New utility file: `scripts/calculations/portfolio_utils.py`

```python
from __future__ import annotations

import numpy as np

EPS = 1e-12


def as_2d_float(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape={x.shape}")
    return np.where(np.isfinite(x), x, np.nan)


def sigmoid_np(x: float) -> float:
    """Small replacement for jax.nn.sigmoid, avoids importing JAX for one scalar."""
    x = float(np.clip(x, -50.0, 50.0))
    return float(1.0 / (1.0 + np.exp(-x)))


def normalize_long_only(
    weights: np.ndarray,
    min_weight: float = 0.0,
    max_weight: float | None = None,
    fallback_n: int | None = None,
) -> np.ndarray:
    """
    Long-only postprocessor:
    - removes non-finite and negative weights;
    - sets tiny weights to zero;
    - renormalizes to sum to 1;
    - caps max weights and redistributes the excess.

    Important: min_weight is used as a position threshold, not as a hard lower bound
    for all assets. Otherwise min_weight=0.01 is infeasible when N > 100.
    """
    w = np.asarray(weights, dtype=float).copy()

    if w.size == 0:
        raise ValueError("weights must not be empty")

    w = np.where(np.isfinite(w), w, 0.0)
    w = np.maximum(w, 0.0)

    if min_weight > 0:
        w[w < min_weight] = 0.0

    if w.sum() <= EPS:
        w = np.ones(w.size, dtype=float) / w.size
    else:
        w = w / w.sum()

    if max_weight is None:
        return w

    max_weight = float(max_weight)
    if max_weight <= 0:
        raise ValueError("max_weight must be positive")

    active = w > 0
    active_count = int(active.sum())
    if active_count == 0:
        return np.ones(w.size, dtype=float) / w.size

    max_eff = max(max_weight, 1.0 / active_count)

    capped = np.zeros_like(w, dtype=bool)
    for _ in range(w.size + 2):
        over = (w > max_eff) & (~capped)
        if not np.any(over):
            break

        capped[over] = True
        w[over] = max_eff

        remainder = 1.0 - w[capped].sum()
        uncapped = ~capped
        if remainder <= EPS or not np.any(uncapped):
            break

        uncapped_sum = w[uncapped].sum()
        if uncapped_sum <= EPS:
            w[uncapped] = remainder / uncapped.sum()
        else:
            w[uncapped] = w[uncapped] / uncapped_sum * remainder

    w = np.maximum(w, 0.0)
    total = w.sum()
    if total <= EPS:
        return np.ones(w.size, dtype=float) / w.size

    return w / total


def tail_cvar_loss(
    scenario_returns: np.ndarray,
    weights: np.ndarray,
    alpha: float = 0.95,
    min_tail_count: int = 5,
) -> float:
    """
    Positive loss CVaR objective. Works even with few scenarios by enforcing
    at least min_tail_count observations in the loss tail.
    """
    r = np.asarray(scenario_returns, dtype=float)
    w = np.asarray(weights, dtype=float)

    portfolio_returns = r @ w
    losses = -portfolio_returns
    losses = losses[np.isfinite(losses)]

    if losses.size == 0:
        return 1e6

    tail_count = max(int(np.ceil((1.0 - alpha) * losses.size)), int(min_tail_count))
    tail_count = min(tail_count, losses.size)

    tail = np.partition(losses, -tail_count)[-tail_count:]
    return float(np.mean(tail))
```

---

## 2. Replace `scripts/optimizers/base.py`

This fixes the duplicated Marchenko-Pastur bug by using the shared covariance estimator, makes optimization robust to failures, and guarantees final weights sum to one.

```python
from __future__ import annotations

import numpy as np

from beartype import beartype
from typing import Any, Callable
from scipy.optimize import minimize

from scripts.calculations.covariance import clean_covariance, nearest_psd
from scripts.calculations.portfolio_utils import normalize_long_only


class BaseCVaR:
    def __init__(
        self,
        config: dict[str, Any],
        market_cap: np.ndarray,
        historical_returns: np.ndarray,
        pred_returns: np.ndarray,
        w_previous: np.ndarray,
        bound: tuple = (0, None),
    ):
        self.config = config
        self.method = self.config.get("method", "SLSQP")
        self.min_weight = float(self.config.get("min_weight", 0.01))
        self.max_weight = float(self.config.get("max_weight", 0.1))
        self.is_all_methods = bool(self.config.get("is_all_methods", True))
        self.max_iter = self.config.get("max_iter", None)

        self.turnover_penalty = float(self.config.get("turnover_penalty", 1.0))
        self.penalty_type = self.config.get("penalty_type", "L1")

        self.market_cap = np.asarray(market_cap, dtype=float) if market_cap is not None else None
        self.historical_returns = np.asarray(historical_returns, dtype=float)
        self.pred_returns = np.asarray(pred_returns, dtype=float)
        self.w_previous = np.asarray(w_previous, dtype=float)

        self.all_methods: list[str] = []
        self.optimal_values: list[float] = []
        self.optimal_weights: list[np.ndarray] = []

    @beartype
    def remove_noise_cov_matrix(self) -> np.ndarray:
        """
        Replaces the old inline MP implementation.

        Config options:
            cov_method: "oas" | "ledoit_wolf" | "mp" | "graphical_lasso" | "sample"
            detone_n: number of market components to remove, default 0 for optimizers
        """
        cov_method = self.config.get("cov_method", "oas")
        detone_n = int(self.config.get("detone_n", 0))
        return nearest_psd(clean_covariance(self.historical_returns, method=cov_method, detone_n=detone_n))

    def calculate_turnover_penalty(self, weights: np.ndarray) -> float:
        if self.penalty_type == "L1":
            turnover = np.sum(np.abs(weights - self.w_previous))
        elif self.penalty_type == "L2":
            turnover = np.sum((weights - self.w_previous) ** 2)
        else:
            raise ValueError(f"Unknown penalty type {self.penalty_type}")
        return float(self.turnover_penalty * turnover)

    def function_to_optimize(self, *args, **kwargs) -> Callable:
        raise NotImplementedError("Child class must implement function_to_optimize")

    def _postprocess_weights(self, weights: np.ndarray) -> np.ndarray:
        return normalize_long_only(
            weights,
            min_weight=self.min_weight,
            max_weight=self.max_weight,
            fallback_n=weights.size,
        )

    @beartype
    def optimizer(self, args: tuple, count_etf: int, method_type: str) -> None:
        effective_max = max(self.max_weight, 1.0 / count_etf)
        bounds = [(0.0, effective_max)] * count_etf
        x0 = np.ones(count_etf, dtype=float) / count_etf

        if self.method == "COBYLA":
            constraints = [
                {"type": "ineq", "fun": lambda x: 1.0 - np.sum(x)},
                {"type": "ineq", "fun": lambda x: np.sum(x) - 0.999},
            ]
        else:
            constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1.0}]

        objective = lambda w: self.function_to_optimize(w, *args, method_type)

        options = {}
        if self.max_iter is not None:
            options["maxiter"] = int(self.max_iter)

        result = minimize(
            fun=objective,
            x0=x0,
            method=self.method,
            bounds=bounds if self.method != "COBYLA" else None,
            constraints=constraints,
            options=options or None,
        )

        if not result.success or not np.all(np.isfinite(result.x)):
            weights = x0
            value = objective(weights)
        else:
            weights = result.x
            value = float(result.fun)

        weights = self._postprocess_weights(weights)

        self.optimal_values.append(float(value))
        self.optimal_weights.append(weights)

    @beartype
    def get_results(self) -> tuple[list[float], list[np.ndarray]]:
        return self.optimal_values, self.optimal_weights

    @beartype
    def get_method_names(self) -> list[str]:
        return self.all_methods
```

---

## 3. Patch `scripts/optimizers/bayessian.py`

Replace only `function_to_optimize` in `BayessianCVaR`.

```python
from scripts.calculations.portfolio_utils import tail_cvar_loss
```

```python
@beartype
def function_to_optimize(
    self,
    weights: np.ndarray,
    pred_returns: np.ndarray,
    w_previous: np.ndarray,
    method_type: Literal["cvar_returns", "expected_returns", "deviation_from_target"],
) -> float:

    weights = np.asarray(weights, dtype=float)
    pred_returns = np.asarray(pred_returns, dtype=float)
    w_previous = np.asarray(w_previous, dtype=float)

    penalty = self.calculate_turnover_penalty(weights)

    if method_type == "cvar_returns":
        alpha = float(self.config.get("confidence_level", 0.95))
        min_tail_count = int(self.config.get("min_tail_count", 5))
        optimization_func = tail_cvar_loss(
            pred_returns,
            weights,
            alpha=alpha,
            min_tail_count=min_tail_count,
        )

    elif method_type == "expected_returns":
        expected_returns = np.nanmean(pred_returns, axis=0)
        risk_aversion = float(self.config.get("risk_aversion", 0.0))
        variance_penalty = float(weights.T @ self.cov_matrix @ weights)
        optimization_func = -float(expected_returns @ weights) + risk_aversion * variance_penalty

    elif method_type == "deviation_from_target":
        # Old version had negative sign and maximized deviation.
        # This now minimizes tracking error to the previous portfolio.
        delta = weights - w_previous
        tracking_error = float(delta.T @ self.cov_matrix @ delta)
        optimization_func = tracking_error

    else:
        raise ValueError(f"Unknown type: {method_type}")

    return float(optimization_func + penalty)
```

Recommended YAML change:

```yaml
optimizer:
  - type: bayessian
    method: SLSQP
    min_weight: 0.005
    max_weight: 0.10
    task_type: "cvar_returns"
    confidence_level: 0.95
    min_tail_count: 10
    turnover_penalty: 0.25
    penalty_type: L1
    cov_method: oas
    detone_n: 0
    is_all_methods: false
```

---

## 4. Replace `BaseModel.evaluate_metrics` in `scripts/models/base.py`

This removes the incorrect KDE-as-cross-section logic, avoids fake WAIC, fixes R², and evaluates the predicted scenario distribution against realized cross-sectional returns.

```python
from scipy.stats import norm, spearmanr
```

```python
@beartype
def evaluate_metrics(
    self,
    pred_returns: np.ndarray,
    future_returns: np.ndarray | None = None,
) -> tuple:
    """
    pred_returns: scenario matrix, shape S x N.
    future_returns: realized matrix, shape T x N. If omitted, uses self.future_returns.

    The metric compares scenario distribution per asset with realized next-month return per asset.
    WAIC is returned as np.nan because the current model does not store a proper
    posterior log-likelihood matrix. Reporting fake WAIC is worse than reporting no WAIC.
    """
    pred = np.asarray(pred_returns, dtype=float)
    real = np.asarray(self.future_returns if future_returns is None else future_returns, dtype=float)

    if pred.ndim != 2 or real.ndim != 2:
        raise ValueError(f"pred and real must be 2D, got {pred.shape=} and {real.shape=}")

    n_assets = min(pred.shape[1], real.shape[1])
    pred = pred[:, :n_assets]
    real = real[:, :n_assets]

    real_target = np.nansum(real, axis=0)
    pred_mean = np.nanmean(pred, axis=0)
    pred_std = np.nanstd(pred, axis=0, ddof=1)
    pred_std = np.where(pred_std < 1e-8, 1e-8, pred_std)

    lower = np.nanpercentile(pred, 2.5, axis=0)
    upper = np.nanpercentile(pred, 97.5, axis=0)

    residual = real_target - pred_mean
    mae = float(np.nanmean(np.abs(residual)))
    rmse = float(np.sqrt(np.nanmean(residual ** 2)))
    coverage = float(np.nanmean((real_target >= lower) & (real_target <= upper)))
    sign_precision = float(np.nanmean(np.sign(pred_mean) == np.sign(real_target)))
    avg_true_positives = float(np.nanmean(real_target > 0))

    rho, _ = spearmanr(real_target, pred_mean)
    rank_ic = float(0.0 if np.isnan(rho) else rho)

    ss_res = float(np.nansum((real_target - pred_mean) ** 2))
    ss_tot = float(np.nansum((real_target - np.nanmean(real_target)) ** 2))
    r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else np.nan

    log_score = float(np.nanmean(norm.logpdf(real_target, loc=pred_mean, scale=pred_std)))

    waic = np.nan
    avg_predicted_mean = float(np.nanmean(pred_mean))
    avg_true_mean = float(np.nanmean(real_target))
    mean_deviation = avg_predicted_mean - avg_true_mean

    return (
        log_score,
        waic,
        rank_ic,
        mae,
        coverage,
        sign_precision,
        avg_true_positives,
        r_squared,
        rmse,
        avg_predicted_mean,
        avg_true_mean,
        mean_deviation,
    )
```

Then in `runs/run.py` and `combined_run.py`, call:

```python
model_metrics = modelClass.evaluate_metrics(pred_returns, future_returns.values)
```

---

## 5. Patch `scripts/models/bayessian.py`

Main fixes:
- use local RNG, not global `np.random.seed`;
- normalize `t-student` / `t_student`;
- use `n_samples` as scenario count;
- use cleaned covariance;
- avoid future data controlling the number of scenarios.

```python
from __future__ import annotations

import numpy as np
import pandas as pd

from typing import Any
from beartype import beartype

from scripts.calculations.covariance import clean_covariance, nearest_psd
from scripts.calculations.matrix import calculate_multivariate_shock_matrix
from scripts.models.base import BaseModel


class BayessianModel(BaseModel):
    def __init__(
        self,
        config: dict[str, Any],
        etfs_list: list,
        historical_returns: pd.DataFrame,
        historical_ewma_returns: pd.DataFrame,
        future_returns: np.ndarray,
    ):
        self.config = config
        super().__init__(self.config, etfs_list, historical_returns, historical_ewma_returns, future_returns)

        self.meanReturnType = self.config.get("mean_return", "sampled")
        self.distType = self.config.get("distribution", "t_student").replace("-", "_")
        self.posteriorType = self.config.get("posterior", "horseshoe")
        self.n_samples = int(self.config.get("n_samples", 2000))
        self.seed = int(self.config.get("seed", 42))
        self.rng = np.random.default_rng(self.seed)

        self.count_etf = len(self.etfs_list)

    def _returns_matrix(self) -> np.ndarray:
        returns = self.returns.drop(columns=["Date"], errors="ignore")
        return returns[self.etfs_list].astype(float).to_numpy()

    @beartype
    def calculate_prior_mean_variance(self) -> tuple[float, float]:
        returns = self.ewma_returns.drop(columns=["Date"], errors="ignore")
        returns = returns[self.etfs_list].astype(float)
        prior_mean = float(np.nanmean(returns.to_numpy()))
        prior_var = float(np.nanmean(np.nanvar(returns.to_numpy(), axis=0, ddof=1)))
        return prior_mean, max(prior_var, 1e-10)

    def get_returns_with_distribution(self, returns_etf: pd.Series, mu_i: float) -> np.ndarray:
        x = np.asarray(returns_etf, dtype=float)
        x = x[np.isfinite(x)]

        sigma_i = float(np.std(x, ddof=1)) if x.size > 2 else 1e-4
        sigma_i = max(sigma_i, 1e-8)

        if self.meanReturnType == "historical":
            return x

        if self.distType == "normal":
            return self.rng.normal(mu_i, sigma_i, self.n_samples)

        if self.distType == "t_student":
            df = int(self.config.get("df", 5))
            return mu_i + sigma_i * self.rng.standard_t(df, size=self.n_samples)

        if self.distType == "laplace":
            return self.rng.laplace(mu_i, sigma_i / np.sqrt(2.0), self.n_samples)

        raise ValueError(f"Unsupported distribution: {self.distType}")

    def sample_posterior(self, returns: np.ndarray, prior_mean: float, prior_var: float) -> np.ndarray:
        x = np.asarray(returns, dtype=float)
        x = x[np.isfinite(x)]

        if x.size < 3:
            return np.full(self.n_samples, prior_mean)

        sample_mean = float(np.mean(x))
        sample_var = max(float(np.var(x, ddof=1)), 1e-10)

        if self.posteriorType == "gaussian":
            tau_post = 1.0 / (1.0 / prior_var + x.size / sample_var)
            mu_post = tau_post * (prior_mean / prior_var + x.size * sample_mean / sample_var)
            return self.rng.normal(mu_post, np.sqrt(tau_post), size=self.n_samples)

        if self.posteriorType == "empirical_bayes":
            shrink = prior_var / (prior_var + sample_var / max(x.size, 1))
            mu_post = shrink * sample_mean + (1.0 - shrink) * prior_mean
            sigma_post = np.sqrt(sample_var / max(x.size, 1))
            return self.rng.normal(mu_post, sigma_post, size=self.n_samples)

        if self.posteriorType == "t_student":
            df = int(self.config.get("df", 5))
            scale = np.sqrt(sample_var * (df - 2.0) / df)
            return sample_mean + scale * self.rng.standard_t(df, size=self.n_samples)

        if self.posteriorType == "horseshoe":
            tau = np.clip(abs(self.rng.standard_cauchy()), 0.01, 10.0)
            lam = np.clip(abs(self.rng.standard_cauchy()), 0.01, 10.0)
            shrink = tau * lam / (1.0 + tau * lam)
            mu_post = shrink * sample_mean + (1.0 - shrink) * prior_mean
            sigma_post = np.sqrt(sample_var) * shrink + 1e-6
            return self.rng.normal(mu_post, sigma_post, size=self.n_samples)

        if self.posteriorType == "spike_slab":
            p = float(self.config.get("spike_slab_p", 0.5))
            active = self.rng.binomial(1, p, size=self.n_samples)
            draws = self.rng.normal(sample_mean, np.sqrt(sample_var), size=self.n_samples)
            return active * draws

        raise ValueError(f"Unknown posterior type: {self.posteriorType}")

    @beartype
    def get_posterior_draws(self) -> np.ndarray:
        prior_mean, prior_var = self.calculate_prior_mean_variance()
        mu_i = self.rng.normal(prior_mean, np.sqrt(prior_var), self.count_etf)

        posterior_draws = np.empty((self.n_samples, self.count_etf), dtype=float)

        returns_df = self.returns.drop(columns=["Date"], errors="ignore")
        for ind, etf in enumerate(self.etfs_list):
            simulated_or_hist = self.get_returns_with_distribution(returns_df[etf], mu_i[ind])
            posterior_draws[:, ind] = self.sample_posterior(simulated_or_hist, prior_mean, prior_var)

        return posterior_draws

    @beartype
    def get_multivariate_shock(self) -> np.ndarray:
        x = self._returns_matrix()
        cov_method = self.config.get("cov_method", "oas")
        detone_n = int(self.config.get("detone_n", 0))
        cov = nearest_psd(clean_covariance(x, method=cov_method, detone_n=detone_n))
        return calculate_multivariate_shock_matrix(cov, self.n_samples)

    @beartype
    def prediction(self) -> np.ndarray:
        posterior_draws = self.get_posterior_draws()
        shocks = self.get_multivariate_shock()
        return posterior_draws + shocks
```

Recommended model YAML:

```yaml
model:
  - type: bayessian
  - mean_return: sampled
  - distribution: t_student
  - posterior: empirical_bayes
  - n_samples: 2000
  - df: 5
  - cov_method: oas
  - detone_n: 0
  - seed: 42
```

---

## 6. Patch `scripts/strategies/base.py`

```python
from __future__ import annotations

import numpy as np

from typing import Any
from beartype import beartype

from scripts.calculations.portfolio_utils import normalize_long_only


class BaseStrategy:
    def __init__(
        self,
        config: dict[str, Any],
        historical_returns: np.ndarray,
        pred_returns: np.ndarray,
        market_cap: np.ndarray,
        w_previous: np.ndarray,
    ):
        if "strategy" in config and isinstance(config["strategy"], list):
            strategy_cfg = {k: v for pair in config["strategy"] for k, v in pair.items()}
        else:
            strategy_cfg = config

        self.config = strategy_cfg
        self.min_weight = float(self.config.get("min_weight", 0.0))
        self.max_weight = float(self.config.get("max_weight", 0.1))

        self.historical_returns = np.asarray(historical_returns, dtype=float)
        self.pred_returns = np.asarray(pred_returns, dtype=float) if pred_returns is not None else None
        self.market_cap = np.asarray(market_cap, dtype=float) if market_cap is not None else None
        self.w_previous = np.asarray(w_previous, dtype=float)

        self.weights = np.zeros(self.historical_returns.shape[1], dtype=float)

    @beartype
    def clip_weights(self):
        self.weights = normalize_long_only(
            self.weights,
            min_weight=self.min_weight,
            max_weight=self.max_weight,
            fallback_n=self.historical_returns.shape[1],
        )

    @beartype
    def get_weights(self) -> np.ndarray:
        self.clip_weights()
        return self.weights
```

---

## 7. Fix current benchmark strategies

Patch `scripts/strategies/benchmarks.py`.

```python
from __future__ import annotations

import numpy as np
from scripts.strategies.base import BaseStrategy


class EqualWeightStrategy(BaseStrategy):
    def __init__(self, config, historical_returns, pred_returns, market_cap, w_previous):
        super().__init__(config, historical_returns, pred_returns, market_cap, w_previous)
        n_assets = self.historical_returns.shape[1]
        self.weights = np.ones(n_assets) / n_assets


class InverseVolatilityStrategy(BaseStrategy):
    def __init__(self, config, historical_returns, pred_returns, market_cap, w_previous):
        super().__init__(config, historical_returns, pred_returns, market_cap, w_previous)

        valid_counts = np.sum(np.isfinite(self.historical_returns), axis=0)
        vol = np.nanstd(self.historical_returns, axis=0, ddof=1)
        vol = np.where((valid_counts < 20) | (vol <= 1e-10), np.inf, vol)

        inv_vol = 1.0 / vol
        inv_vol = np.nan_to_num(inv_vol, nan=0.0, posinf=0.0, neginf=0.0)

        if inv_vol.sum() <= 1e-12:
            self.weights = np.ones_like(inv_vol) / len(inv_vol)
        else:
            self.weights = inv_vol / inv_vol.sum()


class LiquidityWeightedStrategy(BaseStrategy):
    def __init__(self, config, historical_returns, pred_returns, market_cap, w_previous):
        super().__init__(config, historical_returns, pred_returns, market_cap, w_previous)

        n_assets = self.historical_returns.shape[1]
        if self.market_cap is None:
            self.weights = np.ones(n_assets) / n_assets
            return

        liquidity = np.nanmean(np.maximum(self.market_cap, 0.0), axis=0)
        liquidity = np.nan_to_num(liquidity, nan=0.0, posinf=0.0, neginf=0.0)

        if liquidity.sum() <= 1e-12:
            self.weights = np.ones(n_assets) / n_assets
        else:
            self.weights = liquidity / liquidity.sum()
```

---

## 8. Add new strategies: `scripts/strategies/momentum_quality.py`

These are useful novelties for your framework because they combine signal, risk, liquidity, and turnover control.

```python
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from scripts.calculations.covariance import clean_covariance, nearest_psd
from scripts.strategies.base import BaseStrategy


class MomentumVolatilityStrategy(BaseStrategy):
    """
    Long-only momentum / volatility strategy.

    Score_i = recent momentum_i / realized volatility_i.
    Keeps top_k assets and normalizes with max-weight cap.
    """

    def __init__(self, config, historical_returns, pred_returns, market_cap, w_previous):
        super().__init__(config, historical_returns, pred_returns, market_cap, w_previous)
        lookback = int(self.config.get("momentum_lookback", 63))
        top_k = int(self.config.get("top_k", min(30, self.historical_returns.shape[1])))

        x = self.historical_returns[-lookback:]
        momentum = np.nansum(x, axis=0)
        vol = np.nanstd(x, axis=0, ddof=1)
        score = momentum / np.maximum(vol, 1e-8)
        score = np.nan_to_num(score, nan=-np.inf, neginf=-np.inf, posinf=0.0)

        k = max(1, min(top_k, score.size))
        selected = np.argsort(score)[-k:]

        raw = np.zeros(score.size)
        positive_score = np.maximum(score[selected], 0.0)
        if positive_score.sum() <= 1e-12:
            raw[selected] = 1.0 / k
        else:
            raw[selected] = positive_score / positive_score.sum()

        self.weights = raw


class LiquidityMomentumStrategy(BaseStrategy):
    """
    Momentum strategy with liquidity awareness.

    Good when you want to avoid tiny or illiquid ETFs dominating due to noisy returns.
    """

    def __init__(self, config, historical_returns, pred_returns, market_cap, w_previous):
        super().__init__(config, historical_returns, pred_returns, market_cap, w_previous)

        lookback = int(self.config.get("momentum_lookback", 63))
        top_k = int(self.config.get("top_k", min(40, self.historical_returns.shape[1])))
        liquidity_power = float(self.config.get("liquidity_power", 0.25))

        x = self.historical_returns[-lookback:]
        momentum = np.nansum(x, axis=0)
        vol = np.nanstd(x, axis=0, ddof=1)
        score = momentum / np.maximum(vol, 1e-8)

        if self.market_cap is not None:
            liq = np.nanmean(np.maximum(self.market_cap, 0.0), axis=0)
            liq = np.nan_to_num(liq, nan=0.0)
            positive_liq = liq[liq > 0]
            if positive_liq.size > 0:
                liq_score = (liq / np.nanmedian(positive_liq)) ** liquidity_power
                score = score * liq_score

        score = np.nan_to_num(score, nan=-np.inf, neginf=-np.inf, posinf=0.0)

        k = max(1, min(top_k, score.size))
        selected = np.argsort(score)[-k:]

        raw = np.zeros(score.size)
        s = np.maximum(score[selected], 0.0)
        raw[selected] = 1.0 / k if s.sum() <= 1e-12 else s / s.sum()
        self.weights = raw


class MinimumCorrelationStrategy(BaseStrategy):
    """
    Long-only minimum correlation strategy.

    Minimizes w.T @ Corr @ w, not covariance variance. Good as a diversification benchmark.
    """

    def __init__(self, config, historical_returns, pred_returns, market_cap, w_previous):
        super().__init__(config, historical_returns, pred_returns, market_cap, w_previous)

        x = np.nan_to_num(self.historical_returns, nan=0.0)
        cov = nearest_psd(clean_covariance(x, method=self.config.get("cov_method", "oas"), detone_n=0))
        std = np.sqrt(np.maximum(np.diag(cov), 1e-12))
        corr = cov / np.outer(std, std)
        corr = nearest_psd(corr)

        n = x.shape[1]
        x0 = np.ones(n) / n
        max_w = max(self.max_weight, 1.0 / n)
        bounds = [(0.0, max_w)] * n
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        def objective(w):
            turnover = np.sum(np.abs(w - self.w_previous[:n])) if self.w_previous.size == n else 0.0
            turnover_penalty = float(self.config.get("turnover_penalty", 0.0))
            return float(w.T @ corr @ w + turnover_penalty * turnover)

        res = minimize(objective, x0=x0, method="SLSQP", bounds=bounds, constraints=cons)
        self.weights = res.x if res.success else x0
```

Patch `scripts/strategies/registry.py`:

```python
from scripts.strategies.momentum_quality import (
    MomentumVolatilityStrategy,
    LiquidityMomentumStrategy,
    MinimumCorrelationStrategy,
)

STRATEGY_REGISTRY: dict[str, Type[BaseStrategy]] = {
    "simple_long": SimpleLongStrategy,
    "risk_parity": RiskParityStrategy,
    "equal_weight": EqualWeightStrategy,
    "inverse_volatility": InverseVolatilityStrategy,
    "liquidity_weighted": LiquidityWeightedStrategy,
    "momentum_volatility": MomentumVolatilityStrategy,
    "liquidity_momentum": LiquidityMomentumStrategy,
    "minimum_correlation": MinimumCorrelationStrategy,
}
```

YAML example:

```yaml
strategy:
  - type: liquidity_momentum
  - min_weight: 0.005
  - max_weight: 0.08
  - momentum_lookback: 63
  - top_k: 40
  - liquidity_power: 0.25
```

---

## 9. Important bug in strategy runs: pass returns, not prices

In `runs/strategy_run.py` and `runs/combined_strategy_run.py`, your strategy constructor currently receives `historical_prices`. That means inverse-volatility, risk-parity, and similar strategies are using price levels as if they were returns.

Replace this:

```python
weights = strategy_class(
    config,
    historical_prices,
    pred_returns,
    market_cap,
    w_previous
).get_weights()
```

with this:

```python
weights = strategy_class(
    config,
    historical_returns,
    pred_returns,
    market_cap,
    w_previous[mask].values if hasattr(w_previous, "values") else w_previous,
).get_weights()
```

---

## 10. Replace hard-coded Black-Litterman views

Replace `_construct_investor_views` in `scripts/optimizers/black_litterman.py` with this signal-derived version:

```python
@beartype
def _construct_investor_views(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Builds one relative view:
        recent top momentum basket should outperform recent bottom momentum basket.

    This avoids positional ETF bugs such as row[17] / row[62].
    """
    n = self.count_etf
    x = np.asarray(self.historical_returns, dtype=float)

    lookback = int(self.config.get("bl_view_lookback", min(63, x.shape[0])))
    k = int(self.config.get("bl_view_k", max(1, min(10, n // 10))))

    recent = x[-lookback:]
    momentum = np.nansum(recent, axis=0)
    momentum = np.nan_to_num(momentum, nan=0.0)

    order = np.argsort(momentum)
    bottom = order[:k]
    top = order[-k:]

    P = np.zeros((1, n), dtype=float)
    P[0, top] = 1.0 / k
    P[0, bottom] = -1.0 / k

    observed_spread = float(np.nanmean(momentum[top]) - np.nanmean(momentum[bottom]))
    confidence_scale = float(self.config.get("bl_view_scale", 0.25))
    Q = np.array([confidence_scale * observed_spread], dtype=float)

    omega = self.tau * (P @ self.cov_matrix @ P.T)
    omega = omega + 1e-8 * np.eye(omega.shape[0])

    return P, Q, omega
```

Also replace inverse calls with pseudo-inverse for stability:

```python
omega_inv = np.linalg.pinv(self.omega)
tau_sigma_inv = np.linalg.pinv(self.tau * self.cov_matrix)
posterior_cov_inv = np.linalg.pinv(
    tau_sigma_inv + self.matrix_p.T @ omega_inv @ self.matrix_p
)
```

---

## 11. Replace JAX sigmoid in run files

Remove:

```python
import jax
```

Add:

```python
from scripts.calculations.portfolio_utils import sigmoid_np
```

Replace:

```python
alpha = jax.nn.sigmoid(exp_sum)
```

with:

```python
alpha = sigmoid_np(exp_sum)
```

---

## 12. Lookback window fix

In `scripts/dataloader/filter.py`, make the lookback configurable:

```python
@beartype
def find_start_time(
    dates: pd.Series,
    day: pd.Timestamp,
    lookback_years: int = 3,
) -> pd.Timestamp:
    target = day - pd.DateOffset(years=lookback_years)
    idx = (dates - target).abs().idxmin()
    return dates.loc[idx]
```

Then in run files:

```python
lookback_years = int(config.get("lookback_years", 3))
startTime = find_start_time(first_business_days, day, lookback_years=lookback_years)
```

YAML:

```yaml
lookback_years: 3
```

---

## 13. Better execution cost settings

Current `c_bps: 1e-2` means 1% per unit turnover. For ETF monthly rebalancing, test multiple cost regimes instead of one hard-coded high cost:

```yaml
execution:
  - type: base_exec
    return_type: "log-returns"
    c_bps: 0.0010
```

Run sensitivity:
- `0.0001` low cost
- `0.0010` medium cost
- `0.0025` high ETF friction
- `0.0100` stress scenario only

---

## 14. Research novelty roadmap

After the above fixes, add these strategies/models:

1. **Ensemble optimizer**  
   Combine CVaR, min-variance, momentum-volatility, and minimum-correlation weights:
   `w = lambda_1*w_CVaR + lambda_2*w_MV + lambda_3*w_MOM + lambda_4*w_DECORR`.
   Choose lambdas using previous 12 months only, not the current prediction.

2. **Regime-aware allocation**  
   Estimate market regime from realized volatility and cross-sectional correlation:
   high-vol/high-corr -> minimum variance or CVaR; low-vol/low-corr -> momentum/expected return; drawdown regime -> reduce turnover and cap risky ETFs.

3. **Hierarchical Risk Parity**  
   You already have clustering logic in covariance cleaning. Use it for HRP allocation.

4. **Robust expected return model**  
   Replace raw posterior mean with shrinkage:
   `mu_i = lambda*mu_momentum_i + (1-lambda)*mu_EWMA_i`.
   Then shrink all signals toward zero based on prediction uncertainty.

5. **Purged validation for method selection**  
   Do not choose optimizer/alpha on the same prediction matrix used to optimize weights.

6. **Explicit cash sleeve**  
   Either renormalize to 1 or create a cash asset. Do not let cash appear accidentally from broken clipping.

---

## 15. Minimal smoke tests

Add `tests/test_core.py`.

```python
import numpy as np

from scripts.calculations.portfolio_utils import normalize_long_only, tail_cvar_loss
from scripts.calculations.covariance import clean_covariance


def test_weights_sum_to_one():
    w = np.array([0.001, 0.2, 0.3, 0.5])
    out = normalize_long_only(w, min_weight=0.01, max_weight=0.6)
    assert np.isclose(out.sum(), 1.0)
    assert np.all(out >= 0)


def test_covariance_psd():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(252, 50))
    cov = clean_covariance(x, method="oas", detone_n=0)
    eig = np.linalg.eigvalsh(cov)
    assert eig.min() > -1e-8


def test_cvar_loss_positive_for_bad_tail():
    r = np.array([[0.01], [0.02], [-0.10], [-0.05], [0.03]])
    w = np.array([1.0])
    cvar = tail_cvar_loss(r, w, alpha=0.8, min_tail_count=2)
    assert cvar > 0
```
