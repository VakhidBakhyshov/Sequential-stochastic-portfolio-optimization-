from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from typing import Any

try:
    from beartype import beartype
except ImportError:  # pragma: no cover
    def beartype(obj):
        return obj


EPS = 1e-12


class BaseModel:
    """Base model with scenario-aware posterior-predictive diagnostics.

    The original project evaluated Monte Carlo scenario rows as though they were an
    ordered forecast path. That is not statistically meaningful. This implementation
    automatically recognizes a scenario fan and compares its cross-sectional
    distribution with the realized holding-period return vector.
    """

    def __init__(
        self,
        config: dict[str, Any],
        etfs_list: list,
        historical_returns: pd.DataFrame,
        historical_ewma_returns: pd.DataFrame,
        future_returns: np.ndarray,
    ):
        self.config = config
        self.etfs_list = list(etfs_list)
        self.returns = historical_returns
        self.ewma_returns = historical_ewma_returns
        self.future_returns = np.asarray(future_returns, dtype=float)

    @beartype
    def prediction(self):
        raise NotImplementedError("Child class must implement prediction()")

    def _aggregate_realized_horizon(self) -> np.ndarray:
        y = np.asarray(self.future_returns, dtype=float)
        if y.ndim == 1:
            y = y.reshape(1, -1)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        return_type = str(self.config.get("return_type", self.config.get("scenario_return_type", "log-returns")))
        if return_type == "log-returns":
            return y.sum(axis=0)
        if return_type == "returns":
            return np.prod(1.0 + np.clip(y, -0.999999, None), axis=0) - 1.0
        raise ValueError(f"Unknown return_type={return_type!r}")

    @beartype
    def evaluate_metrics(self, pred_returns: np.ndarray) -> tuple:
        """Return the historical 12-field metric tuple, but evaluate scenario fans correctly.

        Returned fields remain compatible with ``BaseResults``:
        log_score, waic_proxy, rank_ic, mae, coverage_95, sign_precision,
        avg_true_positives, r_squared, rmse, avg_predicted_mean,
        avg_true_mean, mean_deviation.

        ``waic_proxy`` is a predictive deviance proxy, not full Bayesian WAIC, because
        the pipeline stores posterior-predictive scenarios rather than pointwise
        log-likelihood draws. The report labels this limitation explicitly.
        """
        scenarios = np.asarray(pred_returns, dtype=float)
        if scenarios.ndim == 1:
            scenarios = scenarios.reshape(1, -1)
        scenarios = np.nan_to_num(scenarios, nan=0.0, posinf=0.0, neginf=0.0)
        if scenarios.ndim != 2 or scenarios.shape[1] == 0:
            return (np.nan,) * 12

        realized = self._aggregate_realized_horizon()
        n_assets = min(scenarios.shape[1], realized.shape[0])
        scenarios = scenarios[:, :n_assets]
        realized = realized[:n_assets]

        pred_mean = scenarios.mean(axis=0)
        pred_sd = scenarios.std(axis=0, ddof=1) if len(scenarios) > 1 else np.full(n_assets, EPS)
        pred_sd = np.where(np.isfinite(pred_sd) & (pred_sd > EPS), pred_sd, EPS)

        log_density = norm.logpdf(realized, loc=pred_mean, scale=pred_sd)
        log_score = float(np.nanmean(log_density))
        # Predictive deviance proxy. Kept in the old 'waic' column for schema compatibility.
        waic_proxy = float(-2.0 * np.nansum(log_density))

        rho, _ = spearmanr(realized, pred_mean) if n_assets > 1 else (np.nan, np.nan)
        rank_ic = float(rho) if np.isfinite(rho) else 0.0
        errors = pred_mean - realized
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors ** 2)))

        lower = np.quantile(scenarios, 0.025, axis=0)
        upper = np.quantile(scenarios, 0.975, axis=0)
        coverage = float(np.mean((realized >= lower) & (realized <= upper)))
        sign_precision = float(np.mean(np.sign(pred_mean) == np.sign(realized)))
        avg_true_positives = float(np.mean(realized > 0.0))

        true_mean = float(np.mean(realized))
        ss_res = float(np.sum(errors ** 2))
        ss_tot = float(np.sum((realized - true_mean) ** 2))
        r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > EPS else np.nan

        avg_predicted_mean = float(np.mean(pred_mean))
        avg_true_mean = true_mean
        mean_deviation = avg_predicted_mean - avg_true_mean
        self.mean_deviation = mean_deviation

        return (
            log_score,
            waic_proxy,
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
