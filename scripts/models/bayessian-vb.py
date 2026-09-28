from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    from beartype import beartype
except ImportError:  # pragma: no cover
    def beartype(obj):
        return obj

from scripts.calculations.covariance_cleaning import estimate_covariance
from scripts.calculations.matrix import make_positive_semifinite_matrix
from scripts.models.base import BaseModel

EPS = 1e-12


class BayessianModel(BaseModel):
    """Bayesian-shrinkage posterior-predictive scenario model.

    The class name is intentionally kept as ``BayessianModel`` for registry and
    configuration compatibility with the original project.

    Important unit convention:
    - historical inputs are daily returns;
    - ``prediction()`` emits holding-period scenarios for ``horizon`` days;
    - if ``scenario_return_type`` is ``log-returns``, daily simulated log returns
      are summed; if it is ``returns``, daily simple returns are compounded.
    """

    def __init__(
        self,
        config: dict[str, Any],
        etfs_list: list,
        historical_returns: pd.DataFrame,
        historical_ewma_returns: pd.DataFrame,
        future_returns: np.ndarray,
    ):
        self.config = dict(config)
        super().__init__(self.config, etfs_list, historical_returns, historical_ewma_returns, future_returns)

        self.meanReturnType = self.config.get("mean_return", "sampled")
        self.distType = self.config.get("distribution", "t_student")
        self.posteriorType = self.config.get("posterior", "horseshoe")
        self.matrixType = self.config.get("matrix_type", "fast")
        self.is_sampled_mean = bool(self.config.get("is_sampled_mean", False))
        self.random_state = int(self.config.get("random_state", 42))
        self.rng = np.random.default_rng(self.random_state)

        self.count_etf = len(self.etfs_list)
        self.len_returns = len(np.asarray(future_returns))
        
        self.horizon = int(self.config.get("horizon", self.len_returns or 21))
        self.n_samples = int(self.config.get("n_samples", 1000))
        self.n_scenarios = int(self.config.get("n_scenarios", self.n_samples))
        self.scenario_return_type = str(self.config.get("scenario_return_type", self.config.get("return_type", "log-returns")))

        self.last_covariance_daily: np.ndarray | None = None
        self.last_covariance_horizon: np.ndarray | None = None
        self.last_correlation: np.ndarray | None = None
        self.last_posterior_mean_daily: np.ndarray | None = None
        self.last_posterior_mean_horizon: np.ndarray | None = None
        self.last_daily_scenario_paths: np.ndarray | None = None

    def _returns_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'Date' in df.columns:
            x = df.copy().drop(columns=["Date"], errors="ignore")
        x = x.reindex(columns=self.etfs_list)
        return x.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    @beartype
    def calculate_prior_mean_variance(self) -> tuple[float, float]:
        returns = self._returns_frame(self.ewma_returns)
        prior_mean = float(returns.mean().mean()) if not returns.empty else 0.0
        prior_var = float((returns.std(ddof=1) ** 2).mean()) if len(returns) > 1 else 1e-6
        if not np.isfinite(prior_var) or prior_var <= EPS:
            prior_var = 1e-6
        return prior_mean, prior_var

    def get_returns_with_distribution(self, returns_etf: pd.Series, mu_i: float) -> tuple[float, float, np.ndarray]:
        x = pd.to_numeric(returns_etf, errors="coerce").dropna().to_numpy(dtype=float)
        sigma = float(np.std(x, ddof=1)) if x.size > 1 else 1e-4
        sigma = max(sigma, 1e-4)
        size = max(2, self.n_samples)
        if self.meanReturnType == "historical":
            draws = x if x.size else np.array([0.0])
        elif self.distType == "normal":
            draws = self.rng.normal(mu_i, sigma, size=size)
        elif self.distType in {"t_student", "t-student"}:
            df = max(3, int(self.config.get("df", 5)))
            # scale so the t draw has approximately sigma standard deviation
            scale = sigma * np.sqrt((df - 2.0) / df)
            draws = mu_i + scale * self.rng.standard_t(df, size=size)
        elif self.distType == "laplace":
            draws = self.rng.laplace(mu_i, sigma / np.sqrt(2.0), size=size)
        else:
            raise ValueError(f"Unsupported distribution: {self.distType}")
        return sigma ** 2, float(np.mean(draws)), np.asarray(draws, dtype=float)

    def sample_posterior(self, returns: np.ndarray, prior_mean: float, prior_var: float) -> np.ndarray:
        x = np.asarray(returns, dtype=float)
        sample_mean = float(np.mean(x))
        sample_var = float(np.var(x, ddof=1)) if x.size > 1 else float(prior_var)
        sample_var = max(sample_var, 1e-10)
        prior_var = max(float(prior_var), 1e-10)
        size = max(2, self.n_samples)

        if self.posteriorType == "gaussian":
            tau_post = 1.0 / (1.0 / prior_var + x.size / sample_var)
            mu_post = tau_post * (prior_mean / prior_var + x.size * sample_mean / sample_var)
            return self.rng.normal(mu_post, np.sqrt(tau_post), size=size)
        if self.posteriorType == "empirical_bayes":
            data_weight = prior_var / (prior_var + sample_var / max(x.size, 1))
            mu_post = data_weight * sample_mean + (1.0 - data_weight) * prior_mean
            return self.rng.normal(mu_post, np.sqrt(sample_var / max(x.size, 1)), size=size)
        if self.posteriorType in {"t_student", "t-student"}:
            df = max(3, int(self.config.get("df", 5)))
            scale = np.sqrt(sample_var * (df - 2.0) / df)
            return sample_mean + scale * self.rng.standard_t(df, size=size)
        if self.posteriorType == "laplace":
            return self.rng.laplace(sample_mean, np.sqrt(sample_var / 2.0), size=size)
        if self.posteriorType == "horseshoe":
            tau = np.clip(abs(self.rng.standard_cauchy()), 0.01, 5.0)
            lam = np.clip(abs(self.rng.standard_cauchy()), 0.01, 5.0)
            data_weight = np.clip(tau * lam / (1.0 + tau * lam), float(self.config.get("shrink_floor", 0.35)), 1.0)
            mu_post = data_weight * sample_mean + (1.0 - data_weight) * prior_mean
            posterior_sd = np.sqrt(sample_var / max(x.size, 1)) * max(data_weight, 0.05)
            return self.rng.normal(mu_post, max(posterior_sd, 1e-8), size=size)
        if self.posteriorType == "spike_slab":
            p_active = float(self.config.get("spike_slab_p", 0.5))
            active = self.rng.binomial(1, p_active, size=size)
            return active * self.rng.normal(sample_mean, np.sqrt(sample_var), size=size)
        raise ValueError(f"Unknown posterior type: {self.posteriorType}")

    @beartype
    def get_posterior_draws(self) -> np.ndarray:
        prior_mean, prior_var = self.calculate_prior_mean_variance()
        returns_df = self._returns_frame(self.returns)
        draws = np.zeros((max(2, self.n_samples), self.count_etf), dtype=float)
        prior_mu = self.rng.normal(prior_mean, np.sqrt(prior_var), self.count_etf)
        for j, etf in enumerate(self.etfs_list):
            series = returns_df[etf] if etf in returns_df else pd.Series(dtype=float)
            sampled = self.get_returns_with_distribution(series, float(prior_mu[j]))[-1]
            post = self.sample_posterior(sampled, prior_mean, prior_var)
            draws[:, j] = post[: draws.shape[0]]
        return np.nan_to_num(draws, nan=0.0, posinf=0.0, neginf=0.0)

    def _posterior_mean_vector(self) -> np.ndarray:
        rdf = self._returns_frame(self.returns)
        edf = self._returns_frame(self.ewma_returns)
        sample_mean = rdf.mean(axis=0).to_numpy(dtype=float)
        prior_i = edf.mean(axis=0).to_numpy(dtype=float)

        # Deterministic empirical-Bayes-style shrinkage is preferred for repeatable
        # cross-sectional signals; optional random horseshoe draws can be enabled.
        if bool(self.config.get("random_horseshoe_mean", False)):
            tau = np.clip(np.abs(self.rng.standard_cauchy(self.count_etf)), 0.01, 5.0)
            lam = np.clip(np.abs(self.rng.standard_cauchy(self.count_etf)), 0.01, 5.0)
            kappa = np.clip(tau * lam / (1.0 + tau * lam), float(self.config.get("shrink_floor", 0.35)), 1.0)
        else:
            n = max(len(rdf), 1)
            sample_var = rdf.var(axis=0, ddof=1).fillna(0.0).to_numpy(dtype=float)
            prior_strength = float(self.config.get("prior_strength", 63.0))
            kappa = n / (n + prior_strength * (1.0 + sample_var / max(float(np.nanmedian(sample_var)), EPS)))
            kappa = np.clip(kappa, float(self.config.get("shrink_floor", 0.35)), 1.0)
        mu = kappa * sample_mean + (1.0 - kappa) * prior_i

        # 12-1 cross-sectional momentum tilt, calculated only from history.
        tilt = float(self.config.get("mom_tilt", 0.15))
        look = int(self.config.get("mom_lookback", 252))
        gap = int(self.config.get("mom_gap", 21))
        if tilt != 0.0 and len(rdf) >= look:
            window = rdf.iloc[-look:-gap] if gap > 0 else rdf.iloc[-look:]
            if self.scenario_return_type == "log-returns":
                mom = window.sum(axis=0).to_numpy(dtype=float)
            else:
                mom = (1.0 + window).prod(axis=0).to_numpy(dtype=float) - 1.0
            sd = float(np.nanstd(mom))
            if sd > EPS:
                z = (mom - float(np.nanmean(mom))) / sd
                daily_scale = max(float(np.nanmedian(rdf.std(axis=0, ddof=1))), EPS)
                mu = mu + tilt * z * daily_scale

        # Guard against implausible daily means dominating the optimizer.
        mean_clip_sigma = float(self.config.get("mean_clip_sigma", 2.0))
        daily_vol = rdf.std(axis=0, ddof=1).fillna(0.0).to_numpy(dtype=float)
        mu = np.clip(mu, -mean_clip_sigma * daily_vol, mean_clip_sigma * daily_vol)
        return np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)

    def _estimate_daily_covariance(self) -> tuple[np.ndarray, np.ndarray]:
        rdf = self._returns_frame(self.returns)
        method = str(self.config.get("cov_method", "ledoit_wolf"))
        try:
            cov = np.asarray(estimate_covariance(rdf, method=method), dtype=float)
        except Exception:
            cov = np.atleast_2d(np.cov(rdf.to_numpy(dtype=float), rowvar=False))
        cov = np.nan_to_num((cov + cov.T) / 2.0, nan=0.0, posinf=0.0, neginf=0.0)
        cov = make_positive_semifinite_matrix(cov, self.matrixType)
        eigvals, eigvecs = np.linalg.eigh((cov + cov.T) / 2.0)
        eigvals = np.clip(eigvals, 1e-12, None)
        cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
        std = np.sqrt(np.clip(np.diag(cov), EPS, None))
        corr = cov / np.outer(std, std)
        corr = np.clip(corr, -1.0, 1.0)
        np.fill_diagonal(corr, 1.0)
        self.last_covariance_daily = cov
        self.last_correlation = corr
        return cov, corr

    def _daily_shock_cube(self, n_scenarios: int, horizon: int) -> np.ndarray:
        cov, _ = self._estimate_daily_covariance()
        chol = np.linalg.cholesky(cov + 1e-12 * np.eye(cov.shape[0]))
        shape = (int(n_scenarios), int(horizon), self.count_etf)
        if self.distType in {"t_student", "t-student"}:
            df = max(3, int(self.config.get("df", 5)))
            # variance-standardized t innovations
            z = self.rng.standard_t(df, size=shape) * np.sqrt((df - 2.0) / df)
        else:
            z = self.rng.standard_normal(size=shape)
        return z @ chol.T

    @beartype
    def get_multivariate_shock(self) -> np.ndarray:
        return self._daily_shock_cube(max(2, self.n_samples), 1)[:, 0, :]

    def get_risk_matrix_snapshot(self) -> dict[str, Any]:
        if self.last_covariance_daily is None or self.last_correlation is None:
            self._estimate_daily_covariance()
        # Keep large matrices as NumPy arrays while the backtest is running.
        # Converting every monthly N x N matrix to nested Python lists here can
        # multiply memory usage several-fold; across a long walk-forward run this
        # can trigger an OS-level OOM kill during final JSON serialization.
        return {
            "tickers": list(self.etfs_list),
            "covariance_daily": np.asarray(self.last_covariance_daily, dtype=float).copy(),
            "covariance_horizon": np.asarray(
                self.last_covariance_horizon
                if self.last_covariance_horizon is not None
                else self.last_covariance_daily * self.horizon,
                dtype=float,
            ).copy(),
            "correlation": np.asarray(self.last_correlation, dtype=float).copy(),
            "posterior_mean_daily": None if self.last_posterior_mean_daily is None else np.asarray(self.last_posterior_mean_daily, dtype=float).copy(),
            "posterior_mean_horizon": None if self.last_posterior_mean_horizon is None else np.asarray(self.last_posterior_mean_horizon, dtype=float).copy(),
            "horizon": int(self.horizon),
            "return_type": self.scenario_return_type,
            "cov_method": str(self.config.get("cov_method", "ledoit_wolf")),
            # Export the dimension/information ratio used by Proposition 6.
            # These values make the historical theorem bridge independent of
            # assumptions about the original raw-data files.
            "estimation_observations": int(self.returns.shape[0]),
            "selected_dimension": int(self.count_etf),
        }


    def get_scenario_paths(self) -> np.ndarray:
        """Return the most recently generated daily scenario-path cube (S,H,N)."""
        if self.last_daily_scenario_paths is None:
            raise RuntimeError("prediction() must be called before get_scenario_paths().")
        return np.asarray(self.last_daily_scenario_paths, dtype=float).copy()

    @beartype
    def prediction(self) -> np.ndarray:
        S = max(50, int(self.n_scenarios))
        H = max(1, int(self.horizon))
        mu_daily = self._posterior_mean_vector()
        shocks = self._daily_shock_cube(S, H)
        daily = mu_daily[None, None, :] + shocks

        if self.scenario_return_type == "log-returns":
            # Additive log-return paths are exactly compatible with the linear CDaR
            # running-peak construction: cumulative sums are log-wealth changes.
            self.last_daily_scenario_paths = np.asarray(daily, dtype=float)
            scenarios = daily.sum(axis=1)
            cov_h = self.last_covariance_daily * H
            mu_h = mu_daily * H
        elif self.scenario_return_type == "returns":
            daily = np.clip(daily, -0.95, 5.0)
            # Store the same clipped simple-return draws used to form terminal
            # scenarios.  The CDaR LP then uses their additive/uncompounded path
            # coordinate; exact compounded-wealth CDaR is a nonlinear robustness arm.
            self.last_daily_scenario_paths = np.asarray(daily, dtype=float)
            scenarios = np.prod(1.0 + daily, axis=1) - 1.0
            cov_h = np.cov(scenarios, rowvar=False)
            mu_h = scenarios.mean(axis=0)
        else:
            raise ValueError("scenario_return_type must be 'log-returns' or 'returns'")

        self.last_covariance_horizon = np.atleast_2d(np.asarray(cov_h, dtype=float))
        self.last_posterior_mean_daily = mu_daily
        self.last_posterior_mean_horizon = np.asarray(mu_h, dtype=float)
        return np.nan_to_num(scenarios, nan=0.0, posinf=0.0, neginf=0.0)
