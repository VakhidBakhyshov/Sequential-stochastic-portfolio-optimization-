import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd

from typing import Any
from beartype import beartype

from scripts.calculations.matrix import *
from scripts.models.base import BaseModel

from scripts.calculations.covariance_cleaning import estimate_covariance

class BayessianModel(BaseModel):
    def __init__(
        self,
        config: dict[str, Any],
        etfs_list: list,
        historical_returns: pd.DataFrame,
        historical_ewma_returns: pd.DataFrame,
        future_returns: np.ndarray
    ):
        
        self.config = config
        super().__init__(self.config, etfs_list, historical_returns, historical_ewma_returns, future_returns)
        
        self.meanReturnType = self.config.get("mean_return", "sampled")
        self.distType = self.config.get("distribution", "t_student")
        self.posteriorType = self.config.get("posterior", "horseshoe")
        self.matrixType = self.config.get("matrix_type", "fast")
        self.is_sampled_mean = self.config.get("is_sampled_mean", False)
        self.random_state = int(self.config.get("random_state", 42))
        self.rng = np.random.default_rng(self.random_state)
        
        self.count_etf = len(self.etfs_list)
        self.len_returns = len(future_returns)
            
        self.n_samples = self.config.get("n_samples", 1000) if self.is_sampled_mean else 1  
        self.posterior_draws = np.empty((self.len_returns, self.count_etf), dtype=float)
        
        
    def _returns_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'Date' in df.columns:
            df = df.drop('Date', axis=1)
        return df.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        
    @beartype
    def calculate_prior_mean_variance(self) -> tuple[float, float]:        
        returns = self._returns_frame(self.ewma_returns)
        prior_mean_0 = float(returns.mean().mean())
        squared_prior_variance_0 = float((returns.std(ddof=1) ** 2).mean())
        if not np.isfinite(squared_prior_variance_0) or squared_prior_variance_0 <= 1e-12:
            squared_prior_variance_0 = 1e-6
        return prior_mean_0, squared_prior_variance_0


    def get_returns_with_distribution(self, returns_etf: pd.Series, mu_i: float) -> tuple[float, float, np.ndarray]:
        sigma_i = float(returns_etf.std(ddof=1))
        if not np.isfinite(sigma_i) or sigma_i <= 1e-12:
            sigma_i = 1e-4
        
        # 1 way - calculate posterior_mean_variance from historical returns
        if self.meanReturnType == "historical":
            r_i = returns_etf.values.astype(float)
            mean_return = float(np.mean(r_i))
            
        # 2 way - calculate posterior_mean_variance from sampled returns
        elif self.meanReturnType == "sampled":
            if self.distType == "normal":
                r_i = self.rng.normal(mu_i, sigma_i, self.n_samples)
            elif self.distType == "t_student":
                df = int(self.config.get("df", 5)) # degrees of freedom for t-distribution
                r_i = mu_i + sigma_i * self.rng.standard_t(df, size=self.n_samples)
            elif self.distType == "laplace":
                r_i = self.rng.laplace(mu_i, sigma_i / np.sqrt(2), self.n_samples)
            elif self.distType == "lognormal":
                r_i = self.rng.lognormal(mean=mu_i, sigma=sigma_i, size=self.n_samples)
            else:
                raise ValueError(f"Unsupported distribution: {self.distType}")
            mean_return = float(np.mean(r_i))
            
        else:
            raise ValueError(f"Unsupported mean return type: {self.meanReturnType}")

        return sigma_i**2, mean_return, np.asarray(r_i, dtype=float)


    def sample_posterior(self, returns: np.ndarray, prior_mean: float, prior_var: float) -> np.ndarray:
        returns = np.asarray(returns, dtype=float)
        sample_mean = float(np.mean(returns))
        sample_var = float(np.var(returns, ddof=1)) if returns.size > 1 else prior_var
        sample_var = max(sample_var, 1e-10)
        prior_var = max(float(prior_var), 1e-10)

        if self.posteriorType == "gaussian":
            tau_post = 1 / (1 / prior_var + self.n_samples / sample_var)
            mu_post = tau_post * (prior_mean / prior_var + self.n_samples * sample_mean / sample_var)
            return self.rng.normal(mu_post, np.sqrt(tau_post), size=self.n_samples)
            
            # sample_mean, sample_var = np.mean(returns), np.var(returns)
            # # Empirical Bayes shrinkage
            # shrinkage = sample_var / (sample_var + prior_var + 1e-8)
            # mu_post = shrinkage * sample_mean + (1 - shrinkage) * prior_mean
            # tau_post = 1.0 / (1.0/prior_var + self.n_samples/sample_var)
            # return np.random.normal(mu_post, np.sqrt(tau_post), size=self.n_samples)
        
        elif self.posteriorType == "empirical_bayes":
            # James-Stein style shrinkage
            # grand_mean = np.mean(returns)
            # shrinkage = max(0, 1 - (len(returns) - 3) * sample_var / (self.n_samples * (sample_mean - grand_mean)**2 + 1e-10))
            # mu_post = shrinkage * sample_mean + (1 - shrinkage) * grand_mean
            # sigma_post = np.sqrt(sample_var * shrinkage)
            # return np.random.normal(mu_post, sigma_post, size=self.n_samples)
            
            shrinkage = sample_var / (sample_var + prior_var)
            mu_post = shrinkage * sample_mean + (1 - shrinkage) * prior_mean
            sigma_post = np.sqrt(sample_var * max(shrinkage, 1e-4))
            return self.rng.normal(mu_post, sigma_post, size=self.n_samples)

        elif self.posteriorType == "t_student":
            df = self.config.get("df", 5)
            scale = np.sqrt(sample_var * max((df - 2) / df, 1e-6))
            return sample_mean + scale * self.rng.standard_t(df, size=self.n_samples)

        elif self.posteriorType == "laplace":
            b = np.sqrt(sample_var / 2)
            return self.rng.laplace(sample_mean, b, size=self.n_samples)

        elif self.posteriorType == "horseshoe":
            tau = np.clip(abs(self.rng.standard_cauchy()), 0.01, 5.0)
            lam = np.clip(abs(self.rng.standard_cauchy()), 0.01, 5.0)
            shrinkage = tau * lam / (1.0 + tau * lam)
            mu_post = sample_mean * shrinkage + prior_mean * (1.0 - shrinkage)
            sigma_post = np.sqrt(sample_var) * max(shrinkage, 0.05)
            return self.rng.normal(mu_post, sigma_post, size=self.n_samples)

        elif self.posteriorType == "spike_slab":
            p_active = float(self.config.get("spike_slab_p", 0.5))  # probability of being "active"
            # is_active = self.rng.binomial(1, p_active)
            # if is_active:
            #     return self.rng.normal(sample_mean, np.sqrt(sample_var), size=self.n_samples)
            # return np.zeros(self.n_samples)
            active = self.rng.binomial(1, p_active, size=self.n_samples)
            draws = self.rng.normal(sample_mean, np.sqrt(sample_var), size=self.n_samples)
            return active * draws

        else:
            raise ValueError(f"Unknown posterior type: {self.posteriorType}")


    @beartype
    def get_posterior_draws(self) -> np.ndarray:
        prior_mean_0, squared_prior_variance_0 = self.calculate_prior_mean_variance()
        mu_i = np.random.normal(prior_mean_0, np.sqrt(squared_prior_variance_0), self.count_etf)
        returns_df = self._returns_frame(self.returns)
        
        for ind, etf in enumerate(self.etfs_list):
            series = returns_df[etf] if etf in returns_df.columns else pd.Series(np.zeros(len(returns_df)))
            r_i = self.get_returns_with_distribution(series, mu_i[ind])[-1]
            posterior_samples = self.sample_posterior(r_i, prior_mean_0, squared_prior_variance_0)
            self.posterior_draws[:, ind] = np.mean(posterior_samples) if self.is_sampled_mean else posterior_samples
            # print(f"{self.posterior_draws.shape}", f"{posterior_samples.shape=}")

        return np.nan_to_num(self.posterior_draws, nan=0.0, posinf=0.0, neginf=0.0)


    @beartype
    def get_multivariate_shock(self) -> np.ndarray:
        returns_df = self._returns_frame(self.returns[self.etfs_list])
        cov_method = self.config.get("cov_method", "ledoit_wolf")
        try:
            cov_matrix = estimate_covariance(returns_df, method=cov_method)
        except Exception:
            cov_matrix = np.cov(np.nan_to_num(returns_df, nan=0.0), rowvar=False)
        
        stds = np.sqrt(np.clip(np.diag(cov_matrix), 1e-12, None))
        inv_stds = np.diag(1.0 / np.where(stds < 1e-10, 1e-10, stds))
        
        corr_matrix = np.dot(np.dot(inv_stds, cov_matrix), inv_stds)
        corr_matrix = make_positive_semifinite_matrix(corr_matrix, self.matrixType)
        cov_matrix = np.dot(np.dot(np.diag(stds), corr_matrix), np.diag(stds))
        
        # 1 way - updated + complicated
        cov_matrix = cholesky_decomposition(cov_matrix)
        
        # --- FIX FOR UNIQUE N_SAMPLES PER DAY ---
        if self.is_sampled_mean:
            # 1. Generate standard normal random variables in 3D: (21, 1000, 211)
            if self.distType in {"t_student", "t-student"}:
                df = int(self.config.get("df", 5))
                # Match the 3D shape for the chi-square scaling factor
                g = self.rng.chisquare(df=df, size=(self.len_returns, self.n_samples)) / df
                z = self.rng.standard_normal(size=(self.len_returns, self.n_samples, self.count_etf)) / np.sqrt(g)[:, :, None]
            else:
                z = self.rng.standard_normal(size=(self.len_returns, self.n_samples, self.count_etf))

            # 2. Matrix multiply 3D array by 2D Cholesky matrix
            # Shape transformation: (21, 1000, 211) @ (211, 211) -> (21, 1000, 211)
            shocks_3d = z @ cov_matrix.T 
            
            # 3. Take the mean across axis 1 (the 1000 samples dimension)
            # Shape transformation: (21, 1000, 211) -> (21, 211)
            return np.mean(shocks_3d, axis=1)
            
        else:
            # Fallback for when is_sampled_mean is False (original logic)
            if self.distType in {"t_student", "t-student"}:
                df = int(self.config.get("df", 5))
                g = self.rng.chisquare(df=df, size=self.n_samples) / df
                z = self.rng.standard_normal(size=(self.n_samples, self.count_etf)) / np.sqrt(g)[:, None]
            else:
                z = self.rng.standard_normal(size=(self.n_samples, self.count_etf))

            return z @ cov_matrix.T
    
        # # 2 way - more simple + BEST WORKING
        # return calculate_multivariate_shock_matrix(cov_matrix, self.len_returns)


    @beartype
    def prediction(self) -> np.ndarray:
        posterior_draws = self.get_posterior_draws()
        shocks = self.get_multivariate_shock()
        return np.nan_to_num(posterior_draws + shocks, nan=0.0, posinf=0.0, neginf=0.0)
