# scripts/calculations/covariance.py
from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf, OAS, GraphicalLassoCV

EPS = 1e-10


def _symmetrize(x: np.ndarray) -> np.ndarray:
    return 0.5 * (x + x.T)


def nearest_psd(matrix: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    matrix = _symmetrize(np.asarray(matrix, dtype=float))
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.clip(eigvals, eps, None)
    return _symmetrize(eigvecs @ np.diag(eigvals) @ eigvecs.T)


def cov_to_corr(cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cov = _symmetrize(cov)
    std = np.sqrt(np.clip(np.diag(cov), EPS, None))
    inv_std = np.diag(1.0 / std)
    corr = inv_std @ cov @ inv_std
    return _symmetrize(corr), std


def corr_to_cov(corr: np.ndarray, std: np.ndarray) -> np.ndarray:
    d = np.diag(std)
    return _symmetrize(d @ corr @ d)


def winsorize_by_mad(x: np.ndarray, z: float = 8.0) -> np.ndarray:
    med = np.nanmedian(x, axis=0)
    mad = np.nanmedian(np.abs(x - med), axis=0)
    scale = 1.4826 * np.where(mad < EPS, 1.0, mad)
    lo = med - z * scale
    hi = med + z * scale
    return np.clip(x, lo, hi)


def standardize_returns(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    x = np.where(np.isfinite(x), x, np.nan)
    x = winsorize_by_mad(x)
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0, ddof=1)
    std = np.where(std < EPS, 1.0, std)
    z = (np.nan_to_num(x, nan=mean) - mean) / std
    return z, std


def mp_clip_correlation(z: np.ndarray) -> np.ndarray:
    t, n = z.shape # z: standardized residuals, shape T x N
    corr = np.corrcoef(z, rowvar=False)
    corr = _symmetrize(corr)

    eigvals, eigvecs = np.linalg.eigh(corr)
    q = max(t / max(n, 1), 1.0 + EPS)
    lambda_plus = (1.0 + np.sqrt(1.0 / q)) ** 2

    noise_mask = eigvals < lambda_plus
    if np.any(noise_mask):
        bulk_mean = eigvals[noise_mask].mean()
        eigvals[noise_mask] = bulk_mean

    corr_clean = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return nearest_psd(corr_clean)


def detone_correlation(corr: np.ndarray, n_components: int = 1) -> np.ndarray:
    if n_components <= 0:
        return nearest_psd(corr)

    eigvals, eigvecs = np.linalg.eigh(_symmetrize(corr))
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    market = eigvecs[:, :n_components] @ np.diag(eigvals[:n_components]) @ eigvecs[:, :n_components].T
    detoned = _symmetrize(corr - market)
    np.fill_diagonal(detoned, 1.0)
    return nearest_psd(detoned)


def clean_covariance(
    returns: np.ndarray,
    method: str = "oas",
    detone_n: int = 1,
) -> np.ndarray:
    x = np.asarray(returns, dtype=float)
    if x.ndim != 2:
        raise ValueError("returns must be 2D: T x N")

    z, std = standardize_returns(x)

    if method == "oas":
        cov_z = OAS().fit(z).covariance_
        corr, _ = cov_to_corr(cov_z)
    elif method == "ledoit_wolf":
        cov_z = LedoitWolf().fit(z).covariance_
        corr, _ = cov_to_corr(cov_z)
    elif method == "graphical_lasso":
        cov_z = GraphicalLassoCV().fit(z).covariance_
        corr, _ = cov_to_corr(cov_z)
    elif method == "mp":
        corr = mp_clip_correlation(z)
    else:
        corr = np.corrcoef(z, rowvar=False)

    corr = nearest_psd(corr)
    corr = detone_correlation(corr, n_components=detone_n)
    cov = corr_to_cov(corr, std)
    return nearest_psd(cov)


def simulate_student_t_scenarios(
    mu: np.ndarray,
    historical_returns: np.ndarray,
    n_scenarios: int = 2000,
    dof: int = 7,
    cov_method: str = "oas",
    detone_n: int = 1,
    seed: int = 42,
) -> np.ndarray:
    """
    mu: vector expected monthly log-returns, shape (N,)
    returns scenarios: shape (n_scenarios, N)
    """
    rng = np.random.default_rng(seed)
    mu = np.asarray(mu, dtype=float).reshape(1, -1)

    cov = clean_covariance(historical_returns, method=cov_method, detone_n=detone_n)
    chol = np.linalg.cholesky(nearest_psd(cov))

    g = rng.chisquare(df=dof, size=n_scenarios) / dof
    z = rng.standard_normal(size=(n_scenarios, mu.shape[1]))
    t_noise = z / np.sqrt(g)[:, None]

    return mu + t_noise @ chol.T
