from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Union
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf, OAS, GraphicalLassoCV
from sklearn.decomposition import PCA

EPS = 1e-10


def as_returns_frame(returns: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
    df = returns.copy() if isinstance(returns, pd.DataFrame) else pd.DataFrame(np.asarray(returns))
    if "Date" in df.columns:
        df = df.drop(columns=["Date"])
    return df.astype(float)


def clean_returns(
    returns: Union[pd.DataFrame, np.ndarray],
    winsor_z: Union[float, None] = 5.0,
    min_obs: int = 60,
    fill_method: Union[str, None] = None,
) -> pd.DataFrame:
    df = as_returns_frame(returns).replace([np.inf, -np.inf], np.nan)
    df = df.loc[:, df.notna().sum(axis=0) >= min_obs]

    if winsor_z is not None:
        med = df.median(axis=0)
        mad = (df - med).abs().median(axis=0).replace(0.0, np.nan)
        scale = 1.4826 * mad
        lower = med - winsor_z * scale
        upper = med + winsor_z * scale
        df = df.clip(lower=lower, upper=upper, axis=1)

    if fill_method == "zero":
        df = df.fillna(0.0)
    elif fill_method == "ffill":
        df = df.ffill().dropna(axis=0, how="any")
    else:
        df = df.dropna(axis=0, how="any")

    if df.shape[1] < 2:
        raise ValueError("После очистки осталось меньше двух активов.")
    if df.shape[0] < 3:
        raise ValueError("После очистки осталось слишком мало наблюдений.")
    return df


def cov_to_corr(cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cov = np.asarray(cov, dtype=float)
    std = np.sqrt(np.clip(np.diag(cov), EPS, None))
    corr = cov / np.outer(std, std)
    corr = np.clip((corr + corr.T) / 2.0, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr, std


def corr_to_cov(corr: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (np.asarray(corr) * np.outer(std, std) + 0.0).astype(float)


def nearest_psd(matrix: np.ndarray, eps: float = 1e-8, unit_diag: bool = False) -> np.ndarray:
    a = (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(a)
    eigvals = np.clip(eigvals, eps, None)
    out = eigvecs @ np.diag(eigvals) @ eigvecs.T
    out = (out + out.T) / 2.0
    if unit_diag:
        np.fill_diagonal(out, 1.0)
    return out


def sample_covariance(returns: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    x = clean_returns(returns).to_numpy()
    return nearest_psd(np.cov(x, rowvar=False, ddof=1))


def ledoit_wolf_covariance(returns: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    x = clean_returns(returns).to_numpy()
    return nearest_psd(LedoitWolf().fit(x).covariance_)


def oas_covariance(returns: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    x = clean_returns(returns).to_numpy()
    return nearest_psd(OAS().fit(x).covariance_)


def pca_factor_covariance(
    returns: Union[pd.DataFrame, np.ndarray],
    n_components: Union[int, None] = None,
    explained_variance: float = 0.80,
) -> np.ndarray:
    df = clean_returns(returns)
    x = df.to_numpy()
    x = x - x.mean(axis=0, keepdims=True)

    if n_components is None:
        pca_full = PCA().fit(x)
        csum = np.cumsum(pca_full.explained_variance_ratio_)
        n_components = int(np.searchsorted(csum, explained_variance) + 1)
        n_components = max(1, min(n_components, x.shape[1]))

    pca = PCA(n_components=n_components).fit(x)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    common = loadings @ loadings.T
    resid = x - pca.transform(x) @ pca.components_
    specific = np.diag(np.var(resid, axis=0, ddof=1))
    return nearest_psd(common + specific)


def graphical_lasso_covariance(returns: Union[pd.DataFrame, np.ndarray], alphas: int = 5) -> np.ndarray:
    x = clean_returns(returns).to_numpy()
    return nearest_psd(GraphicalLassoCV(alphas=alphas).fit(x).covariance_)


def cluster_order(corr: np.ndarray) -> np.ndarray:
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, 1.0))
    np.fill_diagonal(dist, 0.0)
    return leaves_list(linkage(squareform(dist, checks=False), method="average"))


def tapered_correlation(corr: np.ndarray, bandwidth: int = 10) -> np.ndarray:
    order = cluster_order(np.asarray(corr, dtype=float))
    c = corr[np.ix_(order, order)]
    n = c.shape[0]
    idx = np.arange(n)
    kernel = np.maximum(1.0 - np.abs(idx[:, None] - idx[None, :]) / max(bandwidth, 1), 0.0)
    c_taper = nearest_psd(c * kernel, unit_diag=True)
    inv_order = np.argsort(order)
    return c_taper[np.ix_(inv_order, inv_order)]


def rmt_denoised_covariance(
    returns: Union[pd.DataFrame, np.ndarray],
    preserve_factors: int = 1,
    replace: str = "average",
) -> np.ndarray:
    df = clean_returns(returns)
    x = df.to_numpy()
    std = x.std(axis=0, ddof=1)
    std = np.where(std < EPS, EPS, std)
    z = (x - x.mean(axis=0, keepdims=True)) / std

    corr = nearest_psd(np.corrcoef(z, rowvar=False), unit_diag=True)
    eigvals, eigvecs = np.linalg.eigh(corr)

    n_obs, n_assets = z.shape
    q = n_assets / n_obs
    lam_plus = (1.0 + np.sqrt(q)) ** 2

    keep = np.zeros_like(eigvals, dtype=bool)
    if preserve_factors > 0:
        keep[np.argsort(eigvals)[-preserve_factors:]] = True

    bulk_mask = (eigvals <= lam_plus) & (~keep)
    clean_eigvals = eigvals.copy()

    if np.any(bulk_mask):
        if replace == "average":
            clean_eigvals[bulk_mask] = eigvals[bulk_mask].mean()
        elif replace == "clip":
            clean_eigvals[bulk_mask] = lam_plus
        else:
            raise ValueError("replace must be 'average' or 'clip'")

    corr_clean = nearest_psd(eigvecs @ np.diag(clean_eigvals) @ eigvecs.T, unit_diag=True)
    return nearest_psd(corr_to_cov(corr_clean, std))


def estimate_covariance(returns: Union[pd.DataFrame, np.ndarray], method: str = "ledoit_wolf", **kwargs) -> np.ndarray:
    method = method.lower()
    if method == "sample":
        return sample_covariance(returns)
    if method in {"ledoit_wolf", "lw"}:
        return ledoit_wolf_covariance(returns)
    if method == "oas":
        return oas_covariance(returns)
    if method == "rmt":
        return rmt_denoised_covariance(returns, **kwargs)
    if method == "pca":
        return pca_factor_covariance(returns, **kwargs)
    if method in {"glasso", "graphical_lasso"}:
        return graphical_lasso_covariance(returns, **kwargs)
    if method == "tapered":
        cov = sample_covariance(returns)
        corr, std = cov_to_corr(cov)
        return nearest_psd(corr_to_cov(tapered_correlation(corr, **kwargs), std))
    raise ValueError(f"Неизвестный метод: {method}")
