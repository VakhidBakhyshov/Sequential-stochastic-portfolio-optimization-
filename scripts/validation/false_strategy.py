"""Detection of false investment strategies.

The implementation follows the research logic used in the supplied López de Prado
lectures and related Bailey--López de Prado work:

* Probabilistic Sharpe Ratio (PSR): adjusts Sharpe inference for finite samples,
  skewness and kurtosis.
* False Strategy Theorem: estimates the Sharpe level expected from selecting the
  maximum among many uninformed trials.
* Deflated Sharpe Ratio (DSR): evaluates the observed strategy against that
  multiple-testing benchmark.
* FWER/FDR corrections: Bonferroni, Holm, Benjamini--Hochberg and
  Benjamini--Yekutieli.
* A nested validation/internal-test overfitting diagnostic and a structural
  causality audit for the monthly walk-forward engine.

All Sharpe values exposed to users are annualized. Internally, PSR is calculated
at the return sampling frequency, as required by the theorem.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.stats import kurtosis, norm, skew

EPS = 1e-12
EULER_MASCHERONI = 0.5772156649015329


@dataclass(frozen=True)
class SharpeInference:
    annualized_sharpe: float
    benchmark_sharpe: float
    probability: float
    z_stat: float
    observations: int
    effective_observations: float
    skewness: float
    kurtosis: float


def _clean_returns(returns: pd.Series | np.ndarray | Iterable[float]) -> pd.Series:
    return pd.Series(np.asarray(list(returns) if not isinstance(returns, (pd.Series, np.ndarray)) else returns, dtype=float)).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()


def annualized_sharpe(
    returns: pd.Series | np.ndarray | Iterable[float],
    *,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 12,
) -> float:
    r = _clean_returns(returns)
    if len(r) < 2:
        return np.nan
    excess = r - float(risk_free_rate) / float(periods_per_year)
    sd = float(excess.std(ddof=1))
    return float(excess.mean() / sd * math.sqrt(periods_per_year)) if sd > EPS else np.nan


def effective_sample_size(
    returns: pd.Series | np.ndarray | Iterable[float],
    *,
    max_lag: int | None = None,
) -> float:
    """Bartlett-kernel effective sample size for serially correlated returns."""
    r = _clean_returns(returns)
    n = len(r)
    if n < 3:
        return float(n)
    lag = int(max_lag if max_lag is not None else min(12, max(1, n // 5)))
    lag = min(lag, n - 2)
    inflation = 1.0
    for k in range(1, lag + 1):
        rho = float(r.autocorr(lag=k))
        if not np.isfinite(rho):
            continue
        weight = 1.0 - k / (lag + 1.0)
        inflation += 2.0 * weight * rho
    inflation = max(inflation, EPS)
    return float(np.clip(n / inflation, 2.0, float(n)))


def probabilistic_sharpe_ratio(
    returns: pd.Series | np.ndarray | Iterable[float],
    *,
    benchmark_sharpe: float = 0.0,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 12,
    adjust_serial_correlation: bool = True,
    max_lag: int | None = None,
    return_details: bool = False,
) -> float | SharpeInference:
    """Estimate P(SR > benchmark) with finite-sample/non-normality corrections.

    ``benchmark_sharpe`` is annualized for a convenient public API. The Bailey--
    López de Prado statistic is evaluated using non-annualized Sharpe values.
    """
    r = _clean_returns(returns)
    n = len(r)
    if n < 3:
        result = SharpeInference(np.nan, float(benchmark_sharpe), np.nan, np.nan, n, float(n), np.nan, np.nan)
        return result if return_details else result.probability

    excess = r - float(risk_free_rate) / float(periods_per_year)
    sd = float(excess.std(ddof=1))
    if sd <= EPS:
        result = SharpeInference(np.nan, float(benchmark_sharpe), np.nan, np.nan, n, float(n), np.nan, np.nan)
        return result if return_details else result.probability

    sr_freq = float(excess.mean() / sd)
    sr_ann = sr_freq * math.sqrt(periods_per_year)
    sr0_freq = float(benchmark_sharpe) / math.sqrt(periods_per_year)
    sk = float(skew(excess, bias=False)) if n >= 3 else 0.0
    ku = float(kurtosis(excess, fisher=False, bias=False)) if n >= 4 else 3.0
    n_eff = effective_sample_size(excess, max_lag=max_lag) if adjust_serial_correlation else float(n)

    variance_term = 1.0 - sk * sr_freq + ((ku - 1.0) / 4.0) * sr_freq * sr_freq
    if not np.isfinite(variance_term) or variance_term <= EPS:
        result = SharpeInference(sr_ann, float(benchmark_sharpe), np.nan, np.nan, n, n_eff, sk, ku)
        return result if return_details else result.probability

    z_stat = (sr_freq - sr0_freq) * math.sqrt(max(n_eff - 1.0, 1.0)) / math.sqrt(variance_term)
    probability = float(norm.cdf(z_stat))
    result = SharpeInference(sr_ann, float(benchmark_sharpe), probability, float(z_stat), n, n_eff, sk, ku)
    return result if return_details else result.probability


def expected_max_sharpe(
    num_trials: int | float,
    *,
    sharpe_mean: float = 0.0,
    sharpe_std: float = 1.0,
) -> float:
    """False Strategy Theorem approximation for E[max(SR_1, ..., SR_K)]."""
    k = max(float(num_trials), 1.0)
    if k <= 1.0:
        return float(sharpe_mean)
    p1 = np.clip(1.0 - 1.0 / k, EPS, 1.0 - EPS)
    p2 = np.clip(1.0 - 1.0 / (k * math.e), EPS, 1.0 - EPS)
    extreme = (1.0 - EULER_MASCHERONI) * norm.ppf(p1) + EULER_MASCHERONI * norm.ppf(p2)
    return float(sharpe_mean + max(float(sharpe_std), 0.0) * extreme)


def exact_expected_max_sharpe_gaussian(
    num_trials: int | float,
    *,
    sharpe_mean: float = 0.0,
    sharpe_std: float = 1.0,
) -> float:
    """Numerical Gaussian reference for E[max(SR)] used to validate the FST approximation.

    This is not a replacement for the Bailey-Lopez de Prado extreme-value
    approximation used by DSR.  It is a unit-test/diagnostic reference under the
    same iid Gaussian null assumption.
    """
    k = max(float(num_trials), 1.0)
    sigma = max(float(sharpe_std), 0.0)
    if sigma <= EPS or k <= 1.0:
        return float(sharpe_mean)

    def integrand(z: float) -> float:
        log_density = math.log(k) + float(norm.logpdf(z)) + (k - 1.0) * float(norm.logcdf(z))
        if log_density < -745.0:
            return 0.0
        return z * math.exp(log_density)

    # For K up to millions the maximum of standard Gaussians remains well inside
    # [-10, 10].  The omitted tail mass is negligible at double precision.
    ez, _ = quad(integrand, -10.0, 10.0, epsabs=1e-10, epsrel=1e-8, limit=250)
    return float(sharpe_mean + sigma * ez)


def expected_max_sharpe_reference_curve(
    trial_counts: Iterable[int | float],
    *,
    sharpe_mean: float = 0.0,
    sharpe_std: float = 1.0,
) -> pd.DataFrame:
    """Compare the False Strategy extreme-value approximation with exact Gaussian integration.

    This is a diagnostic for the pedagogical ``std(SR)=1`` figure and for any
    research-calibrated Gaussian null.  It does *not* change the DSR benchmark;
    DSR continues to use the Bailey--Lopez de Prado approximation.
    """
    rows: list[dict[str, float]] = []
    for value in trial_counts:
        k = max(float(value), 1.0)
        approx = expected_max_sharpe(k, sharpe_mean=sharpe_mean, sharpe_std=sharpe_std)
        exact = exact_expected_max_sharpe_gaussian(k, sharpe_mean=sharpe_mean, sharpe_std=sharpe_std)
        abs_error = approx - exact
        rows.append({
            "number_of_trials": k,
            "expected_max_sharpe_approx": float(approx),
            "expected_max_sharpe_exact_gaussian": float(exact),
            "approx_minus_exact": float(abs_error),
            "relative_error": float(abs_error / exact) if abs(exact) > EPS else np.nan,
            "null_sharpe_mean": float(sharpe_mean),
            "null_sharpe_std": float(sharpe_std),
        })
    return pd.DataFrame(rows)


def hac_mean_test(
    values: pd.Series | np.ndarray | Iterable[float],
    *,
    benchmark: float = 0.0,
    max_lag: int | None = None,
) -> dict[str, float]:
    """One-sided mean test using a Bartlett/Newey-West long-run variance estimate.

    The returned p-value tests ``E[x] > benchmark`` with a normal approximation.
    It is preferable to an iid t-test for overlapping validation-window scores,
    although a block bootstrap remains a useful publication robustness check.
    """
    x = _clean_returns(values) - float(benchmark)
    n = len(x)
    if n < 2:
        return {"mean": float(x.mean()) if n else np.nan, "hac_se": np.nan, "hac_standard_error": np.nan, "z_stat": np.nan, "pvalue": 1.0, "pvalue_one_sided": 1.0, "lag": 0.0}
    arr = x.to_numpy(dtype=float)
    mean = float(arr.mean())
    u = arr - mean
    lag = int(max_lag if max_lag is not None else min(12, max(1, n // 5)))
    lag = min(lag, n - 1)
    gamma0 = float(np.dot(u, u) / n)
    lrv = gamma0
    for k in range(1, lag + 1):
        weight = 1.0 - k / (lag + 1.0)
        gamma = float(np.dot(u[k:], u[:-k]) / n)
        lrv += 2.0 * weight * gamma
    lrv = max(float(lrv), EPS)
    se = math.sqrt(lrv / n)
    z = mean / se if se > EPS else np.nan
    p_one = float(1.0 - norm.cdf(z)) if np.isfinite(z) else 1.0
    return {"mean": mean, "hac_se": float(se), "hac_standard_error": float(se), "z_stat": float(z) if np.isfinite(z) else np.nan, "pvalue": float(np.clip(p_one, 0.0, 1.0)), "pvalue_one_sided": float(np.clip(p_one, 0.0, 1.0)), "lag": float(lag)}


def regularize_trial_correlation(
    correlation: pd.DataFrame | np.ndarray,
    *,
    shrinkage: float = 0.10,
) -> np.ndarray:
    """Return a finite positive-semidefinite trial correlation matrix.

    Sparse recipe panels create undefined pairwise correlations.  Rather than
    silently treating every missing pair as exactly independent, missing off-diagonal
    entries are filled with the median observed cross-trial correlation, the matrix
    is shrunk toward the identity, projected to PSD, and renormalized to unit
    diagonal.  This is a robustness-oriented estimator, not a structural model.
    """
    a = np.asarray(correlation, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] == 0:
        return np.eye(1)
    a = (a + a.T) / 2.0
    n = a.shape[0]
    off = a[~np.eye(n, dtype=bool)]
    finite_off = off[np.isfinite(off)]
    prior = float(np.median(finite_off)) if finite_off.size else 0.0
    a = np.where(np.isfinite(a), a, prior)
    np.fill_diagonal(a, 1.0)
    lam = float(np.clip(shrinkage, 0.0, 1.0))
    a = (1.0 - lam) * a + lam * np.eye(n)
    eigval, eigvec = np.linalg.eigh((a + a.T) / 2.0)
    eigval = np.clip(eigval, 1e-10, None)
    a = (eigvec * eigval) @ eigvec.T
    d = np.sqrt(np.clip(np.diag(a), 1e-12, None))
    a = a / np.outer(d, d)
    np.fill_diagonal(a, 1.0)
    return np.clip((a + a.T) / 2.0, -1.0, 1.0)


def deflated_sharpe_ratio(
    returns: pd.Series | np.ndarray | Iterable[float],
    *,
    num_trials: int | float = 1,
    trial_sharpe_mean: float = 0.0,
    trial_sharpe_std: float = 1.0,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 12,
    adjust_serial_correlation: bool = True,
    return_details: bool = False,
) -> float | SharpeInference:
    benchmark = expected_max_sharpe(
        num_trials,
        sharpe_mean=trial_sharpe_mean,
        sharpe_std=trial_sharpe_std,
    )
    return probabilistic_sharpe_ratio(
        returns,
        benchmark_sharpe=benchmark,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
        adjust_serial_correlation=adjust_serial_correlation,
        return_details=return_details,
    )


def minimum_track_record_length(
    returns: pd.Series | np.ndarray | Iterable[float],
    *,
    benchmark_sharpe: float = 0.0,
    confidence: float = 0.95,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 12,
) -> float:
    """Minimum observations required for PSR to exceed ``confidence``."""
    r = _clean_returns(returns)
    if len(r) < 4:
        return np.nan
    excess = r - float(risk_free_rate) / float(periods_per_year)
    sd = float(excess.std(ddof=1))
    if sd <= EPS:
        return np.inf
    sr = float(excess.mean() / sd)
    sr0 = float(benchmark_sharpe) / math.sqrt(periods_per_year)
    gap = sr - sr0
    if gap <= EPS:
        return np.inf
    sk = float(skew(excess, bias=False))
    ku = float(kurtosis(excess, fisher=False, bias=False))
    variance_term = 1.0 - sk * sr + ((ku - 1.0) / 4.0) * sr * sr
    z = float(norm.ppf(np.clip(confidence, 0.500001, 1.0 - EPS)))
    return float(1.0 + variance_term * (z / gap) ** 2)


def effective_number_of_trials(correlation: pd.DataFrame | np.ndarray) -> float:
    """Eigenvalue participation ratio, bounded between 1 and the trial count."""
    a = np.asarray(correlation, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] == 0:
        return 1.0
    a = regularize_trial_correlation(a, shrinkage=0.0)
    eig = np.linalg.eigvalsh(a)
    eig = np.clip(eig, 0.0, None)
    denom = float(np.square(eig).sum())
    if denom <= EPS:
        return 1.0
    rank = float(eig.sum() ** 2 / denom)
    return float(np.clip(rank, 1.0, a.shape[0]))


def false_strategy_density_surface(
    *,
    trial_counts: Iterable[float] | None = None,
    max_sharpes: Iterable[float] | None = None,
    sharpe_mean: float = 0.0,
    sharpe_std: float = 1.0,
) -> pd.DataFrame:
    """Density surface of max(SR) under K iid Gaussian *null* strategies.

    "Uninformed" means a no-skill/null population of Sharpe estimates; it does
    *not* mean a Uniform distribution.  ``sharpe_std=1`` is a standardized
    pedagogical null unless the caller calibrates mean/std from the research
    trial universe.
    """
    counts = np.asarray(list(trial_counts) if trial_counts is not None else np.unique(np.geomspace(10, 1_000_000, 120).astype(int)), dtype=float)
    ys = np.asarray(list(max_sharpes) if max_sharpes is not None else np.linspace(-0.5, 6.5, 180), dtype=float)
    sigma = max(float(sharpe_std), EPS)
    rows: list[dict[str, float]] = []
    for k in counts:
        z = (ys - float(sharpe_mean)) / sigma
        log_pdf = math.log(max(k, 1.0)) - math.log(sigma) + norm.logpdf(z) + (k - 1.0) * norm.logcdf(z)
        density = np.exp(np.clip(log_pdf, -745.0, 50.0))
        # Normalize each vertical slice so the heatmap emphasizes the conditional distribution.
        peak = float(np.nanmax(density)) if density.size else 0.0
        relative = density / peak if peak > EPS else density
        expected = expected_max_sharpe(k, sharpe_mean=sharpe_mean, sharpe_std=sharpe_std)
        for y, d, rel in zip(ys, density, relative):
            z_y = (float(y) - float(sharpe_mean)) / sigma
            cdf = float(np.exp(np.clip(k * norm.logcdf(z_y), -745.0, 0.0)))
            rows.append({
                "number_of_trials": float(k),
                "max_sharpe": float(y),
                "density": float(d),
                "log_density": float(np.log(max(float(d), np.finfo(float).tiny))),
                "relative_density": float(rel),
                "cdf_max_sharpe": cdf,
                "exceedance_probability": float(np.clip(1.0 - cdf, 0.0, 1.0)),
                "expected_max_sharpe": float(expected),
                "null_sharpe_mean": float(sharpe_mean),
                "null_sharpe_std": float(sharpe_std),
            })
    return pd.DataFrame(rows)


def adjusted_pvalues(pvalues: Iterable[float], method: str = "benjamini_hochberg") -> np.ndarray:
    """Return monotone adjusted p-values for common FWER/FDR procedures."""
    p = np.asarray(list(pvalues), dtype=float)
    p = np.clip(np.nan_to_num(p, nan=1.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    m = len(p)
    if m == 0:
        return p
    method = method.lower().replace("-", "_").replace(" ", "_")

    if method in {"bonferroni", "bonf"}:
        return np.minimum(p * m, 1.0)
    if method in {"sidak", "sidak_single_step"}:
        return np.clip(1.0 - np.power(1.0 - p, m), 0.0, 1.0)

    order = np.argsort(p)
    ranked = p[order]
    adjusted_ranked = np.empty(m, dtype=float)

    if method in {"holm", "holm_bonferroni"}:
        raw = (m - np.arange(m)) * ranked
        adjusted_ranked = np.maximum.accumulate(raw)
    elif method in {"hochberg", "hochberg_step_up"}:
        raw = (m - np.arange(m)) * ranked
        adjusted_ranked = np.minimum.accumulate(raw[::-1])[::-1]
    elif method in {"benjamini_hochberg", "bh", "fdr_bh"}:
        raw = ranked * m / np.arange(1, m + 1)
        adjusted_ranked = np.minimum.accumulate(raw[::-1])[::-1]
    elif method in {"benjamini_yekutieli", "by", "fdr_by"}:
        harmonic = float(np.sum(1.0 / np.arange(1, m + 1)))
        raw = ranked * m * harmonic / np.arange(1, m + 1)
        adjusted_ranked = np.minimum.accumulate(raw[::-1])[::-1]
    else:
        raise ValueError(f"Unknown p-value adjustment method: {method}")

    out = np.empty(m, dtype=float)
    out[order] = np.clip(adjusted_ranked, 0.0, 1.0)
    return out


def _stable_recipe_id(row: Mapping[str, Any]) -> str:
    payload = {
        "source": row.get("source"),
        "method": row.get("method"),
        "params": row.get("params"),
        "alpha": row.get("alpha"),
    }
    raw = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def recipe_level_multiple_testing(
    trial_audit: pd.DataFrame,
    *,
    score_column: str = "validation_sharpe",
    stage: str = "validation",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Test recipe-level mean score > 0 and apply FWER/FDR corrections.

    Repeated monthly evaluations of the same recipe form the observations. This is
    intentionally more conservative than treating every candidate-month as an
    independent discovery.
    """
    if trial_audit is None or trial_audit.empty or score_column not in trial_audit.columns:
        return pd.DataFrame()
    df = trial_audit.copy()
    if "stage" in df.columns:
        df = df[df["stage"].astype(str).str.lower() == stage.lower()]
    if df.empty:
        return pd.DataFrame()
    if "recipe_id" not in df.columns:
        df["recipe_id"] = df.apply(_stable_recipe_id, axis=1)
    df[score_column] = pd.to_numeric(df[score_column], errors="coerce")

    rows: list[dict[str, Any]] = []
    for recipe_id, group in df.groupby("recipe_id", sort=False):
        x = group[score_column].dropna().astype(float)
        if len(x) >= 2 and float(x.std(ddof=1)) > EPS:
            # Validation Sharpe windows overlap heavily.  A Bartlett/Newey-West
            # long-run variance test is more defensible than an iid t-statistic.
            hac = hac_mean_test(x)
            n_eff = effective_sample_size(x)
            t_stat = float(hac["z_stat"])
            p_one = float(hac["pvalue"])
            hac_se = float(hac["hac_se"])
            hac_lag = int(hac["lag"])
        else:
            n_eff = float(len(x))
            t_stat = np.nan
            hac_se = np.nan
            hac_lag = 0
            # One observation cannot establish a repeatable recipe-level discovery.
            p_one = 1.0
        first = group.iloc[0]
        rows.append({
            "recipe_id": recipe_id,
            "source": first.get("source"),
            "method": first.get("method"),
            "alpha": first.get("alpha"),
            "params": first.get("params"),
            "n_rebalances": int(len(x)),
            "effective_rebalances": float(n_eff),
            "mean_validation_sharpe": float(x.mean()) if len(x) else np.nan,
            "std_validation_sharpe": float(x.std(ddof=1)) if len(x) > 1 else np.nan,
            "t_stat": float(t_stat) if np.isfinite(t_stat) else np.nan,
            "hac_standard_error": float(hac_se) if np.isfinite(hac_se) else np.nan,
            "hac_lag": int(hac_lag),
            "raw_pvalue": float(np.clip(p_one, 0.0, 1.0)),
        })

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    p = result["raw_pvalue"].to_numpy(dtype=float)
    result["p_bonferroni"] = adjusted_pvalues(p, "bonferroni")
    result["p_sidak"] = adjusted_pvalues(p, "sidak")
    result["p_holm"] = adjusted_pvalues(p, "holm")
    result["p_hochberg"] = adjusted_pvalues(p, "hochberg")
    result["p_fdr_bh"] = adjusted_pvalues(p, "benjamini_hochberg")
    result["p_fdr_by"] = adjusted_pvalues(p, "benjamini_yekutieli")
    for col in ["raw_pvalue", "p_bonferroni", "p_sidak", "p_holm", "p_hochberg", "p_fdr_bh", "p_fdr_by"]:
        result[f"reject_{col.replace('p_', '')}"] = result[col] <= float(alpha)
    return result.sort_values(["p_fdr_by", "raw_pvalue"]).reset_index(drop=True)


def nested_selection_overfitting_proxy(
    trial_audit: pd.DataFrame,
    *,
    validation_metric: str = "validation_sharpe",
    test_metric: str = "test_sharpe",
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Rank the validation winner on the internal-test candidates for each rebalance.

    This is a nested walk-forward analogue of PBO, not the exact CSCV estimator.
    A percentile <= 0.5 means the validation winner landed in the lower half of
    the internal-test ranking.
    """
    if trial_audit is None or trial_audit.empty or "rebalance_date" not in trial_audit.columns:
        return pd.DataFrame(), {"nested_pbo_proxy": np.nan, "mean_test_rank_percentile": np.nan, "n_rebalances": 0.0}
    df = trial_audit.copy()
    if "stage" not in df.columns:
        return pd.DataFrame(), {"nested_pbo_proxy": np.nan, "mean_test_rank_percentile": np.nan, "n_rebalances": 0.0}
    rows: list[dict[str, Any]] = []
    for date, block in df.groupby("rebalance_date"):
        val = block[block["stage"].astype(str).str.lower() == "validation"].copy()
        tst = block[block["stage"].astype(str).str.lower() == "test"].copy()
        if val.empty or tst.empty or validation_metric not in val or test_metric not in tst:
            continue
        val[validation_metric] = pd.to_numeric(val[validation_metric], errors="coerce")
        tst[test_metric] = pd.to_numeric(tst[test_metric], errors="coerce")
        val = val.dropna(subset=[validation_metric])
        tst = tst.dropna(subset=[test_metric])
        if val.empty or tst.empty:
            continue
        winner_id = str(val.sort_values(validation_metric, ascending=False).iloc[0]["candidate_id"])
        winner = tst[tst["candidate_id"].astype(str) == winner_id]
        if winner.empty:
            continue
        ranks = tst[test_metric].rank(method="average", pct=True, ascending=True)
        pct = float(ranks.loc[winner.index[0]])
        clipped = float(np.clip(pct, 1e-6, 1.0 - 1e-6))
        rows.append({
            "rebalance_date": date,
            "validation_winner": winner_id,
            "validation_metric": float(val.sort_values(validation_metric, ascending=False).iloc[0][validation_metric]),
            "internal_test_metric": float(winner.iloc[0][test_metric]),
            "internal_test_rank_percentile": pct,
            "logit_rank": float(math.log(clipped / (1.0 - clipped))),
            "overfit_flag": bool(pct <= 0.5),
            "n_internal_test_candidates": int(len(tst)),
        })
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, {"nested_pbo_proxy": np.nan, "mean_test_rank_percentile": np.nan, "n_rebalances": 0.0}
    summary = {
        "nested_pbo_proxy": float(detail["overfit_flag"].mean()),
        "mean_test_rank_percentile": float(detail["internal_test_rank_percentile"].mean()),
        "median_logit_rank": float(detail["logit_rank"].median()),
        "n_rebalances": float(len(detail)),
    }
    return detail, summary


def _latest_timestamp(value: Any) -> pd.Timestamp | pd.NaT:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return pd.NaT
    if isinstance(value, (list, tuple)):
        vals = [pd.to_datetime(v, errors="coerce") for v in value]
        vals = [v for v in vals if not pd.isna(v)]
        return max(vals) if vals else pd.NaT
    if isinstance(value, str) and value.strip().startswith("["):
        try:
            return _latest_timestamp(json.loads(value))
        except Exception:
            pass
    return pd.to_datetime(value, errors="coerce")



def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y", "on"}

def causality_audit(selection_audit: pd.DataFrame) -> pd.DataFrame:
    """Check that every data split ends no later than its rebalance decision date."""
    if selection_audit is None or selection_audit.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for _, row in selection_audit.iterrows():
        decision = pd.to_datetime(row.get("date", row.get("rebalance_date")), errors="coerce")
        history_end = _latest_timestamp(row.get("history_end"))
        train_end = _latest_timestamp(row.get("train_end"))
        validation_end = _latest_timestamp(row.get("validation_end", row.get("validation_months")))
        test_end = _latest_timestamp(row.get("internal_test_end", row.get("test_months")))
        checks = {
            "history_before_decision": bool(pd.notna(decision) and (pd.isna(history_end) or history_end <= decision)),
            "train_before_decision": bool(pd.notna(decision) and (pd.isna(train_end) or train_end <= decision)),
            "validation_before_decision": bool(pd.notna(decision) and (pd.isna(validation_end) or validation_end <= decision)),
            "internal_test_before_decision": bool(pd.notna(decision) and (pd.isna(test_end) or test_end <= decision)),
            "nested_engine_used": _as_bool(row.get("engine_used", False)),
            "no_fallback": bool(
                pd.isna(row.get("fallback_reason", ""))
                or not str(row.get("fallback_reason", "")).strip()
            ),
        }
        rows.append({
            "rebalance_date": decision,
            "history_end": history_end,
            "train_end": train_end,
            "validation_end": validation_end,
            "internal_test_end": test_end,
            **checks,
            "causality_pass": bool(all(checks[k] for k in [
                "history_before_decision", "train_before_decision", "validation_before_decision", "internal_test_before_decision"
            ])),
            "full_process_pass": bool(all(checks.values())),
        })
    return pd.DataFrame(rows)
