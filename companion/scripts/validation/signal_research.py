"""Publication-oriented dependence and incremental-information research for HDRC signals.

The module implements the diagnostics requested by the theoretical manuscripts:

* Pearson/Spearman/Kendall and mutual-information dependence matrices;
* rolling and regime-ready correlations, lead/lag profiles and defensive-event overlap;
* ADF/KPSS stationarity screening and conditional Engle-Granger/Johansen tests;
* distribution-family comparisons (AIC/BIC plus parametric-bootstrap KS goodness-of-fit);
* signal-set relationships through RV coefficients and first canonical correlation;
* Frisch-Waugh-Lovell diagnostics, HAC forecast-encompassing tests and causal expanding-window
  forward selection with a causal mean-only baseline and stop-on-no-improvement;
* exhaustive timer-subset overlay research with matched-average-exposure controls and PSR/DSR/FDR;
* timer-pair response surfaces plus empirical conditional-risk heatmaps.

Cointegration is deliberately *not* forced on bounded exposure timers.  It is run only
when unbounded precursor series look I(1): level ADF fails to reject a unit root and the
first difference rejects it.  This follows the evidence boundary stated in the supplied
HDRC methodology documents.

The output is exploratory research evidence.  Any subset chosen after inspecting these
artifacts belongs in the DSR/FDR research ledger before a confirmatory holdout is opened.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import warnings
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cross_decomposition import CCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint, kpss
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from scripts.validation.false_strategy import (
    adjusted_pvalues,
    deflated_sharpe_ratio,
    effective_number_of_trials,
    probabilistic_sharpe_ratio,
    regularize_trial_correlation,
)

EPS = 1e-12

TIMER_COLUMNS = (
    "backward_vol_timer",
    "forward_risk_timer",
    "forward_cvar_timer",
    "forward_cdar_timer",
    "conviction_timer",
)


def _preferred_timer_columns(panel: pd.DataFrame) -> list[str]:
    """Use one active forward-risk timer to avoid double-counting its CVaR/CDaR aliases."""
    cols: list[str] = []
    if "backward_vol_timer" in panel.columns:
        cols.append("backward_vol_timer")
    forward = None
    for c in ("forward_risk_timer", "forward_cdar_timer", "forward_cvar_timer"):
        if c in panel.columns:
            x = pd.to_numeric(panel[c], errors="coerce")
            if x.notna().sum() >= 4 and float(x.std(ddof=1) or 0.0) > EPS:
                forward = c
                break
    if forward is not None:
        cols.append(forward)
    if "conviction_timer" in panel.columns:
        cols.append("conviction_timer")
    return cols
NON_SIGNAL_COLUMNS = {
    "date", "rebalance_date", "comments", "overlay_combination", "smart_signal_error",
    "smart_position_sizing_applied", "signal_centering_applied", "smart_signal_table",
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if np.isfinite(v) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(value)
    return value


def _read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kwargs)
    except Exception:
        return pd.DataFrame()


def _date_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out
    date_col = next((c for c in ("date", "Date", "rebalance_date") if c in out.columns), None)
    if date_col is None and len(out.columns):
        first = str(out.columns[0])
        if first.startswith("Unnamed") or first.lower() in {"index", "timestamp"}:
            parsed = pd.to_datetime(out.iloc[:, 0], errors="coerce")
            if parsed.notna().mean() >= 0.8:
                date_col = out.columns[0]
    if date_col is not None:
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
        out = out.dropna(subset=[date_col]).set_index(date_col)
    elif not isinstance(out.index, pd.DatetimeIndex):
        idx = pd.to_datetime(out.index, errors="coerce")
        out.index = idx
        out = out[~out.index.isna()]
    return out.sort_index()


def _flatten_asset_signal_json(smart: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-ETF signal tables into causal monthly cross-sectional summaries."""
    if smart.empty or "smart_signal_table" not in smart.columns:
        return pd.DataFrame(index=smart.index)
    rows: list[dict[str, float]] = []
    for dt, raw in smart["smart_signal_table"].items():
        try:
            records = json.loads(raw) if isinstance(raw, str) else raw
            table = pd.DataFrame(records)
        except Exception:
            continue
        numeric = table.drop(columns=[c for c in ("ticker", "index") if c in table], errors="ignore").apply(
            pd.to_numeric, errors="coerce"
        )
        rec: dict[str, float] = {"date": dt}
        for col in numeric.columns:
            vals = numeric[col].replace([np.inf, -np.inf], np.nan).dropna()
            if vals.empty:
                continue
            rec[f"asset_mean_{col}"] = float(vals.mean())
            rec[f"asset_median_{col}"] = float(vals.median())
            rec[f"asset_dispersion_{col}"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        rows.append(rec)
    if not rows:
        return pd.DataFrame(index=smart.index)
    out = pd.DataFrame(rows).set_index("date").sort_index()
    out.index = pd.to_datetime(out.index)
    return out


def _next_event_values(signal_dates: pd.DatetimeIndex, series: pd.Series) -> pd.Series:
    """Map each signal date to the first strictly later realized return observation."""
    s = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index, errors="coerce")
        s = s[~s.index.isna()].sort_index()
    idx = s.index.to_numpy(dtype="datetime64[ns]")
    vals = s.to_numpy(dtype=float)
    out = []
    for dt in signal_dates:
        pos = int(np.searchsorted(idx, np.datetime64(dt), side="right"))
        out.append(vals[pos] if pos < len(vals) else np.nan)
    return pd.Series(out, index=signal_dates, dtype=float)


def load_signal_panel(results_dir: str | Path) -> pd.DataFrame:
    """Build a monthly signal panel from standard pipeline artifacts."""
    folder = Path(results_dir)
    smart = _date_index(_read_csv(folder / "smart_rebalance_audit.csv"))
    forecast = _date_index(_read_csv(folder / "forecast_risk.csv"))
    selection = _date_index(_read_csv(folder / "selection_audit.csv"))

    pieces: list[pd.DataFrame] = []
    if not smart.empty:
        smart_numeric = smart.drop(columns=[c for c in NON_SIGNAL_COLUMNS if c in smart.columns], errors="ignore").apply(
            pd.to_numeric, errors="coerce"
        )
        pieces.append(smart_numeric)
        agg = _flatten_asset_signal_json(smart)
        if not agg.empty:
            pieces.append(agg)
    if not forecast.empty:
        f = forecast.apply(pd.to_numeric, errors="coerce")
        f = f.rename(columns={c: f"forecast_{c}" for c in f.columns if c in smart.columns})
        pieces.append(f)
    if not selection.empty:
        keep = [c for c in selection.columns if any(k in c.lower() for k in ("alpha", "beta", "lambda", "regime", "score"))]
        if keep:
            sel = selection[keep].apply(pd.to_numeric, errors="coerce")
            sel = sel.rename(columns={c: f"state_{c}" for c in sel.columns if c in smart.columns or c in forecast.columns})
            pieces.append(sel)

    if not pieces:
        raise FileNotFoundError(f"No smart_rebalance_audit.csv / forecast_risk.csv / usable selection_audit.csv in {folder}")
    panel = pd.concat(pieces, axis=1).sort_index()
    panel = panel.loc[:, ~panel.columns.duplicated()]
    panel = panel.replace([np.inf, -np.inf], np.nan)

    pnl = _date_index(_read_csv(folder / "pnl.csv"))
    if not pnl.empty and "Returns" in pnl.columns:
        next_ret = _next_event_values(panel.index, pnl["Returns"])
        panel["next_realized_return"] = next_ret
        panel["next_downside_loss"] = (-next_ret).clip(lower=0.0)
        panel["next_abs_return"] = next_ret.abs()
        panel["next_squared_return"] = next_ret.pow(2)
        # Reconstruct the underlying risky-sleeve return when the managed return is
        # a cash-overlay mixture and the original overlay fraction is available.
        overlay_col = "overlay_fraction" if "overlay_fraction" in panel else None
        if overlay_col is not None:
            k = pd.to_numeric(panel[overlay_col], errors="coerce")
            rf_m = 0.02 / 12.0
            panel["reconstructed_risky_return"] = (next_ret - (1.0 - k) * rf_m) / k.where(k.abs() > EPS)
    return panel


def _numeric_signal_columns(panel: pd.DataFrame) -> list[str]:
    cols = []
    for c in panel.columns:
        if c.startswith("next_") or c in {"reconstructed_risky_return"}:
            continue
        x = pd.to_numeric(panel[c], errors="coerce")
        if x.notna().sum() >= 8 and float(x.std(ddof=1) or 0.0) > EPS:
            cols.append(c)
    return cols


def pairwise_dependence(panel: pd.DataFrame, signals: list[str]) -> dict[str, pd.DataFrame]:
    x = panel[signals].apply(pd.to_numeric, errors="coerce")
    pearson = x.corr(method="pearson", min_periods=6)
    spearman = x.corr(method="spearman", min_periods=6)
    kendall = x.corr(method="kendall", min_periods=6)

    mi = pd.DataFrame(np.nan, index=signals, columns=signals, dtype=float)
    for a in signals:
        mi.loc[a, a] = 1.0
        for b in signals:
            if a >= b:
                continue
            pair = x[[a, b]].dropna()
            if len(pair) < 10:
                continue
            try:
                vab = float(mutual_info_regression(pair[[a]], pair[b], random_state=42)[0])
                vba = float(mutual_info_regression(pair[[b]], pair[a], random_state=42)[0])
                val = 0.5 * (vab + vba)
            except Exception:
                val = np.nan
            mi.loc[a, b] = mi.loc[b, a] = val
    return {"pearson": pearson, "spearman": spearman, "kendall": kendall, "mutual_information": mi}


def rolling_correlations(panel: pd.DataFrame, signals: list[str], window: int = 24) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for a, b in itertools.combinations(signals, 2):
        r = panel[a].rolling(window, min_periods=max(8, window // 2)).corr(panel[b])
        for dt, val in r.dropna().items():
            rows.append({"date": dt, "signal_a": a, "signal_b": b, "rolling_pearson": float(val), "window": window})
    return pd.DataFrame(rows)


def lead_lag_analysis(panel: pd.DataFrame, signals: list[str], max_lag: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cross-correlation Corr(a_t, b_{t+lag}) for all signal pairs.

    The implementation works on a pre-extracted NumPy matrix so the full K x K x
    lag cube remains practical for dozens of signals.
    """
    arr = panel[signals].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    best = pd.DataFrame(np.nan, index=signals, columns=signals, dtype=float)
    nobs = arr.shape[0]
    for ia, a in enumerate(signals):
        xa0 = arr[:, ia]
        for ib, b in enumerate(signals):
            xb0 = arr[:, ib]
            best_abs, best_corr, best_lag = -1.0, np.nan, 0
            for lag in range(-max_lag, max_lag + 1):
                if lag > 0:
                    xa, xb = xa0[:-lag], xb0[lag:]
                elif lag < 0:
                    h = -lag
                    xa, xb = xa0[h:], xb0[:-h]
                else:
                    xa, xb = xa0, xb0
                mask = np.isfinite(xa) & np.isfinite(xb)
                n = int(mask.sum())
                if n >= 8:
                    xx, yy = xa[mask], xb[mask]
                    sx, sy = float(np.std(xx, ddof=1)), float(np.std(yy, ddof=1))
                    corr = float(np.corrcoef(xx, yy)[0, 1]) if sx > EPS and sy > EPS else np.nan
                else:
                    corr = np.nan
                rows.append({"signal_a": a, "signal_b": b, "lag": lag, "correlation": corr, "n": n})
                if np.isfinite(corr) and abs(corr) > best_abs:
                    best_abs, best_corr, best_lag = abs(corr), corr, lag
            best.loc[a, b] = best_lag if np.isfinite(best_corr) else np.nan
    return pd.DataFrame(rows), best


def event_overlap(panel: pd.DataFrame, signals: list[str], quantile: float = 0.20) -> pd.DataFrame:
    events: dict[str, pd.Series] = {}
    for c in signals:
        x = pd.to_numeric(panel[c], errors="coerce")
        threshold = float(x.quantile(quantile))
        events[c] = x <= threshold
    out = pd.DataFrame(np.nan, index=signals, columns=signals, dtype=float)
    for a, b in itertools.product(signals, repeat=2):
        va = events[a].fillna(False)
        vb = events[b].fillna(False)
        union = int((va | vb).sum())
        inter = int((va & vb).sum())
        out.loc[a, b] = inter / union if union else np.nan
    return out


def _is_bounded_timer(x: pd.Series) -> bool:
    z = pd.to_numeric(x, errors="coerce").dropna()
    if z.empty:
        return False
    return bool(z.min() >= -1e-9 and z.max() <= 1.0 + 1e-9)


def stationarity_tests(panel: pd.DataFrame, signals: list[str]) -> pd.DataFrame:
    rows = []
    for c in signals:
        x = pd.to_numeric(panel[c], errors="coerce").dropna()
        row: dict[str, Any] = {"signal": c, "n": int(len(x)), "bounded_0_1": _is_bounded_timer(x)}
        if len(x) < 16 or float(x.std(ddof=1)) <= EPS:
            row.update({"adf_stat": np.nan, "adf_pvalue": np.nan, "adf_diff_pvalue": np.nan, "kpss_pvalue": np.nan, "integration_order": "insufficient"})
        else:
            try:
                row["adf_stat"], row["adf_pvalue"] = map(float, adfuller(x, autolag="AIC")[:2])
            except Exception:
                row["adf_stat"], row["adf_pvalue"] = np.nan, np.nan
            try:
                dx = x.diff().dropna()
                row["adf_diff_pvalue"] = float(adfuller(dx, autolag="AIC")[1]) if len(dx) >= 15 else np.nan
            except Exception:
                row["adf_diff_pvalue"] = np.nan
            try:
                # statsmodels emits an InterpolationWarning when the statistic lies
                # outside its tabulated p-value range.  The returned boundary p-value
                # is still useful for our I(0)/I(1) gate, so keep it and silence only
                # that expected diagnostic warning.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", InterpolationWarning)
                    row["kpss_pvalue"] = float(kpss(x, regression="c", nlags="auto")[1])
            except Exception:
                row["kpss_pvalue"] = np.nan
            if row["bounded_0_1"] and np.isfinite(row["adf_pvalue"]) and row["adf_pvalue"] <= 0.05:
                order = "I(0)-bounded"
            elif np.isfinite(row["adf_pvalue"]) and row["adf_pvalue"] <= 0.05:
                order = "I(0)"
            elif np.isfinite(row["adf_diff_pvalue"]) and row["adf_diff_pvalue"] <= 0.05:
                order = "I(1)"
            else:
                order = "undetermined"
            row["integration_order"] = order
        rows.append(row)
    return pd.DataFrame(rows)


def cointegration_tests(panel: pd.DataFrame, stationarity: pd.DataFrame, signals: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    order = stationarity.set_index("signal")["integration_order"].to_dict() if not stationarity.empty else {}
    eligible = [c for c in signals if order.get(c) == "I(1)" and not _is_bounded_timer(panel[c])]
    out = pd.DataFrame(np.nan, index=signals, columns=signals, dtype=float)
    rows: list[dict[str, Any]] = []
    for a, b in itertools.combinations(eligible, 2):
        pair = panel[[a, b]].dropna()
        if len(pair) < 20:
            continue
        try:
            score, pvalue, _ = coint(pair[a], pair[b], trend="c", autolag="aic")
            out.loc[a, b] = out.loc[b, a] = float(pvalue)
            rows.append({"signal_a": a, "signal_b": b, "engle_granger_stat": float(score), "pvalue": float(pvalue), "n": int(len(pair))})
        except Exception:
            continue
    johansen: dict[str, Any] = {"eligible_i1_signals": eligible, "performed": False}
    if 2 <= len(eligible) <= 8:
        joint = panel[eligible].dropna()
        if len(joint) >= max(30, 5 * len(eligible)):
            try:
                jres = coint_johansen(joint, det_order=0, k_ar_diff=1)
                johansen = {
                    "eligible_i1_signals": eligible,
                    "performed": True,
                    "trace_statistics": [float(v) for v in jres.lr1],
                    "trace_critical_values_95": [float(v) for v in jres.cvt[:, 1]],
                    "max_eigen_statistics": [float(v) for v in jres.lr2],
                    "max_eigen_critical_values_95": [float(v) for v in jres.cvm[:, 1]],
                }
            except Exception as exc:
                johansen["error"] = str(exc)
    return pd.DataFrame(rows), {"matrix": out, "summary": johansen}


def distribution_fits(
    panel: pd.DataFrame,
    signals: list[str],
    *,
    bootstrap_samples: int = 99,
    random_state: int = 42,
) -> pd.DataFrame:
    """Fit common parametric families and report AIC/BIC plus calibrated GOF.

    The ordinary KS p-value after estimating parameters on the same observations is
    not distribution-free.  We therefore retain its statistic as a descriptive
    distance and compute a Monte-Carlo/parametric-bootstrap KS p-value only for the
    best-AIC family of each signal.  This avoids interpreting the naive fitted-KS
    p-value as formal evidence of fit.
    """
    candidates: dict[str, tuple[Any, dict[str, float]]] = {
        "normal": (stats.norm, {}),
        "student_t": (stats.t, {}),
        "laplace": (stats.laplace, {}),
        "logistic": (stats.logistic, {}),
        "skew_normal": (stats.skewnorm, {}),
    }
    rows: list[dict[str, Any]] = []
    data_cache: dict[str, np.ndarray] = {}
    spec_cache: dict[tuple[str, str], tuple[Any, dict[str, float]]] = {}
    for c in signals:
        x = pd.to_numeric(panel[c], errors="coerce").dropna().to_numpy(dtype=float)
        if len(x) < 12 or np.std(x, ddof=1) <= EPS:
            continue
        data_cache[c] = x
        local = dict(candidates)
        if np.all(x > 0):
            # Positive-only families are easier to interpret with a zero location.
            local["gamma"] = (stats.gamma, {"floc": 0.0})
            local["lognormal"] = (stats.lognorm, {"floc": 0.0})
        bounded = np.min(x) >= -1e-10 and np.max(x) <= 1.0 + 1e-10
        if bounded:
            local["beta_[0,1]"] = (stats.beta, {"floc": 0.0, "fscale": 1.0})

        for name, (dist, fit_kw) in local.items():
            try:
                fit_x = np.clip(x, 1e-6, 1.0 - 1e-6) if name == "beta_[0,1]" else x
                params = dist.fit(fit_x, **fit_kw)
                ll = float(np.sum(dist.logpdf(fit_x, *params)))
                # scipy returns all fitted parameters, including fixed loc/scale.
                fixed = int("floc" in fit_kw) + int("fscale" in fit_kw)
                k_free = max(len(params) - fixed, 1)
                aic = 2.0 * k_free - 2.0 * ll
                bic = math.log(len(fit_x)) * k_free - 2.0 * ll
                ks_stat = float(stats.kstest(fit_x, dist.cdf, args=params).statistic)
                rows.append({
                    "signal": c,
                    "distribution": name,
                    "aic": aic,
                    "bic": bic,
                    "ks_stat": ks_stat,
                    "ks_pvalue_naive_fitted": np.nan,
                    "ks_pvalue_parametric_bootstrap": np.nan,
                    "bootstrap_samples": int(bootstrap_samples),
                    "n_params_free": int(k_free),
                    "params": json.dumps([float(v) for v in params]),
                })
                spec_cache[(c, name)] = (dist, fit_kw)
            except Exception:
                pass

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["aic_rank_within_signal"] = out.groupby("signal")["aic"].rank(method="min")
    out["best_by_aic"] = out["aic_rank_within_signal"] == 1

    # Calibrate only the selected family for each signal.  scipy.goodness_of_fit
    # refits unknown parameters in every Monte Carlo sample, which is the relevant
    # correction for a distribution chosen after parameter estimation.
    rng = np.random.default_rng(random_state)
    for idx, row in out[out["best_by_aic"]].iterrows():
        c = str(row["signal"]); name = str(row["distribution"])
        x = data_cache[c]
        fit_x = np.clip(x, 1e-6, 1.0 - 1e-6) if name == "beta_[0,1]" else x
        dist, fit_kw = spec_cache[(c, name)]
        known: dict[str, float] = {}
        if "floc" in fit_kw:
            known["loc"] = float(fit_kw["floc"])
        if "fscale" in fit_kw:
            known["scale"] = float(fit_kw["fscale"])
        try:
            result = stats.goodness_of_fit(
                dist,
                fit_x,
                known_params=known or None,
                statistic="ks",
                n_mc_samples=max(int(bootstrap_samples), 19),
                rng=rng,
            )
            out.loc[idx, "ks_pvalue_parametric_bootstrap"] = float(result.pvalue)
            out.loc[idx, "ks_stat"] = float(result.statistic)
        except Exception:
            pass
    return out.sort_values(["signal", "aic"]).reset_index(drop=True)


def _rv_coefficient(x: np.ndarray, y: np.ndarray) -> float:
    xc = x - x.mean(axis=0, keepdims=True)
    yc = y - y.mean(axis=0, keepdims=True)
    sxy = xc.T @ yc
    sxx = xc.T @ xc
    syy = yc.T @ yc
    denom = math.sqrt(float(np.trace(sxx @ sxx)) * float(np.trace(syy @ syy)))
    return float(np.trace(sxy @ sxy.T) / denom) if denom > EPS else np.nan


def signal_group_relationships(panel: pd.DataFrame, signals: list[str]) -> pd.DataFrame:
    groups: dict[str, list[str]] = {
        "timers": [c for c in signals if "timer" in c and c != "overlay_fraction"],
        "asset_quality": [c for c in signals if c.startswith("asset_mean_") or c.startswith("asset_median_")],
        "model_risk": [c for c in signals if any(k in c for k in ("cvar", "vol_model", "scenario_mean")) and "timer" not in c],
        "state": [c for c in signals if c.startswith("state_")],
    }
    groups = {k: v for k, v in groups.items() if v}
    rows = []
    for ga, gb in itertools.combinations(groups, 2):
        cols_a, cols_b = groups[ga], groups[gb]
        data = panel[cols_a + cols_b].dropna()
        if len(data) < max(12, len(cols_a) + len(cols_b) + 2):
            continue
        xa = StandardScaler().fit_transform(data[cols_a])
        xb = StandardScaler().fit_transform(data[cols_b])
        try:
            cca = CCA(n_components=1, max_iter=1000)
            ua, ub = cca.fit_transform(xa, xb)
            can = float(np.corrcoef(ua[:, 0], ub[:, 0])[0, 1])
        except Exception:
            can = np.nan
        rows.append({"group_a": ga, "group_b": gb, "n": len(data), "features_a": json.dumps(cols_a), "features_b": json.dumps(cols_b), "rv_coefficient": _rv_coefficient(xa, xb), "first_canonical_correlation": can})
    return pd.DataFrame(rows)


def group_cointegration_relationships(
    panel: pd.DataFrame,
    stationarity: pd.DataFrame,
    signals: list[str],
    *,
    max_system_signals: int = 8,
) -> pd.DataFrame:
    """Johansen rank diagnostics for pairs of economically defined signal groups.

    Cointegration is only meaningful for I(1) precursors.  Bounded exposure timers
    and stationary diagnostics are therefore excluded rather than forced into a
    cointegration table simply because the user requested every pair.
    """
    order = stationarity.set_index("signal")["integration_order"].to_dict() if not stationarity.empty else {}
    groups: dict[str, list[str]] = {
        "timers": [c for c in signals if "timer" in c],
        "asset_quality": [c for c in signals if c.startswith("asset_mean_")],
        "model_risk": [c for c in signals if any(k in c for k in ("cvar", "vol_model", "scenario_mean")) and "timer" not in c],
        "state": [c for c in signals if c.startswith("state_")],
    }
    groups = {g: [c for c in cols if order.get(c) == "I(1)" and not _is_bounded_timer(panel[c])] for g, cols in groups.items()}
    rows: list[dict[str, Any]] = []
    for ga, gb in itertools.combinations(groups, 2):
        cols = list(dict.fromkeys(groups[ga] + groups[gb]))
        if len(groups[ga]) == 0 or len(groups[gb]) == 0 or len(cols) < 2:
            continue
        if len(cols) > max_system_signals:
            cols = cols[:max_system_signals]
        joint = panel[cols].apply(pd.to_numeric, errors="coerce").dropna()
        if len(joint) < max(30, 5 * len(cols)):
            continue
        try:
            res = coint_johansen(joint, det_order=0, k_ar_diff=1)
            rank95 = int(np.sum(np.asarray(res.lr1) > np.asarray(res.cvt[:, 1])))
            rows.append({
                "group_a": ga, "group_b": gb, "signals": json.dumps(cols),
                "n": int(len(joint)), "system_dimension": int(len(cols)),
                "johansen_trace_rank_95": rank95,
                "trace_statistics": json.dumps([float(v) for v in res.lr1]),
                "trace_critical_values_95": json.dumps([float(v) for v in res.cvt[:, 1]]),
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


def _expanding_oos_mse(
    data: pd.DataFrame, features: list[str], target: str, min_train: int = 24, n_splits: int = 4,
) -> tuple[float, int]:
    """Chronological expanding-window CV MSE with an optional mean-only baseline."""
    cols = features + [target]
    d = data[cols].dropna().copy()
    if len(d) < min_train + 8:
        return np.nan, 0
    remaining = len(d) - min_train
    block = max(2, remaining // max(n_splits, 1))
    preds: list[float] = []
    actual: list[float] = []
    train_end = min_train
    while train_end < len(d):
        test_end = min(len(d), train_end + block)
        tr = d.iloc[:train_end]
        te = d.iloc[train_end:test_end]
        if te.empty:
            break
        if features:
            scaler = StandardScaler().fit(tr[features])
            model = Ridge(alpha=1.0).fit(scaler.transform(tr[features]), tr[target])
            pred = model.predict(scaler.transform(te[features]))
        else:
            pred = np.repeat(float(tr[target].mean()), len(te))
        preds.extend(np.asarray(pred, dtype=float).tolist())
        actual.extend(te[target].to_numpy(dtype=float).tolist())
        train_end = test_end
    if not actual:
        return np.nan, 0
    return float(np.mean((np.asarray(actual) - np.asarray(preds)) ** 2)), len(actual)


def incremental_signal_research(
    panel: pd.DataFrame, signals: list[str], target: str = "next_abs_return",
    min_train: int = 24, max_steps: int = 12, min_improvement: float = 0.0,
) -> pd.DataFrame:
    """Greedy one-by-one signal addition using causal expanding-window OOS MSE.

    The baseline is itself causal (an expanding historical-mean forecast).  The
    search stops when the best remaining signal no longer improves OOS MSE, so the
    output does not mechanically keep adding weak variables merely because a fixed
    maximum number of steps was requested.
    """
    if target not in panel:
        return pd.DataFrame()
    remaining = [c for c in signals if panel[c].notna().sum() >= min_train + 8]
    accepted: list[str] = []
    rows: list[dict[str, Any]] = []
    current_mse, baseline_n = _expanding_oos_mse(panel, [], target, min_train=min_train)
    if not np.isfinite(current_mse):
        return pd.DataFrame()
    rows.append({
        "step": 0, "added_signal": "__causal_mean_baseline__", "accepted_set": "[]",
        "oos_mse": float(current_mse), "marginal_oos_mse_improvement": np.nan,
        "relative_mse_improvement": 0.0, "oos_predictions": int(baseline_n),
        "fwl_population_mse_gain_descriptive": np.nan, "accepted": True,
    })
    step = 1
    while remaining and step <= max_steps:
        candidates = []
        for c in remaining:
            feats = accepted + [c]
            mse, n = _expanding_oos_mse(panel, feats, target, min_train=min_train)
            if np.isfinite(mse):
                d = panel[feats + [target]].dropna()
                if accepted and len(d) > len(accepted) + 5:
                    z = d[accepted].to_numpy(dtype=float)
                    yr = d[target].to_numpy(dtype=float) - LinearRegression().fit(z, d[target]).predict(z)
                    xr = d[c].to_numpy(dtype=float) - LinearRegression().fit(z, d[c]).predict(z)
                else:
                    yr = d[target].to_numpy(dtype=float) - float(d[target].mean())
                    xr = d[c].to_numpy(dtype=float) - float(d[c].mean())
                fwl = float(np.cov(yr, xr, ddof=1)[0, 1] ** 2 / max(np.var(xr, ddof=1), EPS)) if len(d) > 2 else np.nan
                candidates.append((mse, c, n, fwl))
        if not candidates:
            break
        mse, chosen, n, fwl = min(candidates, key=lambda z: z[0])
        improvement = float(current_mse - mse)
        accepted_flag = bool(improvement > float(min_improvement) + EPS)
        rows.append({
            "step": step, "added_signal": chosen,
            "accepted_set": json.dumps(accepted + ([chosen] if accepted_flag else [])),
            "oos_mse": float(mse), "marginal_oos_mse_improvement": improvement,
            "relative_mse_improvement": float(improvement / current_mse) if current_mse > EPS else np.nan,
            "oos_predictions": int(n), "fwl_population_mse_gain_descriptive": fwl,
            "accepted": accepted_flag,
        })
        if not accepted_flag:
            break
        accepted.append(chosen)
        remaining.remove(chosen)
        current_mse = float(mse)
        step += 1
    return pd.DataFrame(rows)


def partial_information_tests(
    panel: pd.DataFrame,
    signals: list[str],
    *,
    target: str = "next_abs_return",
    max_lag: int = 3,
    max_signals: int = 20,
) -> pd.DataFrame:
    """HAC residual/encompassing tests for pairwise incremental signal information.

    For each ordered pair (a, b), estimate target ~ a + b with standardized signals
    and report whether b has incremental information conditional on a.  These are
    descriptive/full-sample econometric tests; the expanding OOS MSE search remains
    the causal prediction diagnostic.
    """
    cols = [c for c in signals if c in panel.columns][:max_signals]
    rows: list[dict[str, Any]] = []
    if target not in panel:
        return pd.DataFrame()
    for a, b in itertools.permutations(cols, 2):
        d = panel[[a, b, target]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(d) < 18:
            continue
        x = d[[a, b]].to_numpy(dtype=float)
        x = StandardScaler().fit_transform(x)
        X = sm.add_constant(x, has_constant="add")
        try:
            fit = sm.OLS(d[target].to_numpy(dtype=float), X).fit(cov_type="HAC", cov_kwds={"maxlags": min(max_lag, max(1, len(d)//5))})
            rows.append({
                "target": target, "conditioning_signal": a, "incremental_signal": b,
                "n": int(len(d)), "coef_incremental_standardized": float(fit.params[2]),
                "hac_t_incremental": float(fit.tvalues[2]), "raw_pvalue_incremental": float(fit.pvalues[2]),
                "adjusted_r2": float(fit.rsquared_adj),
            })
        except Exception:
            continue
    out = pd.DataFrame(rows)
    if not out.empty:
        p = out["raw_pvalue_incremental"].to_numpy(dtype=float)
        out["p_fdr_bh"] = adjusted_pvalues(p, "benjamini_hochberg")
        out["p_fdr_by"] = adjusted_pvalues(p, "benjamini_yekutieli")
        out["reject_fdr_by_5pct"] = out["p_fdr_by"] <= 0.05
    return out


def _performance_metrics(r: pd.Series, periods_per_year: int = 12, rf_annual: float = 0.02) -> dict[str, float]:
    x = pd.to_numeric(r, errors="coerce").dropna().astype(float)
    if len(x) < 2:
        return {k: np.nan for k in ("annualized_return", "annualized_volatility", "sharpe", "sortino", "max_drawdown", "cvar95")}
    years = len(x) / periods_per_year
    total = float((1.0 + x).prod() - 1.0)
    ann = float((1.0 + total) ** (1.0 / years) - 1.0) if years > 0 and total > -1 else np.nan
    vol = float(x.std(ddof=1) * math.sqrt(periods_per_year))
    ex = x - rf_annual / periods_per_year
    sharpe = float(ex.mean() / ex.std(ddof=1) * math.sqrt(periods_per_year)) if ex.std(ddof=1) > EPS else np.nan
    neg = x[x < 0]
    downside = float(neg.std(ddof=1) * math.sqrt(periods_per_year)) if len(neg) > 1 else np.nan
    sortino = float((ann - rf_annual) / downside) if np.isfinite(downside) and downside > EPS else np.nan
    eq = (1.0 + x).cumprod()
    mdd = float((eq / eq.cummax() - 1.0).min())
    q = float(x.quantile(0.05)); cvar = float(x[x <= q].mean())
    return {"annualized_return": ann, "annualized_volatility": vol, "sharpe": sharpe, "sortino": sortino, "max_drawdown": mdd, "cvar95": cvar}


def _match_mean_exposure(k: pd.Series, target_mean: float, floor: float = 0.30) -> pd.Series:
    base = pd.to_numeric(k, errors="coerce").fillna(1.0).clip(floor, 1.0)
    lo, hi = 0.0, 10.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        m = float((mid * base).clip(floor, 1.0).mean())
        if m < target_mean:
            lo = mid
        else:
            hi = mid
    return (0.5 * (lo + hi) * base).clip(floor, 1.0)


def _combine_timers(t: pd.DataFrame, mode: str, floor: float) -> pd.Series:
    z = t.apply(pd.to_numeric, errors="coerce").clip(floor, 1.0)
    if mode == "geometric_mean":
        return np.exp(np.log(z.clip(lower=EPS)).mean(axis=1)).clip(floor, 1.0)
    if mode == "product":
        return z.prod(axis=1).clip(floor, 1.0)
    raise ValueError(f"Unknown timer combination: {mode}")


def timer_subset_research(
    panel: pd.DataFrame,
    timer_cols: list[str] | None = None,
    floor: float = 0.30,
    rf_annual: float = 0.02,
) -> pd.DataFrame:
    """Exhaustive timer subsets with matched-exposure and selection-aware inference."""
    timer_cols = timer_cols or _preferred_timer_columns(panel)
    timer_cols = [c for c in timer_cols if c in panel.columns]
    if not timer_cols or "reconstructed_risky_return" not in panel:
        return pd.DataFrame()
    if len(timer_cols) > 10:
        timer_cols = timer_cols[:10]
    risky = pd.to_numeric(panel["reconstructed_risky_return"], errors="coerce")
    ref = pd.to_numeric(panel.get("overlay_fraction", pd.Series(1.0, index=panel.index)), errors="coerce").clip(floor, 1.0)
    target_mean = float(ref.mean())
    rows: list[dict[str, Any]] = []
    return_paths: dict[str, pd.Series] = {}
    row_id = 0
    for size in range(1, len(timer_cols) + 1):
        for subset in itertools.combinations(timer_cols, size):
            t = panel[list(subset)]
            for mode in ("geometric_mean", "product"):
                k = _combine_timers(t, mode, floor)
                for matched in (False, True):
                    kk = _match_mean_exposure(k, target_mean, floor=floor) if matched else k
                    ret = kk * risky + (1.0 - kk) * (rf_annual / 12.0)
                    met = _performance_metrics(ret, rf_annual=rf_annual)
                    key = f"subset_{row_id}"
                    return_paths[key] = ret
                    rows.append({
                        "row_id": key, "signals": "+".join(subset), "n_signals": size,
                        "combination": mode, "matched_average_exposure": matched,
                        "average_exposure": float(kk.mean()),
                        "exposure_turnover_one_way": float(0.5 * kk.diff().abs().dropna().mean()),
                        **met,
                    })
                    row_id += 1
    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Inference is performed within each research family so a matched-exposure
    # ablation is not mixed into the trial distribution of the unmatched family.
    for (mode, matched), idx in out.groupby(["combination", "matched_average_exposure"]).groups.items():
        ids = list(idx)
        family = out.loc[ids]
        sharpes = pd.to_numeric(family["sharpe"], errors="coerce")
        mu = float(sharpes.mean()) if sharpes.notna().any() else 0.0
        sigma = float(sharpes.std(ddof=1)) if sharpes.notna().sum() > 1 else 0.0
        ret_matrix = pd.concat({out.loc[i, "row_id"]: return_paths[out.loc[i, "row_id"]] for i in ids}, axis=1)
        corr = ret_matrix.corr(min_periods=max(6, len(ret_matrix)//4))
        if len(corr) > 1:
            corr_reg = regularize_trial_correlation(corr.to_numpy(dtype=float), shrinkage=0.10)
            k_eff = effective_number_of_trials(corr_reg)
        else:
            k_eff = 1.0
        k_raw = max(len(ids), 1)
        raw_p: list[float] = []
        for i in ids:
            r = return_paths[out.loc[i, "row_id"]]
            psr = probabilistic_sharpe_ratio(r, benchmark_sharpe=0.0, risk_free_rate=rf_annual, periods_per_year=12)
            dsr_eff = deflated_sharpe_ratio(r, num_trials=k_eff, trial_sharpe_mean=mu, trial_sharpe_std=sigma, risk_free_rate=rf_annual, periods_per_year=12)
            dsr_raw = deflated_sharpe_ratio(r, num_trials=k_raw, trial_sharpe_mean=mu, trial_sharpe_std=sigma, risk_free_rate=rf_annual, periods_per_year=12)
            out.loc[i, "psr"] = float(psr)
            out.loc[i, "dsr_effective_trials"] = float(dsr_eff)
            out.loc[i, "dsr_raw_trials"] = float(dsr_raw)
            out.loc[i, "family_raw_trials"] = float(k_raw)
            out.loc[i, "family_effective_trials"] = float(k_eff)
            out.loc[i, "family_trial_sharpe_mean"] = mu
            out.loc[i, "family_trial_sharpe_std"] = sigma
            raw_p.append(float(np.clip(1.0 - psr, 0.0, 1.0)))
        out.loc[ids, "psr_one_sided_pvalue"] = raw_p
        out.loc[ids, "psr_p_fdr_bh"] = adjusted_pvalues(raw_p, "benjamini_hochberg")
        out.loc[ids, "psr_p_fdr_by"] = adjusted_pvalues(raw_p, "benjamini_yekutieli")

    out["research_rank"] = out.groupby(["combination", "matched_average_exposure"])["dsr_effective_trials"].rank(method="min", ascending=False)
    return out.sort_values(["matched_average_exposure", "dsr_effective_trials", "sharpe"], ascending=[False, False, False]).reset_index(drop=True)


def timer_incremental_comparisons(subsets: pd.DataFrame, core: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Compare one-at-a-time additions and leave-one-out variants against the two-signal core."""
    if subsets.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    if core is None:
        # Infer the two-signal core from the subset ledger itself.
        names = set()
        for value in subsets.get("signals", pd.Series(dtype=str)).astype(str):
            names.update(value.split("+"))
        forward = next((c for c in ("forward_risk_timer", "forward_cdar_timer", "forward_cvar_timer") if c in names), None)
        core = tuple(c for c in ("backward_vol_timer", forward) if c)
    core_set = set(core)
    for (mode, matched), fam in subsets.groupby(["combination", "matched_average_exposure"]):
        parsed = fam.assign(_set=fam["signals"].astype(str).map(lambda s: set(s.split("+"))))
        core_rows = parsed[parsed["_set"].map(lambda z: z == core_set)]
        if core_rows.empty:
            continue
        base = core_rows.sort_values("dsr_effective_trials", ascending=False).iloc[0]
        full_set = set().union(*parsed["_set"].tolist()) if len(parsed) else set()
        for _, row in parsed.iterrows():
            ss = row["_set"]
            relation = None
            changed = None
            if core_set.issubset(ss) and len(ss) == len(core_set) + 1:
                relation = "one_at_a_time_addition"
                changed = next(iter(ss - core_set))
            else:
                missing = full_set - ss
                if len(missing) == 1 and ss == (full_set - missing):
                    relation = "leave_one_out"
                    changed = next(iter(missing))
            if relation is None:
                continue
            rows.append({
                "combination": mode, "matched_average_exposure": bool(matched),
                "relation": relation, "changed_signal": changed, "signals": row["signals"],
                "delta_sharpe_vs_core": float(row["sharpe"] - base["sharpe"]),
                "delta_annualized_return_vs_core": float(row["annualized_return"] - base["annualized_return"]),
                "delta_max_drawdown_vs_core": float(row["max_drawdown"] - base["max_drawdown"]),
                "delta_cvar95_vs_core": float(row["cvar95"] - base["cvar95"]),
                "delta_turnover_vs_core": float(row["exposure_turnover_one_way"] - base["exposure_turnover_one_way"]),
                "dsr_effective_trials": float(row.get("dsr_effective_trials", np.nan)),
                "psr_p_fdr_by": float(row.get("psr_p_fdr_by", np.nan)),
            })
    return pd.DataFrame(rows)


def conditional_signal_surface(
    panel: pd.DataFrame,
    signal_a: str,
    signal_b: str,
    *,
    target: str = "next_abs_return",
    bins: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Quantile-bin two signals and estimate the conditional next-period target."""
    d = panel[[signal_a, signal_b, target]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(d) < max(20, bins * 3):
        return pd.DataFrame(), pd.DataFrame()
    try:
        qa = pd.qcut(d[signal_a], q=bins, labels=False, duplicates="drop")
        qb = pd.qcut(d[signal_b], q=bins, labels=False, duplicates="drop")
    except Exception:
        return pd.DataFrame(), pd.DataFrame()
    work = d.assign(bin_a=qa, bin_b=qb).dropna(subset=["bin_a", "bin_b"])
    mean = work.pivot_table(index="bin_b", columns="bin_a", values=target, aggfunc="mean")
    count = work.pivot_table(index="bin_b", columns="bin_a", values=target, aggfunc="count")
    mean.index.name = f"{signal_b}_quantile"; mean.columns.name = f"{signal_a}_quantile"
    count.index.name = mean.index.name; count.columns.name = mean.columns.name
    return mean, count


def _plot_matrix(df: pd.DataFrame, title: str, path: Path, *, vmin: float | None = None, vmax: float | None = None) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(df.columns)), max(6, 0.40 * len(df.index))))
    im = ax.imshow(df.to_numpy(dtype=float), aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(df.columns))); ax.set_xticklabels(df.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(df.index))); ax.set_yticklabels(df.index, fontsize=7)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def _plot_distribution_best(panel: pd.DataFrame, fits: pd.DataFrame, outdir: Path, max_plots: int = 12) -> None:
    if fits.empty:
        return
    best = fits[fits["best_by_aic"]].head(max_plots)
    for _, row in best.iterrows():
        c = str(row["signal"]); name = str(row["distribution"])
        x = pd.to_numeric(panel[c], errors="coerce").dropna().to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(8, 5)); ax.hist(x, bins="auto", density=True, alpha=0.45)
        grid = np.linspace(float(np.min(x)), float(np.max(x)), 300)
        try:
            params = json.loads(row["params"])
            dist = {"normal": stats.norm, "student_t": stats.t, "laplace": stats.laplace, "logistic": stats.logistic, "skew_normal": stats.skewnorm, "gamma": stats.gamma, "lognormal": stats.lognorm, "beta_[0,1]": stats.beta}[name]
            ax.plot(grid, dist.pdf(grid, *params), label=f"best AIC: {name}")
        except Exception:
            pass
        ax.set_title(f"Signal distribution: {c}"); ax.legend(); ax.grid(alpha=0.2)
        fig.tight_layout(); fig.savefig(outdir / f"distribution_{c[:80].replace('/', '_')}.png", dpi=160, bbox_inches="tight"); plt.close(fig)


def _plot_pair_response_surfaces(panel: pd.DataFrame, timer_cols: list[str], outdir: Path, floor: float = 0.30) -> None:
    for a, b in itertools.combinations(timer_cols[:5], 2):
        grid = np.linspace(floor, 1.0, 80)
        aa, bb = np.meshgrid(grid, grid)
        geo = np.sqrt(aa * bb)
        prod = np.clip(aa * bb, floor, 1.0)
        for name, z in (("geometric", geo), ("product", prod)):
            fig, ax = plt.subplots(figsize=(7, 6))
            im = ax.pcolormesh(grid, grid, z, shading="auto")
            ax.set_xlabel(a); ax.set_ylabel(b); ax.set_title(f"Combined exposure response: {name}")
            fig.colorbar(im, ax=ax, label="combined risky exposure")
            fig.tight_layout(); fig.savefig(outdir / f"response_{name}_{a}_vs_{b}.png", dpi=170, bbox_inches="tight"); plt.close(fig)


def _safe_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(value))[:120]


def run_signal_research(
    results_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    rolling_window: int = 24,
    max_lag: int = 6,
    defensive_quantile: float = 0.20,
    min_train: int = 24,
    distribution_bootstrap_samples: int = 99,
    conditional_bins: int = 5,
    max_surface_signals: int = 6,
) -> dict[str, Any]:
    """Run the publication-oriented multi-signal research suite.

    The routine deliberately separates descriptive dependence evidence from causal
    OOS selection evidence and from multiple-testing-adjusted strategy evidence.
    Cointegration is conditional on I(1) screening; bounded exposure dials are not
    forced into unit-root/cointegration tests.
    """
    results_dir = Path(results_dir)
    out = Path(output_dir) if output_dir is not None else results_dir / "signal_research"
    out.mkdir(parents=True, exist_ok=True)
    panel = load_signal_panel(results_dir)
    panel.to_csv(out / "signal_panel.csv")
    signals = _numeric_signal_columns(panel)

    deps = pairwise_dependence(panel, signals)
    for name, frame in deps.items():
        frame.to_csv(out / f"signal_{name}.csv")
    rolling = rolling_correlations(panel, signals, window=rolling_window)
    rolling.to_csv(out / "rolling_pairwise_correlations.csv", index=False)
    leadlag, bestlag = lead_lag_analysis(panel, signals, max_lag=max_lag)
    leadlag.to_csv(out / "lead_lag_correlations.csv", index=False)
    bestlag.to_csv(out / "best_lead_lag_matrix.csv")
    overlap = event_overlap(panel, signals, quantile=defensive_quantile)
    overlap.to_csv(out / "defensive_event_jaccard.csv")

    station = stationarity_tests(panel, signals)
    station.to_csv(out / "stationarity_tests.csv", index=False)
    coint_pairs, coint_obj = cointegration_tests(panel, station, signals)
    coint_pairs.to_csv(out / "engle_granger_cointegration.csv", index=False)
    coint_obj["matrix"].to_csv(out / "cointegration_pvalue_matrix.csv")
    (out / "johansen_cointegration.json").write_text(
        json.dumps(coint_obj["summary"], indent=2), encoding="utf-8"
    )
    group_coint = group_cointegration_relationships(panel, station, signals)
    group_coint.to_csv(out / "signal_group_cointegration.csv", index=False)

    fits = distribution_fits(
        panel, signals,
        bootstrap_samples=max(int(distribution_bootstrap_samples), 19),
    )
    fits.to_csv(out / "distribution_fits.csv", index=False)
    groups = signal_group_relationships(panel, signals)
    groups.to_csv(out / "signal_group_relationships.csv", index=False)

    # Keep predictive search scientifically narrower than the descriptive census.
    incremental_candidates = [
        c for c in signals
        if (
            c in TIMER_COLUMNS
            or c in {
                "portfolio_kelly", "portfolio_risk_reward", "portfolio_garch_vol",
                "portfolio_realized_vol", "portfolio_vol_regime", "forward_model_cvar",
                "forward_model_cvar_raw", "forward_model_cdar", "forward_model_cdar_raw",
                "portfolio_scenario_mean", "historical_cvar_target", "historical_cdar_target",
                "composite_quality", "cvar_model", "cvar_model_centered", "cvar_model_raw",
                "cdar_model", "cdar_model_centered", "cdar_model_raw", "vol_model",
            }
            or c.startswith("asset_mean_")
        )
    ]
    incremental_candidates = list(dict.fromkeys(incremental_candidates))[:28]
    incremental_risk = incremental_signal_research(
        panel, incremental_candidates, target="next_abs_return", min_train=min_train,
    )
    incremental_risk.to_csv(out / "incremental_signal_order_future_risk.csv", index=False)
    incremental_return = incremental_signal_research(
        panel, incremental_candidates, target="next_realized_return", min_train=min_train,
    )
    incremental_return.to_csv(out / "incremental_signal_order_future_return.csv", index=False)

    partial_risk = partial_information_tests(
        panel, incremental_candidates, target="next_abs_return", max_lag=min(max_lag, 6),
    )
    partial_risk.to_csv(out / "hac_incremental_information_future_risk.csv", index=False)
    partial_return = partial_information_tests(
        panel, incremental_candidates, target="next_realized_return", max_lag=min(max_lag, 6),
    )
    partial_return.to_csv(out / "hac_incremental_information_future_return.csv", index=False)

    timers = _preferred_timer_columns(panel)
    subset = timer_subset_research(panel, timers)
    subset.to_csv(out / "timer_subset_strategy_metrics.csv", index=False)
    timer_increments = timer_incremental_comparisons(subset)
    timer_increments.to_csv(out / "timer_incremental_additions_leaveoneout.csv", index=False)

    # Full matrices are always saved. Raster heatmaps use a publication-sized subset.
    plot_signals = [c for c in incremental_candidates if c in signals][:28]
    if not plot_signals:
        plot_signals = signals[:28]
    _plot_matrix(
        deps["pearson"].loc[plot_signals, plot_signals], "Signal Pearson correlation",
        out / "pearson_correlation_heatmap.png", vmin=-1, vmax=1,
    )
    _plot_matrix(
        deps["spearman"].loc[plot_signals, plot_signals], "Signal Spearman correlation",
        out / "spearman_correlation_heatmap.png", vmin=-1, vmax=1,
    )
    _plot_matrix(
        overlap.loc[plot_signals, plot_signals],
        f"Defensive-event Jaccard overlap (q={defensive_quantile:.2f})",
        out / "defensive_event_overlap_heatmap.png", vmin=0, vmax=1,
    )
    _plot_matrix(
        coint_obj["matrix"].loc[plot_signals, plot_signals],
        "Engle-Granger p-values (only eligible I(1) precursors)",
        out / "cointegration_pvalue_heatmap.png", vmin=0, vmax=1,
    )
    _plot_matrix(
        bestlag.loc[plot_signals, plot_signals], "Lag of strongest pairwise cross-correlation",
        out / "best_lead_lag_heatmap.png", vmin=-max_lag, vmax=max_lag,
    )
    _plot_distribution_best(panel, fits, out)
    _plot_pair_response_surfaces(panel, timers, out)

    # Empirical signal-value heatmaps: unlike response surfaces, these use the
    # observed next-period outcomes and therefore answer whether two signal states
    # jointly map to unusually high/low future risk or return.
    surface_priority = timers + [
        c for c in ("portfolio_kelly", "portfolio_risk_reward", "portfolio_vol_regime",
                    "portfolio_realized_vol", "forward_model_cvar") if c in panel.columns
    ]
    surface_signals = list(dict.fromkeys([c for c in surface_priority if c in signals]))[:max_surface_signals]
    conditional_surfaces: list[str] = []
    for a, b in itertools.combinations(surface_signals, 2):
        for target, label in (("next_abs_return", "future_risk"), ("next_realized_return", "future_return")):
            mean_surface, count_surface = conditional_signal_surface(
                panel, a, b, target=target, bins=max(int(conditional_bins), 2),
            )
            if mean_surface.empty:
                continue
            stem = f"conditional_{label}_{_safe_slug(a)}_vs_{_safe_slug(b)}"
            mean_surface.to_csv(out / f"{stem}_mean.csv")
            count_surface.to_csv(out / f"{stem}_count.csv")
            _plot_matrix(
                mean_surface,
                f"Conditional {label.replace('_', ' ')}: {a} vs {b}",
                out / f"{stem}.png",
            )
            conditional_surfaces.append(stem)

    summary: dict[str, Any] = {
        "results_dir": str(results_dir),
        "observations": int(len(panel)),
        "signal_count": int(len(signals)),
        "signals": signals,
        "timer_signals": timers,
        "cointegration_eligible_i1_signals": coint_obj["summary"].get("eligible_i1_signals", []),
        "cointegration_note": (
            "Bounded timers are not forced into cointegration tests; Engle-Granger/Johansen "
            "are conditional on I(1) screening. Group-vs-group Johansen diagnostics are also conditional on I(1)."
        ),
        "distribution_gof_note": (
            "AIC/BIC select candidate families; the best-AIC family receives a parametric-bootstrap/Monte-Carlo "
            "KS p-value that refits unknown parameters. Naive fitted-KS p-values are intentionally not reported as valid inference."
        ),
        "distribution_bootstrap_samples": int(max(distribution_bootstrap_samples, 19)),
        "best_distributions_by_aic": (
            fits[fits.get("best_by_aic", False)].set_index("signal")["distribution"].to_dict()
            if not fits.empty else {}
        ),
        "conditional_surface_artifacts": conditional_surfaces,
        "best_matched_exposure_timer_subset_by_sharpe": None,
        "best_matched_exposure_timer_subset_by_dsr": None,
        "research_ledger_note": (
            "Any subset or signal selected after inspecting these outputs must be counted as a research trial "
            "for DSR/FDR before confirmatory testing. Subset PSR/DSR/FDR columns are exploratory because the same sample defines the family."
        ),
    }
    if not subset.empty:
        matched = subset[subset["matched_average_exposure"] == True]  # noqa: E712
        if not matched.empty:
            row_sr = matched.sort_values("sharpe", ascending=False).iloc[0]
            row_dsr = matched.sort_values("dsr_effective_trials", ascending=False).iloc[0]
            summary["best_matched_exposure_timer_subset_by_sharpe"] = row_sr.to_dict()
            summary["best_matched_exposure_timer_subset_by_dsr"] = row_dsr.to_dict()
    (out / "signal_research_summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2, default=str, allow_nan=False), encoding="utf-8"
    )
    return summary

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HDRC multi-signal dependence, cointegration, distribution and subset research")
    p.add_argument("--results-dir", required=True)
    p.add_argument("--output-dir")
    p.add_argument("--rolling-window", type=int, default=24)
    p.add_argument("--max-lag", type=int, default=6)
    p.add_argument("--defensive-quantile", type=float, default=0.20)
    p.add_argument("--min-train", type=int, default=24)
    p.add_argument("--distribution-bootstrap-samples", type=int, default=99)
    p.add_argument("--conditional-bins", type=int, default=5)
    p.add_argument("--max-surface-signals", type=int, default=6)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run_signal_research(
        args.results_dir, output_dir=args.output_dir, rolling_window=args.rolling_window,
        max_lag=args.max_lag, defensive_quantile=args.defensive_quantile, min_train=args.min_train,
        distribution_bootstrap_samples=args.distribution_bootstrap_samples,
        conditional_bins=args.conditional_bins, max_surface_signals=args.max_surface_signals,
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
