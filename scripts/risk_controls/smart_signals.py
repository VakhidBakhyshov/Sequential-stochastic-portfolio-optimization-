"""Causal smart signals for monthly ETF allocation.

The module deliberately separates three roles:
1. asset ranking/tilting (efficiency, liquidity, breakout, pullback, spike);
2. forward conviction (posterior-scenario Kelly and risk/reward);
3. portfolio exposure control (backward realised volatility and forward model-CVaR).

The last layer implements the paper's risk-signal-diversification mechanism.  All
inputs must end at the current rebalance date; no future realised return is used.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from scripts.calculations.drawdown import empirical_cdar, portfolio_path_returns

EPS = 1e-12


def _as_returns_frame(returns: pd.DataFrame | np.ndarray, columns: Iterable[str] | None = None) -> pd.DataFrame:
    if isinstance(returns, pd.DataFrame):
        df = returns.copy()
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.set_index("Date")
        df = df.apply(pd.to_numeric, errors="coerce")
    else:
        arr = np.asarray(returns, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        df = pd.DataFrame(arr, columns=list(columns) if columns is not None else None)
    return df.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _scenario_frame(scenarios: np.ndarray | pd.DataFrame | None, columns: list[str]) -> pd.DataFrame:
    if scenarios is None:
        return pd.DataFrame(columns=columns)
    if isinstance(scenarios, pd.DataFrame):
        df = scenarios.copy().drop(columns=["Date"], errors="ignore")
        common = [c for c in columns if c in df.columns]
        if common:
            return df.reindex(columns=columns).apply(pd.to_numeric, errors="coerce").fillna(0.0)
        arr = df.to_numpy(dtype=float)
    else:
        arr = np.asarray(scenarios, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    n = min(arr.shape[1], len(columns))
    out = pd.DataFrame(arr[:, :n], columns=columns[:n])
    return out.reindex(columns=columns, fill_value=0.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def price_proxy_from_returns(returns: pd.DataFrame, start_price: float = 100.0, return_type: str = "log-returns") -> pd.DataFrame:
    r = returns.astype(float)
    px = np.exp(r.cumsum()) * start_price if return_type == "log-returns" else (1.0 + r).cumprod() * start_price
    return px.replace([np.inf, -np.inf], np.nan).ffill().fillna(start_price)


def rolling_efficiency_ratio(price: pd.DataFrame, window: int = 21) -> pd.Series:
    if price.empty:
        return pd.Series(dtype=float)
    p = price.tail(max(2, window + 1))
    net = (p.iloc[-1] - p.iloc[0]).abs()
    path = p.diff().abs().sum(axis=0)
    return (net / path.replace(0.0, np.nan)).clip(0.0, 1.0).fillna(0.0)


def rolling_atr_regime(price: pd.DataFrame, short_window: int = 21, long_window: int = 126) -> pd.Series:
    tr = price.pct_change().abs()
    short = tr.tail(short_window).mean(axis=0)
    long = tr.tail(long_window).mean(axis=0)
    return (short / long.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(1.0)


def rolling_breakout_strength(price: pd.DataFrame, window: int = 63) -> pd.Series:
    p = price.tail(window)
    hi, lo = p.max(axis=0), p.min(axis=0)
    return ((price.iloc[-1] - lo) / (hi - lo).replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.5)


def normalized_return_spike(returns: pd.DataFrame, window: int = 21) -> pd.Series:
    vol = returns.tail(window).std(axis=0, ddof=1)
    return (returns.iloc[-1] / vol.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def pullback_ratio(price: pd.DataFrame, window: int = 63) -> pd.Series:
    p = price.tail(window)
    hi, lo = p.max(axis=0), p.min(axis=0)
    return ((hi - price.iloc[-1]) / (hi - lo).replace(0.0, np.nan)).clip(0.0, 1.0).fillna(0.0)


def net_move(returns: pd.DataFrame, window: int = 21, return_type: str = "log-returns") -> pd.Series:
    r = returns.tail(window)
    return r.sum(axis=0) if return_type == "log-returns" else (1.0 + r).prod(axis=0) - 1.0


def garch11_vol_forecast(returns: pd.DataFrame, omega: float = 1e-6, alpha: float = 0.08, beta: float = 0.90) -> pd.Series:
    """Fast fixed-parameter GARCH(1,1)-style one-day volatility forecast."""
    if alpha < 0 or beta < 0 or alpha + beta >= 1.0:
        raise ValueError("GARCH parameters require alpha>=0, beta>=0 and alpha+beta<1.")
    out: dict[str, float] = {}
    for col in returns.columns:
        x = returns[col].dropna().to_numpy(dtype=float)
        if x.size == 0:
            out[col] = 0.0
            continue
        var = max(float(np.var(x, ddof=1)) if x.size > 1 else float(x[-1] ** 2), EPS)
        for val in x[-504:]:
            var = omega + alpha * float(val) ** 2 + beta * var
        out[col] = float(np.sqrt(max(var, EPS)))
    return pd.Series(out).fillna(0.0)


def kelly_fraction_from_scenarios(scenarios: pd.DataFrame, cap: float = 0.25, floor: float = 0.0) -> pd.Series:
    if scenarios.empty:
        return pd.Series(dtype=float)
    mu = scenarios.mean(axis=0)
    var = scenarios.var(axis=0, ddof=1).replace(0.0, np.nan)
    return (mu / var).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=floor, upper=cap)


def risk_reward_ratio_from_scenarios(scenarios: pd.DataFrame, eps: float = EPS) -> pd.Series:
    if scenarios.empty:
        return pd.Series(dtype=float)
    upside = scenarios.clip(lower=0.0).mean(axis=0)
    downside = (-scenarios.clip(upper=0.0)).mean(axis=0)
    return (upside / (downside + eps)).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def liquidity_score(market_cap_or_volume: pd.DataFrame | np.ndarray | None, columns: list[str], window: int = 21) -> pd.Series:
    if market_cap_or_volume is None:
        return pd.Series(1.0, index=columns)
    df = _as_returns_frame(market_cap_or_volume, columns=columns).reindex(columns=columns).fillna(0.0)
    liq = df.tail(window).median(axis=0)
    positive = liq[liq > 0]
    med = float(positive.median()) if not positive.empty else 0.0
    return (liq / med).clip(0.0, 10.0).fillna(1.0) if med > EPS else pd.Series(1.0, index=columns)


def _zscore(s: pd.Series, clip: float = 3.0) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    sd = float(s.std(ddof=1)) if len(s) > 1 else 0.0
    return ((s - float(s.mean())) / sd).clip(-clip, clip).fillna(0.0) if sd > EPS else pd.Series(0.0, index=s.index)


def _sigmoid(x: pd.Series | float) -> pd.Series | float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _empirical_cvar_loss(x: np.ndarray | pd.Series, alpha: float = 0.95) -> float:
    values = np.asarray(x, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    losses = -values
    var = float(np.quantile(losses, alpha))
    tail = losses[losses >= var]
    return float(tail.mean()) if tail.size else var


def _rolling_path_matrix(x: pd.Series, horizon: int) -> np.ndarray:
    """Causal overlapping historical paths of fixed length for CDaR targeting."""
    v = pd.to_numeric(x, errors="coerce").dropna().to_numpy(dtype=float)
    h = max(1, int(horizon))
    if v.size < h:
        return v.reshape(1, -1) if v.size else np.empty((0, h))
    return np.lib.stride_tricks.sliding_window_view(v, h).copy()


def _predictive_cdar_pair(
    scenario_paths: np.ndarray | None, risky_w: np.ndarray, *, alpha: float, center: bool
) -> tuple[float, float]:
    """Return raw and centered predictive CDaR for an (S,H,N) path cube."""
    if scenario_paths is None:
        return float("nan"), float("nan")
    paths = np.asarray(scenario_paths, dtype=float)
    if paths.ndim != 3 or paths.shape[2] != risky_w.size:
        return float("nan"), float("nan")
    port_raw = portfolio_path_returns(paths, risky_w)
    raw = float(empirical_cdar(port_raw, alpha=alpha))
    if not center:
        return raw, raw
    centered = paths - np.mean(paths, axis=0, keepdims=True)
    ctr = float(empirical_cdar(portfolio_path_returns(centered, risky_w), alpha=alpha))
    return raw, ctr


@dataclass
class PortfolioSignalDiagnostics:
    overlay_fraction: float
    backward_vol_timer: float
    forward_cvar_timer: float
    conviction_timer: float
    portfolio_kelly: float
    portfolio_risk_reward: float
    portfolio_garch_vol: float
    portfolio_realized_vol: float
    portfolio_vol_regime: float
    forward_model_cvar: float
    historical_cvar_target: float
    composite_quality: float
    selected_count: int
    overlay_combination: str
    comments: str
    forward_model_cvar_raw: float = 0.0
    portfolio_scenario_mean: float = 0.0
    historical_cvar_target_raw: float = 0.0
    signal_centering_applied: bool = True
    forward_cdar_timer: float = float("nan")
    forward_risk_timer: float = float("nan")
    forward_model_cdar: float = float("nan")
    forward_model_cdar_raw: float = float("nan")
    historical_cdar_target: float = float("nan")
    historical_cdar_target_raw: float = float("nan")
    forward_risk_measure: str = "cvar"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_asset_signal_table(
    *, etfs_list: list[str], historical_returns: pd.DataFrame | np.ndarray,
    pred_returns: np.ndarray | pd.DataFrame | None = None,
    market_cap_history: pd.DataFrame | np.ndarray | None = None,
    return_type: str = "log-returns", config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    cfg = config or {}
    hist = _as_returns_frame(historical_returns, columns=etfs_list).reindex(columns=etfs_list).fillna(0.0)
    price = price_proxy_from_returns(hist, return_type=return_type)
    scen = _scenario_frame(pred_returns, etfs_list) if pred_returns is not None else hist.tail(252)
    if scen.empty:
        scen = hist.tail(252)

    er = rolling_efficiency_ratio(price, int(cfg.get("er_window", 21)))
    atr = rolling_atr_regime(price, int(cfg.get("atr_short_window", 21)), int(cfg.get("atr_long_window", 126)))
    liq = liquidity_score(market_cap_history, etfs_list, int(cfg.get("liquidity_window", 21)))
    breakout = rolling_breakout_strength(price, int(cfg.get("breakout_window", 63)))
    spike = normalized_return_spike(hist, int(cfg.get("spike_window", 21)))
    pullback = pullback_ratio(price, int(cfg.get("pullback_window", 63)))
    move = net_move(hist, int(cfg.get("net_move_window", 21)), return_type=return_type)
    gvol = garch11_vol_forecast(hist, float(cfg.get("garch_omega", 1e-6)), float(cfg.get("garch_alpha", 0.08)), float(cfg.get("garch_beta", 0.90)))
    rr = risk_reward_ratio_from_scenarios(scen)
    kelly = kelly_fraction_from_scenarios(scen, float(cfg.get("kelly_cap", 0.25)), float(cfg.get("kelly_floor", 0.0)))

    preferred = float(cfg.get("preferred_pullback", 0.25))
    pullback_good = 1.0 - (pullback - preferred).abs().clip(0.0, 1.0)
    raw = (
        float(cfg.get("w_er", 0.22)) * _zscore(er)
        + float(cfg.get("w_breakout", 0.14)) * _zscore(breakout)
        + float(cfg.get("w_liquidity", 0.14)) * _zscore(np.log1p(liq))
        + float(cfg.get("w_risk_reward", 0.18)) * _zscore(np.log1p(rr))
        + float(cfg.get("w_net_move", 0.12)) * _zscore(move)
        + float(cfg.get("w_pullback", 0.08)) * _zscore(pullback_good)
        - float(cfg.get("w_vol_regime", 0.16)) * _zscore(atr)
        - float(cfg.get("w_spike", 0.10)) * _zscore(spike.abs())
    )
    quality = pd.Series(_sigmoid(raw), index=etfs_list).clip(0.05, 0.95)
    return pd.DataFrame({
        "efficiency_ratio": er, "atr_vol_regime": atr, "liquidity_score": liq,
        "breakout_strength": breakout, "normalized_spike": spike,
        "pullback_ratio": pullback, "net_move": move, "garch_vol_forecast": gvol,
        "risk_reward_ratio": rr, "kelly_fraction": kelly, "signal_quality": quality,
    }).reindex(etfs_list).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def apply_smart_position_sizing(*, asset_weights: pd.Series, signal_table: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.Series:
    """Apply bounded tilts while preserving the incoming risky-sleeve exposure."""
    cfg = config or {}
    w = asset_weights.reindex(signal_table.index).fillna(0.0).astype(float).clip(lower=0.0)
    exposure = float(w.sum())
    if exposure <= EPS:
        return w
    base = w / exposure
    q = signal_table["signal_quality"].clip(0.05, 0.95)
    kcap = max(float(cfg.get("kelly_cap", 0.25)), EPS)
    k_mult = float(cfg.get("kelly_multiplier_floor", 0.60)) + (1.0 - float(cfg.get("kelly_multiplier_floor", 0.60))) * signal_table["kelly_fraction"].clip(0.0, kcap) / kcap
    rrcap = max(float(cfg.get("risk_reward_cap", 5.0)), EPS)
    rr_mult = (0.60 + 0.40 * np.log1p(signal_table["risk_reward_ratio"].clip(0.0, rrcap)) / np.log1p(rrcap)).clip(0.50, 1.10)
    quality_strength = float(np.clip(cfg.get("quality_tilt_strength", 0.50), 0.0, 1.0))
    q_mult = (1.0 - quality_strength) + quality_strength * (q / max(float(q.mean()), EPS))
    tilted = base * q_mult * k_mult * rr_mult
    tilted[tilted < float(cfg.get("min_active_weight", 0.0))] = 0.0
    if tilted.sum() <= EPS:
        tilted = base
    else:
        tilted /= tilted.sum()

    cap = cfg.get("max_active_weight")
    if cap is not None:
        cap = max(float(cap), 1.0 / max(int((tilted > 0).sum()), 1))
        for _ in range(len(tilted) + 2):
            over = tilted > cap
            if not over.any():
                break
            excess = float((tilted[over] - cap).sum())
            tilted[over] = cap
            under = (~over) & (tilted > 0)
            if not under.any():
                break
            tilted[under] += excess * tilted[under] / max(float(tilted[under].sum()), EPS)
        tilted /= max(float(tilted.sum()), EPS)
    return (tilted * exposure).fillna(0.0)


def portfolio_signal_diagnostics(
    *, asset_weights: pd.Series, signal_table: pd.DataFrame,
    historical_returns: pd.DataFrame | np.ndarray,
    pred_returns: np.ndarray | pd.DataFrame | None = None,
    scenario_paths: np.ndarray | None = None,
    config: dict[str, Any] | None = None,
) -> PortfolioSignalDiagnostics:
    cfg = config or {}
    cols = list(asset_weights.index)
    hist = _as_returns_frame(historical_returns, columns=cols).reindex(columns=cols).fillna(0.0)
    scen = _scenario_frame(pred_returns, cols) if pred_returns is not None else pd.DataFrame(columns=cols)
    w = asset_weights.reindex(cols).fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
    exposure = float(w.sum())
    if not cols or exposure <= EPS:
        return PortfolioSignalDiagnostics(
            overlay_fraction=1.0, backward_vol_timer=1.0, forward_cvar_timer=1.0,
            conviction_timer=1.0, portfolio_kelly=0.0, portfolio_risk_reward=0.0,
            portfolio_garch_vol=0.0, portfolio_realized_vol=0.0, portfolio_vol_regime=1.0,
            forward_model_cvar=0.0, historical_cvar_target=0.0, composite_quality=0.5,
            selected_count=0, overlay_combination="none", comments="empty book",
        )
    risky_w = w / exposure
    horizon = max(1, int(cfg.get("horizon", 21)))
    alpha = float(cfg.get("cvar_alpha", 0.95))

    port_daily = pd.Series(hist.to_numpy() @ risky_w, index=hist.index)
    port_scen = pd.Series(scen.to_numpy() @ risky_w) if not scen.empty else pd.Series(dtype=float)

    # Backward timer: causal realised/GARCH volatility, converted to the holding-period scale.
    daily_realized = float(port_daily.tail(int(cfg.get("vol_short_window", 21))).std(ddof=1))
    daily_garch = float(garch11_vol_forecast(pd.DataFrame({"portfolio": port_daily}), float(cfg.get("garch_omega", 1e-6)), float(cfg.get("garch_alpha", 0.08)), float(cfg.get("garch_beta", 0.90))).iloc[0])
    monthly_realized = max(daily_realized, 0.0) * np.sqrt(horizon)
    monthly_garch = max(daily_garch, 0.0) * np.sqrt(horizon)
    forecast_vol = max(monthly_realized, monthly_garch, EPS)
    target_vol = float(cfg.get("target_monthly_vol", 0.035))
    min_overlay = float(np.clip(cfg.get("min_overlay", 0.30), 0.0, 1.0))
    k_vol = float(np.clip(target_vol / forecast_vol, min_overlay, 1.0))

    # Forward timer: posterior-predictive CVaR relative to a causal historical target.
    #
    # Construction Rule 3 in the manuscript requires the internal risk signal to
    # be read from de-meaned scenarios. Otherwise, at a multi-day horizon the
    # optimized mean grows as O(H) while volatility grows as O(sqrt(H)), so the
    # nominal CVaR timer becomes a selected return forecast in disguise. Keep the
    # raw quantity for audit, but use the centered quantity for exposure control.
    center_forward_cvar = bool(cfg.get("center_forward_cvar", True))
    scenario_mean = float(port_scen.mean()) if not port_scen.empty else 0.0
    forward_cvar_raw = _empirical_cvar_loss(port_scen, alpha) if not port_scen.empty else 0.0
    centered_scen = port_scen - scenario_mean if (center_forward_cvar and not port_scen.empty) else port_scen
    forward_cvar = _empirical_cvar_loss(centered_scen, alpha) if not centered_scen.empty else 0.0

    historical_horizon = port_daily.rolling(horizon).sum().dropna()
    hist_window = historical_horizon.tail(int(cfg.get("cvar_target_window", 504)))
    hist_cvar_raw = _empirical_cvar_loss(hist_window, alpha) if not hist_window.empty else max(forward_cvar_raw, EPS)
    if center_forward_cvar and not hist_window.empty:
        hist_for_target = hist_window - float(hist_window.mean())
    else:
        hist_for_target = hist_window
    hist_cvar = _empirical_cvar_loss(hist_for_target, alpha) if not hist_for_target.empty else max(forward_cvar, EPS)
    target_cvar = float(cfg.get("target_cvar", hist_cvar * float(cfg.get("cvar_target_multiplier", 1.0))))
    if target_cvar <= EPS:
        target_cvar = max(hist_cvar, EPS)
    k_cvar = float(np.clip(target_cvar / max(forward_cvar, EPS), min_overlay, 1.0)) if forward_cvar > EPS else 1.0

    # Parallel path-dependent forward timer.  It uses the same selected book and
    # the same causal scenario generator, but reads underwater path risk rather
    # than terminal loss risk.  The historical target is constructed from
    # overlapping trailing H-day paths and therefore never uses future returns.
    cdar_alpha = float(cfg.get("cdar_alpha", alpha))
    center_forward_cdar = bool(cfg.get("center_forward_cdar", True))
    forward_cdar_raw, forward_cdar = _predictive_cdar_pair(
        scenario_paths, risky_w, alpha=cdar_alpha, center=center_forward_cdar
    )
    hist_cdar_window = port_daily.tail(int(cfg.get("cdar_target_window", cfg.get("cvar_target_window", 504))))
    hist_paths_raw = _rolling_path_matrix(hist_cdar_window, horizon)
    hist_cdar_raw = float(empirical_cdar(hist_paths_raw, alpha=cdar_alpha)) if hist_paths_raw.size else float("nan")
    hist_centered_series = hist_cdar_window - float(hist_cdar_window.mean()) if (center_forward_cdar and len(hist_cdar_window)) else hist_cdar_window
    hist_paths_ctr = _rolling_path_matrix(hist_centered_series, horizon)
    hist_cdar = float(empirical_cdar(hist_paths_ctr, alpha=cdar_alpha)) if hist_paths_ctr.size else float("nan")
    target_cdar = float(cfg.get("target_cdar", hist_cdar * float(cfg.get("cdar_target_multiplier", 1.0)))) if np.isfinite(hist_cdar) else float("nan")
    k_cdar = (
        float(np.clip(target_cdar / max(forward_cdar, EPS), min_overlay, 1.0))
        if np.isfinite(target_cdar) and np.isfinite(forward_cdar) and forward_cdar > EPS else float("nan")
    )

    forward_measure = str(cfg.get("forward_risk_measure", "cvar")).strip().lower()
    if forward_measure == "cdar" and np.isfinite(k_cdar):
        k_forward = k_cdar
    elif forward_measure in {"min", "minimum", "cvar_cdar_min"} and np.isfinite(k_cdar):
        k_forward = min(k_cvar, k_cdar)
    elif forward_measure in {"geometric", "cvar_cdar_geometric"} and np.isfinite(k_cdar):
        k_forward = float(np.sqrt(max(k_cvar, EPS) * max(k_cdar, EPS)))
    else:
        forward_measure = "cvar"
        k_forward = k_cvar

    mu = scenario_mean if not port_scen.empty else float(historical_horizon.tail(252).mean())
    var = float(port_scen.var(ddof=1)) if len(port_scen) > 1 else float(historical_horizon.tail(252).var(ddof=1))
    kelly = float(np.clip(mu / max(var, EPS), 0.0, float(cfg.get("portfolio_kelly_cap", 1.0))))
    upside = float(port_scen.clip(lower=0.0).mean()) if not port_scen.empty else float(historical_horizon.clip(lower=0.0).tail(252).mean())
    downside = float((-port_scen.clip(upper=0.0)).mean()) if not port_scen.empty else float((-historical_horizon.clip(upper=0.0)).tail(252).mean())
    rr = float(upside / (downside + EPS))
    quality = float(np.average(signal_table["signal_quality"].reindex(cols).fillna(0.5), weights=risky_w))

    rr_scale = float(np.clip(np.log1p(rr) / max(np.log1p(float(cfg.get("target_risk_reward", 1.5))), EPS), min_overlay, 1.0))
    kelly_scale = float(np.clip(kelly / max(float(cfg.get("target_portfolio_kelly", 0.50)), EPS), min_overlay, 1.0))
    quality_scale = float(np.clip(quality / max(float(cfg.get("target_signal_quality", 0.55)), EPS), min_overlay, 1.0))
    conviction = float((rr_scale * kelly_scale * quality_scale) ** (1.0 / 3.0))

    combination = str(cfg.get("overlay_combination", "geometric_mean")).lower()
    include_conviction = bool(cfg.get("include_conviction_timer", True))
    timers = np.array([k_vol, k_forward, conviction] if include_conviction else [k_vol, k_forward], dtype=float)
    if combination == "product":
        overlay = float(np.prod(timers))
    elif combination == "minimum":
        overlay = float(np.min(timers))
    elif combination == "convex_blend":
        fw = float(np.clip(cfg.get("forward_weight", 0.50), 0.0, 1.0))
        risk_blend = fw * k_forward + (1.0 - fw) * k_vol
        if include_conviction:
            cw = float(np.clip(cfg.get("conviction_weight", 0.20), 0.0, 1.0))
            overlay = (1.0 - cw) * risk_blend + cw * conviction
        else:
            overlay = risk_blend
    else:  # gentler than a product; avoids the severe exposure collapse seen in the old smart strategy.
        combination = "geometric_mean"
        overlay = float(np.prod(timers) ** (1.0 / len(timers)))
    overlay = float(np.clip(overlay, min_overlay, 1.0))

    long_vol = float(port_daily.tail(int(cfg.get("vol_long_window", 126))).std(ddof=1)) * np.sqrt(horizon)
    regime = monthly_realized / max(long_vol, EPS) if long_vol > EPS else 1.0
    return PortfolioSignalDiagnostics(
        overlay, k_vol, k_cvar, conviction, kelly, rr, monthly_garch, monthly_realized,
        float(regime), forward_cvar, target_cvar, quality,
        int((asset_weights > float(cfg.get("active_weight_threshold", 1e-4))).sum()), combination,
        (f"backward realised-vol timer + centered forward posterior-{forward_measure.upper()} timer"
         + (" + bounded conviction timer" if include_conviction else "")),
        forward_model_cvar_raw=float(forward_cvar_raw),
        portfolio_scenario_mean=float(scenario_mean),
        historical_cvar_target_raw=float(hist_cvar_raw),
        signal_centering_applied=bool(center_forward_cvar if forward_measure == "cvar" else center_forward_cdar),
        forward_cdar_timer=float(k_cdar) if np.isfinite(k_cdar) else float("nan"),
        forward_risk_timer=float(k_forward),
        forward_model_cdar=float(forward_cdar) if np.isfinite(forward_cdar) else float("nan"),
        forward_model_cdar_raw=float(forward_cdar_raw) if np.isfinite(forward_cdar_raw) else float("nan"),
        historical_cdar_target=float(target_cdar) if np.isfinite(target_cdar) else float("nan"),
        historical_cdar_target_raw=float(hist_cdar_raw) if np.isfinite(hist_cdar_raw) else float("nan"),
        forward_risk_measure=forward_measure,
    )


def smart_rebalance_diagnostics(
    *, etfs_list: list[str], historical_returns: pd.DataFrame | np.ndarray,
    pred_returns: np.ndarray | pd.DataFrame | None,
    scenario_paths: np.ndarray | None = None,
    market_cap_history: pd.DataFrame | np.ndarray | None = None,
    selected_asset_weights: pd.Series, return_type: str,
    config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, PortfolioSignalDiagnostics, pd.Series]:
    cfg = config or {}
    table = build_asset_signal_table(
        etfs_list=etfs_list, historical_returns=historical_returns, pred_returns=pred_returns,
        market_cap_history=market_cap_history, return_type=return_type, config=cfg,
    )
    base = selected_asset_weights.reindex(etfs_list).fillna(0.0).astype(float).clip(lower=0.0)
    tilted = apply_smart_position_sizing(asset_weights=base, signal_table=table, config=cfg)
    diag = portfolio_signal_diagnostics(
        asset_weights=tilted, signal_table=table, historical_returns=historical_returns,
        pred_returns=pred_returns, scenario_paths=scenario_paths, config=cfg,
    )
    if bool(cfg.get("allow_cash_overlay", True)):
        tilted = tilted * diag.overlay_fraction
    return table, diag, tilted
