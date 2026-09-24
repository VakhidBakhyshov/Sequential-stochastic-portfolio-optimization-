"""
Smart risk signals and position controls for the monthly ETF framework.

This module is intentionally dependency-light. It implements the additions requested for the
research pipeline:
  * Kelly criterion sizing from posterior/scenario returns;
  * risk-reward ratio from forward scenarios or historical returns;
  * volatility clustering / GARCH(1,1)-style volatility forecast;
  * backward OHLCV-style signals from returns/price proxies: efficiency ratio, ATR regime,
    liquidity score, breakout strength, normalized spike, pullback ratio and net move;
  * a composite signal-quality score that can either scale target exposure or only be audited.

The functions are causal when called with history ending at the rebalance date.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Iterable

import numpy as np
import pandas as pd

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
        df = pd.DataFrame(arr, columns=list(columns) if columns is not None else None)
    return df.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _scenario_frame(scenarios: np.ndarray | pd.DataFrame | None, columns: list[str]) -> pd.DataFrame:
    if scenarios is None:
        return pd.DataFrame(columns=columns)
    if isinstance(scenarios, pd.DataFrame):
        df = scenarios.copy().drop(columns=["Date"], errors="ignore")
    else:
        arr = np.asarray(scenarios, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != len(columns):
            # Last safe resort: trim to common shape.
            n = min(arr.shape[1], len(columns))
            arr = arr[:, :n]
            columns = columns[:n]
        df = pd.DataFrame(arr, columns=columns)
    return df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)


def price_proxy_from_returns(returns: pd.DataFrame, start_price: float = 100.0, return_type: str = "log-returns") -> pd.DataFrame:
    """Build a price proxy when true OHLC is unavailable."""
    r = returns.copy().astype(float)
    if return_type == "log-returns":
        px = np.exp(r.cumsum()) * start_price
    else:
        px = (1.0 + r).cumprod() * start_price
    return px.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(start_price)


def rolling_efficiency_ratio(price: pd.DataFrame, window: int = 21) -> pd.Series:
    if price.empty:
        return pd.Series(dtype=float)
    net = (price.iloc[-1] - price.iloc[-min(window, len(price))]).abs()
    path = price.diff().abs().tail(window).sum(axis=0)
    return (net / path.replace(0.0, np.nan)).clip(0.0, 1.0).fillna(0.0)


def rolling_atr_regime(price: pd.DataFrame, short_window: int = 21, long_window: int = 126) -> pd.Series:
    """ATR current / ATR long-term using a price proxy when high/low are not present."""
    tr = price.diff().abs()
    short = tr.tail(short_window).mean(axis=0)
    long = tr.tail(long_window).mean(axis=0)
    return (short / long.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(1.0)


def rolling_breakout_strength(price: pd.DataFrame, window: int = 63) -> pd.Series:
    p = price.tail(window)
    hi = p.max(axis=0)
    lo = p.min(axis=0)
    width = (hi - lo).replace(0.0, np.nan)
    # 0.5 = middle of channel, >1 = above recent high, <0 = below recent low
    return ((price.iloc[-1] - lo) / width).replace([np.inf, -np.inf], np.nan).fillna(0.5)


def normalized_return_spike(returns: pd.DataFrame, window: int = 21) -> pd.Series:
    vol = returns.tail(window).std(axis=0, ddof=1)
    return (returns.iloc[-1] / vol.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def pullback_ratio(price: pd.DataFrame, window: int = 63) -> pd.Series:
    p = price.tail(window)
    hi = p.max(axis=0)
    lo = p.min(axis=0)
    denom = (hi - lo).replace(0.0, np.nan)
    # 0 means at high; 1 means at low. We later reward moderate pullbacks, not crashes.
    return ((hi - price.iloc[-1]) / denom).replace([np.inf, -np.inf], np.nan).clip(0.0, 1.0).fillna(0.0)


def net_move(returns: pd.DataFrame, window: int = 21) -> pd.Series:
    return returns.tail(window).sum(axis=0).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def garch11_vol_forecast(returns: pd.DataFrame, omega: float = 1e-6, alpha: float = 0.08, beta: float = 0.90) -> pd.Series:
    """
    Fast GARCH(1,1)-style volatility clustering forecast.

    It avoids the optional `arch` dependency by using fixed but configurable parameters.
    For monthly ETF rebalancing this is usually enough as a risk-regime input.
    """
    r = returns.astype(float)
    out: dict[str, float] = {}
    for col in r.columns:
        x = r[col].dropna().values
        if x.size == 0:
            out[col] = np.nan
            continue
        var = float(np.var(x, ddof=1)) if x.size > 1 else float(x[-1] ** 2)
        var = max(var, EPS)
        for val in x[-252:]:
            var = omega + alpha * float(val) ** 2 + beta * var
        out[col] = float(np.sqrt(max(var, EPS)))
    return pd.Series(out).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def kelly_fraction_from_scenarios(scenarios: pd.DataFrame, cap: float = 0.25, floor: float = 0.0) -> pd.Series:
    """Long-only fractional Kelly: f = mean / variance, clipped to [floor, cap]."""
    if scenarios.empty:
        return pd.Series(dtype=float)
    mu = scenarios.mean(axis=0)
    var = scenarios.var(axis=0, ddof=1).replace(0.0, np.nan)
    k = (mu / var).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # Negative Kelly gets zero in a long-only book.
    return k.clip(lower=floor, upper=cap).fillna(floor)


def risk_reward_ratio_from_scenarios(scenarios: pd.DataFrame, eps: float = EPS) -> pd.Series:
    """Expected positive payoff divided by expected negative payoff."""
    if scenarios.empty:
        return pd.Series(dtype=float)
    upside = scenarios.clip(lower=0.0).mean(axis=0)
    downside = (-scenarios.clip(upper=0.0)).mean(axis=0)
    return (upside / (downside + eps)).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def liquidity_score(market_cap_or_volume: pd.DataFrame | np.ndarray | None, columns: list[str], window: int = 21) -> pd.Series:
    if market_cap_or_volume is None:
        return pd.Series(1.0, index=columns)
    df = _as_returns_frame(market_cap_or_volume, columns=columns).reindex(columns=columns).fillna(0.0)
    if df.empty:
        return pd.Series(1.0, index=columns)
    liq = df.tail(window).mean(axis=0)
    med = float(np.nanmedian(liq.values)) if len(liq) else 0.0
    if not np.isfinite(med) or med <= EPS:
        return pd.Series(1.0, index=columns)
    return (liq / med).clip(lower=0.0, upper=10.0).replace([np.inf, -np.inf], np.nan).fillna(1.0)


def _zscore(s: pd.Series, clip: float = 3.0) -> pd.Series:
    sd = float(s.std(ddof=1)) if len(s) > 1 else 0.0
    if not np.isfinite(sd) or sd <= EPS:
        return pd.Series(0.0, index=s.index)
    z = (s - float(s.mean())) / sd
    return z.clip(-clip, clip).fillna(0.0)


def _sigmoid(x: pd.Series | float) -> pd.Series | float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


@dataclass
class PortfolioSignalDiagnostics:
    overlay_fraction: float
    portfolio_kelly: float
    portfolio_risk_reward: float
    portfolio_garch_vol: float
    portfolio_vol_regime: float
    composite_quality: float
    selected_count: int
    comments: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_asset_signal_table(
    *,
    etfs_list: list[str],
    historical_returns: pd.DataFrame | np.ndarray,
    pred_returns: np.ndarray | pd.DataFrame | None = None,
    market_cap_history: pd.DataFrame | np.ndarray | None = None,
    return_type: str = "log-returns",
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    cfg = config or {}
    hist = _as_returns_frame(historical_returns, columns=etfs_list).reindex(columns=etfs_list).fillna(0.0)
    price = price_proxy_from_returns(hist, return_type=return_type)
    scen = _scenario_frame(pred_returns, etfs_list) if pred_returns is not None else hist.tail(252)
    if scen.empty:
        scen = hist.tail(252)

    er = rolling_efficiency_ratio(price, int(cfg.get("er_window", 21)))
    atr_regime = rolling_atr_regime(price, int(cfg.get("atr_short_window", 21)), int(cfg.get("atr_long_window", 126)))
    liq = liquidity_score(market_cap_history, etfs_list, int(cfg.get("liquidity_window", 21)))
    breakout = rolling_breakout_strength(price, int(cfg.get("breakout_window", 63)))
    spike = normalized_return_spike(hist, int(cfg.get("spike_window", 21)))
    pullback = pullback_ratio(price, int(cfg.get("pullback_window", 63)))
    move = net_move(hist, int(cfg.get("net_move_window", 21)))
    garch_vol = garch11_vol_forecast(
        hist,
        omega=float(cfg.get("garch_omega", 1e-6)),
        alpha=float(cfg.get("garch_alpha", 0.08)),
        beta=float(cfg.get("garch_beta", 0.90)),
    )
    rr = risk_reward_ratio_from_scenarios(scen.reindex(columns=etfs_list).fillna(0.0))
    kelly = kelly_fraction_from_scenarios(
        scen.reindex(columns=etfs_list).fillna(0.0),
        cap=float(cfg.get("kelly_cap", 0.25)),
        floor=float(cfg.get("kelly_floor", 0.0)),
    )

    # Reward efficient trend, strong but not overextended breakout, good liquidity, good RR, and positive net move.
    # Penalize excessive vol regime and extreme one-day spikes.
    pullback_good = 1.0 - (pullback - float(cfg.get("preferred_pullback", 0.25))).abs().clip(0.0, 1.0)
    quality_raw = (
        float(cfg.get("w_er", 0.22)) * _zscore(er) +
        float(cfg.get("w_breakout", 0.16)) * _zscore(breakout) +
        float(cfg.get("w_liquidity", 0.14)) * _zscore(np.log1p(liq)) +
        float(cfg.get("w_risk_reward", 0.18)) * _zscore(np.log1p(rr)) +
        float(cfg.get("w_net_move", 0.14)) * _zscore(move) +
        float(cfg.get("w_pullback", 0.08)) * _zscore(pullback_good) -
        float(cfg.get("w_vol_regime", 0.16)) * _zscore(atr_regime) -
        float(cfg.get("w_spike", 0.10)) * _zscore(spike.abs())
    )
    quality = pd.Series(_sigmoid(quality_raw), index=etfs_list).clip(0.05, 0.95)

    table = pd.DataFrame({
        "efficiency_ratio": er,
        "atr_vol_regime": atr_regime,
        "liquidity_score": liq,
        "breakout_strength": breakout,
        "normalized_spike": spike,
        "pullback_ratio": pullback,
        "net_move": move,
        "garch_vol_forecast": garch_vol,
        "risk_reward_ratio": rr,
        "kelly_fraction": kelly,
        "signal_quality": quality,
    }).reindex(etfs_list)
    return table.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def apply_smart_position_sizing(
    *,
    asset_weights: pd.Series,
    signal_table: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.Series:
    """Tilt selected asset weights by quality, fractional Kelly and risk-reward."""
    cfg = config or {}
    w = asset_weights.astype(float).reindex(signal_table.index).fillna(0.0).clip(lower=0.0)
    if w.sum() <= EPS:
        return w

    q = signal_table["signal_quality"].clip(0.05, 0.95)
    k = signal_table["kelly_fraction"].clip(lower=0.0, upper=float(cfg.get("kelly_cap", 0.25)))
    rr = signal_table["risk_reward_ratio"].clip(lower=0.0, upper=float(cfg.get("risk_reward_cap", 5.0)))

    # Convert Kelly and RR to gentle multipliers so they do not fully dominate optimizer weights.
    k_mult = 0.50 + 0.50 * (k / max(float(cfg.get("kelly_cap", 0.25)), EPS))
    rr_mult = (0.50 + 0.50 * (np.log1p(rr) / np.log1p(float(cfg.get("risk_reward_cap", 5.0))))).clip(0.25, 1.25)
    tilted = w * q * k_mult * rr_mult

    min_w = float(cfg.get("min_active_weight", 0.0))
    max_w = cfg.get("max_active_weight")
    if min_w > 0:
        tilted[tilted < min_w] = 0.0
    if tilted.sum() <= EPS:
        tilted = w.copy()
    else:
        tilted = tilted / tilted.sum()
    if max_w is not None:
        cap = float(max_w)
        for _ in range(len(tilted) + 1):
            over = tilted > cap
            if not over.any():
                break
            excess = float((tilted[over] - cap).sum())
            tilted[over] = cap
            under = ~over & (tilted > 0)
            if not under.any() or excess <= EPS:
                break
            tilted[under] += excess * tilted[under] / tilted[under].sum()
        tilted = tilted / max(tilted.sum(), EPS)
    return tilted.fillna(0.0)


def portfolio_signal_diagnostics(
    *,
    asset_weights: pd.Series,
    signal_table: pd.DataFrame,
    historical_returns: pd.DataFrame | np.ndarray,
    pred_returns: np.ndarray | pd.DataFrame | None = None,
    config: dict[str, Any] | None = None,
) -> PortfolioSignalDiagnostics:
    cfg = config or {}
    cols = list(asset_weights.index)
    hist = _as_returns_frame(historical_returns, columns=cols).reindex(columns=cols).fillna(0.0)
    scen = _scenario_frame(pred_returns, cols) if pred_returns is not None else hist.tail(252)
    w = asset_weights.reindex(cols).fillna(0.0).astype(float).values
    if w.sum() > 1.0 + 1e-6:
        w = w / w.sum()

    if len(cols) == 0 or np.abs(w).sum() <= EPS:
        return PortfolioSignalDiagnostics(1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0, "empty book")

    port_hist = pd.Series(hist.values @ w, index=hist.index)
    if not scen.empty and scen.shape[1] == len(w):
        port_scen = pd.Series(scen.values @ w)
    else:
        port_scen = port_hist.tail(252)

    mu = float(port_scen.mean())
    var = float(port_scen.var(ddof=1)) if len(port_scen) > 1 else 0.0
    raw_kelly = mu / max(var, EPS)
    portfolio_kelly = float(np.clip(raw_kelly, 0.0, float(cfg.get("portfolio_kelly_cap", 1.0))))
    upside = float(port_scen.clip(lower=0.0).mean())
    downside = float((-port_scen.clip(upper=0.0)).mean())
    rr = float(upside / (downside + EPS))

    port_vol = float(garch11_vol_forecast(pd.DataFrame({"portfolio": port_hist})).iloc[0])
    short_vol = float(port_hist.tail(int(cfg.get("vol_short_window", 21))).std(ddof=1))
    long_vol = float(port_hist.tail(int(cfg.get("vol_long_window", 126))).std(ddof=1))
    vol_regime = short_vol / max(long_vol, EPS) if np.isfinite(long_vol) and long_vol > EPS else 1.0
    comp_quality = float(np.average(signal_table["signal_quality"].reindex(cols).fillna(0.5), weights=np.maximum(w, 0))) if np.maximum(w, 0).sum() > EPS else 0.5

    target_vol = float(cfg.get("target_monthly_vol", 0.035))
    vol_scale = float(np.clip(target_vol / max(port_vol, EPS), float(cfg.get("min_overlay", 0.30)), 1.0))
    rr_scale = float(np.clip(np.log1p(rr) / np.log1p(float(cfg.get("target_risk_reward", 1.5))), float(cfg.get("min_overlay", 0.30)), 1.0))
    kelly_scale = float(np.clip(portfolio_kelly / max(float(cfg.get("target_portfolio_kelly", 0.50)), EPS), float(cfg.get("min_overlay", 0.30)), 1.0))
    quality_scale = float(np.clip(comp_quality / max(float(cfg.get("target_signal_quality", 0.55)), EPS), float(cfg.get("min_overlay", 0.30)), 1.0))

    # Multiplicative de-risking: exposure falls when any signal is poor.
    overlay = float(np.clip(vol_scale * rr_scale * kelly_scale * quality_scale, float(cfg.get("min_overlay", 0.30)), 1.0))
    return PortfolioSignalDiagnostics(
        overlay_fraction=overlay,
        portfolio_kelly=portfolio_kelly,
        portfolio_risk_reward=rr,
        portfolio_garch_vol=port_vol,
        portfolio_vol_regime=float(vol_regime),
        composite_quality=comp_quality,
        selected_count=int((asset_weights > float(cfg.get("active_weight_threshold", 1e-4))).sum()),
        comments="multiplicative overlay from Kelly x risk-reward x volatility clustering x composite signal quality",
    )


def smart_rebalance_diagnostics(
    *,
    etfs_list: list[str],
    historical_returns: pd.DataFrame | np.ndarray,
    pred_returns: np.ndarray | pd.DataFrame | None,
    market_cap_history: pd.DataFrame | np.ndarray | None,
    selected_asset_weights: pd.Series,
    return_type: str = "log-returns",
    config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, PortfolioSignalDiagnostics, pd.Series]:
    cfg = config or {}
    signals = build_asset_signal_table(
        etfs_list=etfs_list,
        historical_returns=historical_returns,
        pred_returns=pred_returns,
        market_cap_history=market_cap_history,
        return_type=return_type,
        config=cfg,
    )
    adjusted = apply_smart_position_sizing(asset_weights=selected_asset_weights, signal_table=signals, config=cfg)
    diag = portfolio_signal_diagnostics(
        asset_weights=adjusted,
        signal_table=signals,
        historical_returns=historical_returns,
        pred_returns=pred_returns,
        config=cfg,
    )
    if bool(cfg.get("allow_cash_overlay", True)):
        adjusted = adjusted * diag.overlay_fraction
    signals = signals.assign(original_weight=selected_asset_weights.reindex(signals.index).fillna(0.0), adjusted_weight=adjusted.reindex(signals.index).fillna(0.0))
    return signals, diag, adjusted
