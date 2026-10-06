from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np

EPS = 1e-12


def _clip(x: float, lo: float, hi: float) -> float:
    return float(np.clip(float(x), float(lo), float(hi)))


def _as_2d_float(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise ValueError("Expected a 2D array shaped T x N.")
    arr = np.where(np.isfinite(arr), arr, np.nan)
    return arr


def _offdiag_mean(corr: np.ndarray) -> float:
    if corr.shape[0] <= 1:
        return 0.0
    mask = ~np.eye(corr.shape[0], dtype=bool)
    vals = corr[mask]
    vals = vals[np.isfinite(vals)]
    return float(np.nanmean(vals)) if vals.size else 0.0


@dataclass
class DynamicParameterController:
    """
    Causal monthly controller for optimizer parameters.

    Use at the start of month t to produce optimizer params from information known
    before month t is traded: previous realized portfolio returns + historical
    returns/liquidity up to the rebalance date.
    """

    enabled: bool = True
    returns_type: str = "log-returns"  # "log-returns" or "returns"

    # Base values are copied from optimizer config on initialization.
    base_confidence_level: float = 0.99
    base_turnover_penalty: float = 1.0

    # Bounds.
    confidence_min: float = 0.90
    confidence_max: float = 0.995
    turnover_min: float = 0.01
    turnover_max: float = 10.0

    # Step sizes.
    confidence_step: float = 0.015
    turnover_step: float = 0.35

    # Direction: +1 means parameter increases when score is positive.
    # Set confidence_direction=-1 if you want CVaR to become stricter after losses/risk-off regimes.
    confidence_direction: float = 1.0
    turnover_direction: float = 1.0

    # Score composition.
    performance_weight: float = 0.70
    regime_weight: float = 0.30
    streak_sensitivity: float = 0.55
    sharpe_sensitivity: float = 0.35
    max_streak_used: int = 6

    # EWMA state for realized portfolio returns.
    ewma_decay: float = 0.80
    flat_return_zone: float = 1e-6
    streak: int = 0
    ewma_return: float = 0.0
    ewma_var: float = 1e-6
    n_updates: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_optimizer_config(cls, optimizer_config: dict[str, Any]) -> "DynamicParameterController":
        dyn = optimizer_config.get("dynamic_parameters", {}) or {}
        return cls(
            enabled=bool(dyn.get("enabled", True)),
            returns_type=str(dyn.get("returns_type", "log-returns")),
            base_confidence_level=float(optimizer_config.get("confidence_level", 0.99)),
            base_turnover_penalty=float(optimizer_config.get("turnover_penalty", 1.0)),
            confidence_min=float(dyn.get("confidence_min", 0.90)),
            confidence_max=float(dyn.get("confidence_max", 0.995)),
            turnover_min=float(dyn.get("turnover_min", 0.01)),
            turnover_max=float(dyn.get("turnover_max", 10.0)),
            confidence_step=float(dyn.get("confidence_step", 0.015)),
            turnover_step=float(dyn.get("turnover_step", 0.35)),
            confidence_direction=float(dyn.get("confidence_direction", 1.0)),
            turnover_direction=float(dyn.get("turnover_direction", 1.0)),
            performance_weight=float(dyn.get("performance_weight", 0.70)),
            regime_weight=float(dyn.get("regime_weight", 0.30)),
            streak_sensitivity=float(dyn.get("streak_sensitivity", 0.55)),
            sharpe_sensitivity=float(dyn.get("sharpe_sensitivity", 0.35)),
            max_streak_used=int(dyn.get("max_streak_used", 6)),
            ewma_decay=float(dyn.get("ewma_decay", 0.80)),
            flat_return_zone=float(dyn.get("flat_return_zone", 1e-6)),
        )

    def update_after_realized_return(self, realized_return: float) -> None:
        """Call after month t result is known; it affects month t+1 parameters."""
        r = float(realized_return)
        if not np.isfinite(r):
            return

        if r > self.flat_return_zone:
            s = 1
        elif r < -self.flat_return_zone:
            s = -1
        else:
            s = 0

        if s == 0:
            self.streak = 0
        elif self.streak == 0 or np.sign(self.streak) != s:
            self.streak = s
        else:
            self.streak += s

        if self.n_updates == 0:
            self.ewma_return = r
            self.ewma_var = max(r * r, 1e-6)
        else:
            d = self.ewma_decay
            prev_mean = self.ewma_return
            self.ewma_return = d * self.ewma_return + (1.0 - d) * r
            self.ewma_var = d * self.ewma_var + (1.0 - d) * (r - prev_mean) ** 2

        self.n_updates += 1

    def performance_score(self) -> float:
        if self.n_updates == 0:
            return 0.0

        capped_streak = _clip(self.streak, -self.max_streak_used, self.max_streak_used)
        streak_score = capped_streak / max(float(self.max_streak_used), 1.0)
        ewma_sharpe = self.ewma_return / np.sqrt(max(self.ewma_var, EPS))
        return float(np.tanh(self.streak_sensitivity * streak_score + self.sharpe_sensitivity * ewma_sharpe))

    def _simple_returns(self, historical_returns: np.ndarray) -> np.ndarray:
        x = _as_2d_float(historical_returns)
        if self.returns_type == "log-returns":
            x = np.exp(np.clip(x, -1.0, 1.0)) - 1.0
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    def regime_metrics(self, historical_returns: np.ndarray, market_cap: np.ndarray | None = None) -> dict[str, float]:
        """
        Daily ETF regime metrics from information known at rebalance date.
        market_cap is used here as dollar-volume/liquidity proxy if it is price * volume.
        """
        r = self._simple_returns(historical_returns)
        t, n = r.shape
        if t < 21 or n == 0:
            return {
                "regime_score": 0.0,
                "market_momentum_63d": 0.0,
                "vol_ratio_21_126": 1.0,
                "drawdown_252d": 0.0,
                "breadth_63d": 0.5,
                "corr_stress_63d": 0.0,
                "liquidity_trend": 0.0,
            }

        # Liquidity/market weights. Use last ~month average to avoid one-day noise.
        weights = np.ones(n) / n
        if market_cap is not None:
            mc = _as_2d_float(market_cap)
            if mc.shape[1] == n:
                recent_mc = np.nanmean(np.where(mc[-min(len(mc), 21):] > 0, mc[-min(len(mc), 21):], np.nan), axis=0)
                recent_mc = np.nan_to_num(recent_mc, nan=0.0, posinf=0.0, neginf=0.0)
                if recent_mc.sum() > EPS:
                    weights = recent_mc / recent_mc.sum()

        market_r = r @ weights
        last63 = market_r[-min(t, 63):]
        last126 = market_r[-min(t, 126):]
        last252 = market_r[-min(t, 252):]

        market_mom_63 = float(np.prod(1.0 + last63) - 1.0)
        vol21 = float(np.std(market_r[-min(t, 21):], ddof=1) * np.sqrt(252)) if t >= 22 else 0.0
        vol126 = float(np.std(last126, ddof=1) * np.sqrt(252)) if len(last126) >= 22 else max(vol21, EPS)
        vol_ratio = vol21 / max(vol126, EPS)

        equity_curve = np.cumprod(1.0 + last252)
        peak = np.maximum.accumulate(equity_curve)
        drawdown = float(equity_curve[-1] / max(peak[-1], EPS) - 1.0)

        asset_mom_63 = np.prod(1.0 + r[-min(t, 63):], axis=0) - 1.0
        breadth = float(np.mean(asset_mom_63 > 0.0))

        corr_window = r[-min(t, 63):]
        if corr_window.shape[0] >= 5 and n > 1:
            corr = np.corrcoef(corr_window, rowvar=False)
            corr_stress = _offdiag_mean(corr)
        else:
            corr_stress = 0.0

        liquidity_trend = 0.0
        if market_cap is not None:
            mc = _as_2d_float(market_cap)
            if mc.shape[1] == n and mc.shape[0] >= 42:
                dollar_liq = np.nanmean(np.where(mc > 0, mc, np.nan), axis=1)
                recent_liq = np.nanmean(dollar_liq[-21:])
                long_liq = np.nanmean(dollar_liq[-min(len(dollar_liq), 126):])
                if np.isfinite(recent_liq) and np.isfinite(long_liq) and long_liq > EPS:
                    liquidity_trend = float(np.tanh(np.log(recent_liq / long_liq)))

        # Convert metrics to comparable [-1, 1]-ish components.
        momentum_component = np.tanh(8.0 * market_mom_63)
        breadth_component = 2.0 * (breadth - 0.5)
        vol_component = np.tanh(vol_ratio - 1.0)
        drawdown_component = np.tanh(6.0 * abs(min(drawdown, 0.0)))
        corr_component = np.tanh(max(corr_stress, 0.0))

        raw_score = (
            1.10 * momentum_component
            + 0.80 * breadth_component
            + 0.30 * liquidity_trend
            - 0.70 * vol_component
            - 0.60 * drawdown_component
            - 0.50 * corr_component
        )
        regime_score = _clip(raw_score / 3.0, -1.0, 1.0)

        return {
            "regime_score": regime_score,
            "market_momentum_63d": market_mom_63,
            "vol_ratio_21_126": float(vol_ratio),
            "drawdown_252d": drawdown,
            "breadth_63d": breadth,
            "corr_stress_63d": float(corr_stress),
            "liquidity_trend": float(liquidity_trend),
        }

    def next_optimizer_config(
        self,
        base_optimizer_config: dict[str, Any],
        historical_returns: np.ndarray,
        market_cap: np.ndarray | None = None,
        date: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return a copied optimizer config with updated params and diagnostics."""
        out = copy.deepcopy(base_optimizer_config)
        if not self.enabled:
            info = {
                "date": date,
                "dynamic_enabled": False,
                "confidence_level": float(out.get("confidence_level", self.base_confidence_level)),
                "turnover_penalty": float(out.get("turnover_penalty", self.base_turnover_penalty)),
            }
            return out, info

        perf = self.performance_score()
        regime = self.regime_metrics(historical_returns, market_cap)
        total_score = _clip(self.performance_weight * perf + self.regime_weight * regime["regime_score"], -1.0, 1.0)

        confidence_level = _clip(
            self.base_confidence_level + self.confidence_direction * self.confidence_step * total_score,
            self.confidence_min,
            self.confidence_max,
        )
        turnover_penalty = _clip(
            self.base_turnover_penalty * float(np.exp(self.turnover_direction * self.turnover_step * total_score)),
            self.turnover_min,
            self.turnover_max,
        )

        out["confidence_level"] = confidence_level
        out["turnover_penalty"] = turnover_penalty

        # Optional: dynamically adjust concentration cap if you add these keys to dynamic_parameters.
        dyn = base_optimizer_config.get("dynamic_parameters", {}) or {}
        if "max_weight_step" in dyn and "max_weight" in out:
            max_weight = _clip(
                float(out["max_weight"]) * float(np.exp(float(dyn["max_weight_step"]) * total_score)),
                float(dyn.get("max_weight_min", out.get("min_weight", 0.0))),
                float(dyn.get("max_weight_max", out["max_weight"])),
            )
            out["max_weight"] = max_weight

        info = {
            "date": date,
            "dynamic_enabled": True,
            "performance_score": perf,
            "streak": self.streak,
            "ewma_return": self.ewma_return,
            "ewma_vol": float(np.sqrt(max(self.ewma_var, EPS))),
            "total_score": total_score,
            "confidence_level": confidence_level,
            "turnover_penalty": turnover_penalty,
            **regime,
        }
        return out, info
