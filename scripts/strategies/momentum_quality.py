from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from scripts.calculations.covariance import clean_covariance, nearest_psd
from scripts.strategies.base import BaseStrategy


class MomentumVolatilityStrategy(BaseStrategy):
    """
    Long-only momentum / volatility strategy.

    Score_i = recent momentum_i / realized volatility_i.
    Keeps top_k assets and normalizes with max-weight cap.
    """

    def __init__(self, config, historical_returns, pred_returns, market_cap, w_previous):
        super().__init__(config, historical_returns, pred_returns, market_cap, w_previous)
        lookback = int(self.config.get("momentum_lookback", 63))
        top_k = int(self.config.get("top_k", min(30, self.historical_returns.shape[1])))

        x = self.historical_returns[-lookback:]
        momentum = np.nansum(x, axis=0)
        vol = np.nanstd(x, axis=0, ddof=1)
        score = momentum / np.maximum(vol, 1e-8)
        score = np.nan_to_num(score, nan=-np.inf, neginf=-np.inf, posinf=0.0)

        k = max(1, min(top_k, score.size))
        selected = np.argsort(score)[-k:]

        raw = np.zeros(score.size)
        positive_score = np.maximum(score[selected], 0.0)
        if positive_score.sum() <= 1e-12:
            raw[selected] = 1.0 / k
        else:
            raw[selected] = positive_score / positive_score.sum()

        self.weights = raw


class LiquidityMomentumStrategy(BaseStrategy):
    """
    Momentum strategy with liquidity awareness.

    Good when you want to avoid tiny or illiquid ETFs dominating due to noisy returns.
    """

    def __init__(self, config, historical_returns, pred_returns, market_cap, w_previous):
        super().__init__(config, historical_returns, pred_returns, market_cap, w_previous)

        lookback = int(self.config.get("momentum_lookback", 63))
        top_k = int(self.config.get("top_k", min(40, self.historical_returns.shape[1])))
        liquidity_power = float(self.config.get("liquidity_power", 0.25))

        x = self.historical_returns[-lookback:]
        momentum = np.nansum(x, axis=0)
        vol = np.nanstd(x, axis=0, ddof=1)
        score = momentum / np.maximum(vol, 1e-8)

        if self.market_cap is not None:
            liq = np.nanmean(np.maximum(self.market_cap, 0.0), axis=0)
            liq = np.nan_to_num(liq, nan=0.0)
            positive_liq = liq[liq > 0]
            if positive_liq.size > 0:
                liq_score = (liq / np.nanmedian(positive_liq)) ** liquidity_power
                score = score * liq_score

        score = np.nan_to_num(score, nan=-np.inf, neginf=-np.inf, posinf=0.0)

        k = max(1, min(top_k, score.size))
        selected = np.argsort(score)[-k:]

        raw = np.zeros(score.size)
        s = np.maximum(score[selected], 0.0)
        raw[selected] = 1.0 / k if s.sum() <= 1e-12 else s / s.sum()
        self.weights = raw


class MinimumCorrelationStrategy(BaseStrategy):
    """
    Long-only minimum correlation strategy.

    Minimizes w.T @ Corr @ w, not covariance variance. Good as a diversification benchmark.
    """

    def __init__(self, config, historical_returns, pred_returns, market_cap, w_previous):
        super().__init__(config, historical_returns, pred_returns, market_cap, w_previous)

        x = np.nan_to_num(self.historical_returns, nan=0.0)
        cov = nearest_psd(clean_covariance(x, method=self.config.get("cov_method", "oas"), detone_n=0))
        std = np.sqrt(np.maximum(np.diag(cov), 1e-12))
        corr = cov / np.outer(std, std)
        corr = nearest_psd(corr)

        n = x.shape[1]
        x0 = np.ones(n) / n
        max_w = max(self.max_weight, 1.0 / n)
        bounds = [(0.0, max_w)] * n
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        def objective(w):
            turnover = np.sum(np.abs(w - self.w_previous[:n])) if self.w_previous.size == n else 0.0
            turnover_penalty = float(self.config.get("turnover_penalty", 0.0))
            return float(w.T @ corr @ w + turnover_penalty * turnover)

        res = minimize(objective, x0=x0, method="SLSQP", bounds=bounds, constraints=cons)
        self.weights = res.x if res.success else x0
