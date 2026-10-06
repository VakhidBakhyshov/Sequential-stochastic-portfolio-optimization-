import numpy as np
from beartype import beartype
from scripts.strategies.base import BaseStrategy

class EqualWeightStrategy(BaseStrategy):
    def __init__(self, config, historical_returns, pred_returns, market_cap, w_previous):
        super().__init__(config, historical_returns, pred_returns, market_cap, w_previous)
        n_assets = historical_returns.shape[1]
        self.weights = np.ones(n_assets) / n_assets

class InverseVolatilityStrategy(BaseStrategy):
    def __init__(self, config, historical_returns, pred_returns, market_cap, w_previous):
        super().__init__(config, historical_returns, pred_returns, market_cap, w_previous)
        # vol = np.std(historical_returns, axis=0, ddof=1)
        # inv_vol = 1.0 / (vol + 1e-10)
        # sum_inv_vol = np.sum(inv_vol)
        
        valid_counts = np.sum(~np.isnan(historical_returns), axis=0)
        vol = np.nanstd(historical_returns, axis=0, ddof=1)
        vol = np.where((valid_counts < 20) | (vol <= 1e-10), np.inf, vol) # invalidate unreliable estimates
        
        inv_vol = 1.0 / (vol + 1e-10)
        inv_vol = np.nan_to_num(inv_vol, nan=0.0)    # set NaNs to 0
        
        inv_vol.sum()
        if inv_vol.sum() <= 1e-12:
            self.weights = np.ones_like(inv_vol) / len(inv_vol)
        else:
            self.weights = inv_vol / inv_vol.sum()


class LiquidityWeightedStrategy(BaseStrategy):
    def __init__(self, config, historical_returns, pred_returns, market_cap, w_previous):
        
        super().__init__(config, historical_returns, pred_returns, market_cap, w_previous)
        
        n_assets = historical_returns.shape[1]
        
        if self.market_cap is None:
            self.weights = np.ones(n_assets) / n_assets
            return
        
        # liquidity = np.sum(market_cap, axis=0)   # total_market_cap, using the sum of market_cap for all_days

        liquidity = np.nanmean(np.maximum(self.market_cap, 0.0), axis=0)
        liquidity = np.nan_to_num(liquidity, nan=0.0, posinf=0.0, neginf=0.0)
        
        total = np.sum(liquidity)
        if total <= 1e-12:
            self.weights = np.ones(n_assets) / n_assets
        else:
            self.weights = liquidity / total
