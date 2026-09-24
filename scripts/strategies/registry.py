from typing import Type

from scripts.strategies.base import BaseStrategy
from scripts.strategies.simple import SimpleLongStrategy
from scripts.strategies.risk_parity import RiskParityStrategy
from scripts.strategies.benchmarks import EqualWeightStrategy, InverseVolatilityStrategy, LiquidityWeightedStrategy
from scripts.strategies.momentum_quality import MomentumVolatilityStrategy, LiquidityMomentumStrategy, MinimumCorrelationStrategy

STRATEGY_REGISTRY: dict[str, Type[BaseStrategy]] = {
    "simple_long": SimpleLongStrategy,
    "risk_parity": RiskParityStrategy,
    "equal_weight": EqualWeightStrategy,
    "inverse_volatility": InverseVolatilityStrategy,
    "liquidity_weighted": LiquidityWeightedStrategy,
    "momentum_volatility": MomentumVolatilityStrategy,
    "liquidity_momentum": LiquidityMomentumStrategy,
    "minimum_correlation": MinimumCorrelationStrategy,
}
