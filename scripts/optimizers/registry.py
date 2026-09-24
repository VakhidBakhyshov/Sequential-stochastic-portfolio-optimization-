from typing import Type
from scripts.optimizers.base import BaseCVaR
from scripts.optimizers.bayessian import BayessianCVaR
from scripts.optimizers.black_litterman import BlackLitterman
from scripts.optimizers.markowitz import MarkowitzCVaR

CVAR_REGISTRY: dict[str, Type[BaseCVaR]] = {
    "bayessian": BayessianCVaR,
    "markowitz": MarkowitzCVaR,
    "black_litterman": BlackLitterman,
}
