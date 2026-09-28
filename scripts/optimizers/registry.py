from typing import Type
from scripts.optimizers.base import BaseCVaR
from scripts.optimizers.bayessian import BayessianCVaR
from scripts.optimizers.cdar import BayessianCDaR
from scripts.optimizers.black_litterman import BlackLitterman
from scripts.optimizers.markowitz import MarkowitzCVaR

CVAR_REGISTRY: dict[str, Type[BaseCVaR]] = {
    "bayessian": BayessianCVaR,
    "cdar": BayessianCDaR,
    "markowitz": MarkowitzCVaR,
    "black_litterman": BlackLitterman,
}
