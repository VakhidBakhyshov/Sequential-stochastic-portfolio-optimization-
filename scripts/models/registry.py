from typing import Type

from models.base import BaseModel
from models.bayessian import BayessianModel

MODEL_REGISTRY: dict[str, Type[BaseModel]] = {
    "bayessian": BayessianModel,
}

# The ML regression model pulls in scikit-learn's ensemble extensions, which some Windows
# Application Control policies refuse to load. It is not used by the Bayesian pipeline, so it is
try:
    from models.ml_regression import MLRegressionModel
    MODEL_REGISTRY["ml_regression"] = MLRegressionModel
except Exception as _exc:  # pragma: no cover
    import warnings
    warnings.warn(f"ml_regression model unavailable: {type(_exc).__name__}: {_exc}")
