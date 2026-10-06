from typing import Type

from models.base import BaseModel
from models.bayessian import BayessianModel
from models.ml_regression import MLRegressionModel

MODEL_REGISTRY: dict[str, Type[BaseModel]] = {
    "bayessian": BayessianModel,
    # "ml_regression": MLRegressionModel
}

try:
    from models.ml_regression import MLRegressionModel
    MODEL_REGISTRY["ml_regression"] = MLRegressionModel
except Exception as _exc:  # pragma: no cover
    import warnings
    warnings.warn(f"ml_regression model unavailable: {type(_exc).__name__}: {_exc}")
