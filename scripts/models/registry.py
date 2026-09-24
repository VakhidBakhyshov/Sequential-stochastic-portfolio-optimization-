from typing import Type

from models.base import BaseModel
from models.bayessian import BayessianModel
from models.ml_regression import MLRegressionModel

MODEL_REGISTRY: dict[str, Type[BaseModel]] = {
    "bayessian": BayessianModel,
    "ml_regression": MLRegressionModel
}
