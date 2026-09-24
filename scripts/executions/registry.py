from typing import Type
from scripts.executions.base import BaseExecution

EXEC_REGISTRY: dict[str, Type[BaseExecution]] = {
    "base_exec": BaseExecution
}
