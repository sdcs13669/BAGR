"""Strategy registry: registration, lookup, and listing."""

from typing import Dict, Type

from ...core.base_reader import BaseReader
from .random import RandomStrategy
from .bfs import BfsStrategy
from .max_degree import MaxFrontierDegreeStrategy

_registry: Dict[str, Type[BaseReader]] = {
    'random': RandomStrategy,
    'bfs': BfsStrategy,
    'max-frontier-degree': MaxFrontierDegreeStrategy,
}


def register_strategy(name: str, cls: Type[BaseReader]) -> None:
    _registry[name] = cls


def get_strategy(name: str) -> BaseReader:
    if name not in _registry:
        raise ValueError(f"unknown strategy '{name}', available: {sorted(_registry)}")
    return _registry[name]()


def list_strategies() -> Dict[str, Type[BaseReader]]:
    return dict(_registry)
