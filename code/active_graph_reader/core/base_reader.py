"""BaseReader: the common interface for all readers (fixed and learned)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    from ..engine.graph_state import GraphState


class BaseReader(ABC):
    """Abstract reader base class.

    Subclasses implement:
    - select: pick the next node to reveal from the current graph state.
    - name: the reader name.

    Optional hooks:
    - on_new_frontier: frontier expansion notification (stateful readers
      such as BFS).
    - reset: clear internal state.
    - train/eval: switch training/eval mode.
    """

    def __init__(self):
        super().__init__()
        self._training = False

    @abstractmethod
    def select(self, graph_state: GraphState) -> Optional[int]:
        """Pick the next node to reveal from the frontier; None if empty."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Reader name (e.g. 'random', 'bfs', 'evidence')."""
        ...

    def on_new_frontier(self, edges: List[Tuple[int, int]]) -> None:
        pass

    def reset(self) -> None:
        pass

    def train(self, mode: bool = True):
        self._training = mode
        return self

    def eval(self):
        self._training = False
        return self

    @property
    def training(self) -> bool:
        return self._training

    @training.setter
    def training(self, value: bool):
        self._training = value
