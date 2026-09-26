"""BFS strategy: reveal in frontier-edge discovery order (FIFO)."""

import collections
from typing import Optional, Deque, List, Tuple

from ...core.base_reader import BaseReader
from ...engine.graph_state import GraphState


class BfsStrategy(BaseReader):
    """Pick frontier edges in BFS order for deterministic level-order walks."""

    def __init__(self):
        super().__init__()
        self._queue: Deque[Tuple[int, int]] = collections.deque()

    @property
    def name(self) -> str:
        return 'bfs'

    def select(self, graph_state: GraphState) -> Optional[int]:
        while self._queue:
            edge = self._queue.popleft()
            unrevealed = edge[1]
            if unrevealed not in graph_state.V_revealed:
                return unrevealed
        return None

    def on_new_frontier(self, edges: List[Tuple[int, int]]) -> None:
        for edge in edges:
            self._queue.append(edge)

    def reset(self) -> None:
        self._queue.clear()
