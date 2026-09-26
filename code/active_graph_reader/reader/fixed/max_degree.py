"""Max-frontier-degree strategy: reveal the frontier node with the most
connections to the revealed subgraph."""

from typing import Optional

from ...core.base_reader import BaseReader
from ...engine.graph_state import GraphState


class MaxFrontierDegreeStrategy(BaseReader):
    """Pick the frontier node with the largest frontier_degree (stateless)."""

    @property
    def name(self) -> str:
        return 'max-frontier-degree'

    def select(self, graph_state: GraphState) -> Optional[int]:
        frontier = graph_state.frontier_edges
        if not frontier:
            return None

        best_node = None
        best_degree = -1

        for edge in frontier:
            unrevealed = edge[1]
            deg = graph_state.get_frontier_degree(unrevealed)
            if deg > best_degree:
                best_degree = deg
                best_node = unrevealed

        return best_node
