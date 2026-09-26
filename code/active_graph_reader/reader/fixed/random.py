"""Random strategy: pick a frontier edge uniformly at random and reveal its
unrevealed endpoint."""

import random
from typing import Optional

from ...core.base_reader import BaseReader
from ...engine.graph_state import GraphState


class RandomStrategy(BaseReader):
    """Uniformly pick a frontier edge and return its unrevealed endpoint.


    Deterministic: the RNG is seeded by (seed_node, n_nodes_total), the
    same formula as the random-walk baseline, so the same (graph, seed_i)
    trajectory is reproducible. Candidates are sorted by (u, v) before the
    draw, so the choice never depends on set iteration order.
    """

    def __init__(self):
        super().__init__()
        self._rng: Optional[random.Random] = None

    @property
    def name(self) -> str:
        return 'random'

    def reset(self) -> None:
        self._rng = None

    def select(self, graph_state: GraphState) -> Optional[int]:
        frontier = graph_state.frontier_edges
        if not frontier:
            return None
        if self._rng is None:
            self._rng = random.Random(
                graph_state.seed_node * 1_000_003 + graph_state.n_nodes_total)
        edge = self._rng.choice(sorted(frontier))
        return edge[1]
