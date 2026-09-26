"""Random-walk fixed strategy (AGR protocol adaptation).

Mirrors the neighborhood-random-sampling reading of the GraphSAGE family: the
agent walks along random edges from the current node and reveals a node the
first time it arrives there (revisits cost no budget), then continues from
the new node. Compared with random (uniform frontier sampling, breadth-type),
random-walk is locally correlated (depth-type).

Protocol adaptation: only frontier endpoints are returned each step; only the
current node's adjacency is consumed (no global structure); revisits are
budget-free; the walk length is capped (proportional to n_nodes) and
falling back to uniform frontier sampling counts as a teleport; the RNG is
arithmetically seeded by (seed_node, n_nodes_total) so trajectories are
reproducible."""

import random
from typing import Optional

import sys
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parents[1] / 'code'
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from active_graph_reader.core.base_reader import BaseReader  # noqa: E402
from active_graph_reader.engine.graph_state import GraphState  # noqa: E402


class RandomWalkStrategy(BaseReader):
    """Random-walk reveal: walk random edges; the first arrival at an
    unrevealed node is the next reveal."""

    def __init__(self, max_steps_factor: int = 4):
        super().__init__()
        self._cur: Optional[int] = None
        self._rng: Optional[random.Random] = None
        self._max_steps_factor = max_steps_factor

    @property
    def name(self) -> str:
        return 'random-walk'

    def reset(self) -> None:
        self._cur = None
        self._rng = None

    def _neighbors(self, state: GraphState, node: int):
        """Neighbors of the current node (local view); self-loops ignored."""
        g = state.g
        nbrs = set(g.successors(node).tolist()) | set(g.predecessors(node).tolist())
        nbrs.discard(node)
        return sorted(nbrs)

    def select(self, graph_state: GraphState) -> Optional[int]:
        frontier = graph_state.frontier_edges
        if not frontier:
            return None

        if self._cur is None:
            self._cur = graph_state.seed_node
            self._rng = random.Random(
                graph_state.seed_node * 1_000_003 + graph_state.n_nodes_total)

        cap = self._max_steps_factor * graph_state.n_nodes_total + 16
        cur = self._cur
        for _ in range(cap):
            nbrs = self._neighbors(graph_state, cur)
            if not nbrs:
                break
            nxt = self._rng.choice(nbrs)
            if nxt not in graph_state.V_revealed:
                self._cur = nxt
                return nxt
            cur = nxt

        edge = self._rng.choice(sorted(frontier))
        node = edge[1]
        self._cur = node
        return node
