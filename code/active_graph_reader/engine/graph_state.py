"""Graph state: the partially observable graph G_t during revelation."""

import bisect
import logging
from collections import defaultdict
from typing import Dict, Set, Tuple, List, Optional

import torch
import dgl

logger = logging.getLogger(__name__)


class GraphState:
    """Partially observable graph state G_t = (V_revealed, E_known, X_E, X_V).

    Encapsulates everything knowable during revelation and exposes query
    interfaces for policies; mutation happens via the reveal_action module.
    """

    __slots__ = (
        'g', 'seed_node', 'V_revealed', 'frontier_edges',
        'revealed_edge_keys', '_edge_key_to_id',
        '_frontier_degree', '_frontier_by_unrevealed',
        'n_nodes_total', 'n_edges_total',
        'has_edge_feat', 'edge_feat', 'edge_feat_dim',
        'has_node_feat', 'node_feat', 'node_feat_dim',
        'reveal_history', '_budget_used', '_budget_limit', '_initial_frontier',
        '_sorted_revealed', '_sorted_frontier_nodes', '_revealed_edge_ids',
    )

    def __init__(
        self,
        dgl_graph: dgl.DGLGraph,
        seed_node: int,
        edge_key_to_id: Optional[Dict[Tuple[int, int], int]] = None,
    ):
        self.g = dgl_graph
        self.seed_node = seed_node
        self.n_nodes_total = dgl_graph.num_nodes()
        self.n_edges_total = dgl_graph.num_edges()

        self.has_edge_feat = 'feat' in dgl_graph.edata
        self.edge_feat = dgl_graph.edata['feat'] if self.has_edge_feat else None
        self.edge_feat_dim = (
            self.edge_feat.shape[1] if self.has_edge_feat else 0
        )

        self.has_node_feat = 'feat' in dgl_graph.ndata
        self.node_feat = dgl_graph.ndata['feat'] if self.has_node_feat else None
        self.node_feat_dim = self.node_feat.shape[1] if self.has_node_feat else 0

        # Accept pre-built dict for reuse across trajectories (M4 optimization)
        if edge_key_to_id is not None:
            self._edge_key_to_id = edge_key_to_id
        else:
            self._edge_key_to_id: Dict[Tuple[int, int], int] = {}
            if self.has_edge_feat:
                src = dgl_graph.edges()[0]
                dst = dgl_graph.edges()[1]
                for eid in range(dgl_graph.num_edges()):
                    u, v = int(src[eid]), int(dst[eid])
                    key = (min(u, v), max(u, v))
                    if key not in self._edge_key_to_id:
                        self._edge_key_to_id[key] = eid

        self.V_revealed: Set[int] = set()
        self.frontier_edges: Set[Tuple[int, int]] = set()
        self.revealed_edge_keys: Set[Tuple[int, int]] = set()
        self._frontier_degree: Dict[int, int] = defaultdict(int)
        self._frontier_by_unrevealed: Dict[int, Set[Tuple[int, int]]] = defaultdict(set)

        self.reveal_history: List[int] = []
        self._budget_used: int = 0
        self._budget_limit: Optional[int] = None
        self._initial_frontier: List[Tuple[int, int]] = []

        self._sorted_revealed: List[int] = []
        self._sorted_frontier_nodes: List[int] = []
        self._revealed_edge_ids: List[int] = []

        self._initial_frontier = self._init_g0()

    def _init_g0(self) -> List[Tuple[int, int]]:
        s = self.seed_node
        self.V_revealed.add(s)
        self.reveal_history.append(s)
        self._sorted_revealed.append(s)  # single element, already sorted

        new_frontier: List[Tuple[int, int]] = []
        neighbors = self._get_neighbors(s)

        for nbr in neighbors:
            if nbr == s:
                continue
            edge = (s, nbr)
            self.frontier_edges.add(edge)
            self._frontier_degree[nbr] += 1
            self._frontier_by_unrevealed[nbr].add(edge)
            new_frontier.append(edge)
            bisect.insort(self._sorted_frontier_nodes, nbr)

        logger.debug("G_0 init: seed=%d, neighbors=%d, frontier edges=%d", s, len(neighbors), len(new_frontier))
        return new_frontier

    @property
    def is_frontier_empty(self) -> bool:
        return len(self.frontier_edges) == 0

    def get_frontier_edge_feat(self, u: int, v: int):
        """Frontier edge features; at least one endpoint must be revealed."""
        if not self.has_edge_feat:
            raise ValueError("graph has no edge features")
        if u not in self.V_revealed and v not in self.V_revealed:
            raise ValueError(f"frontier edge ({u}, {v}) has no revealed endpoint")
        return self._get_edge_feat(u, v)

    def get_frontier_degree(self, node: int) -> int:
        return self._frontier_degree.get(node, 0)

    def get_frontier_nodes(self) -> List[int]:
        return self._sorted_frontier_nodes  # already a list, sorted by construction

    def get_revealed_sorted(self) -> List[int]:
        return self._sorted_revealed  # already a list, sorted by construction

    def get_frontier_edges_of(self, node: int) -> List[Tuple[int, int]]:
        return list(self._frontier_by_unrevealed.get(node, []))

    def get_revealed_ratio(self) -> float:
        if self.n_nodes_total == 0:
            return 0.0
        return len(self.V_revealed) / self.n_nodes_total

    def get_budget_used(self) -> int:
        return self._budget_used

    def get_budget_ratio_remaining(self) -> float:
        """Remaining budget fraction 1 - t/B (0.5 if no budget limit set)."""
        if self._budget_limit is None or self._budget_limit <= 0:
            return 0.5
        return 1.0 - self._budget_used / self._budget_limit

    @property
    def n_revealed(self) -> int:
        return len(self.V_revealed)

    def _add_to_revealed(self, v: int) -> None:
        bisect.insort(self._sorted_revealed, v)

    def _add_frontier_node(self, node: int) -> None:
        bisect.insort(self._sorted_frontier_nodes, node)

    def _remove_frontier_node(self, node: int) -> None:
        idx = bisect.bisect_left(self._sorted_frontier_nodes, node)
        if idx < len(self._sorted_frontier_nodes) and self._sorted_frontier_nodes[idx] == node:
            self._sorted_frontier_nodes.pop(idx)

    def _add_revealed_edge_id(self, eid: int) -> None:
        self._revealed_edge_ids.append(eid)

    def _get_neighbors(self, node: int) -> List[int]:
        """Neighbor ids ascending (deterministic BFS); directed or undirected."""
        succ = set(self.g.successors(node).tolist())
        pred = set(self.g.predecessors(node).tolist())
        return sorted(succ | pred)

    def _get_edge_feat(self, u: int, v: int) -> torch.Tensor:
        """O(1) edge feature lookup (cloned)."""
        if not self.has_edge_feat:
            raise RuntimeError("graph has no edge features")
        key = (min(u, v), max(u, v))
        eid = self._edge_key_to_id.get(key)
        if eid is None:
            raise ValueError(f"edge ({u}, {v}) not in graph")
        return self.edge_feat[eid].clone()

    def _get_edge_feat_ref(self, u: int, v: int) -> torch.Tensor:
        """O(1) edge feature lookup (read-only view, no allocation)."""
        if not self.has_edge_feat:
            raise RuntimeError("graph has no edge features")
        key = (min(u, v), max(u, v))
        eid = self._edge_key_to_id.get(key)
        if eid is None:
            raise ValueError(f"edge ({u}, {v}) not in graph")
        return self.edge_feat[eid]

    def _increment_budget(self) -> None:
        self._budget_used += 1

    def stats(self) -> dict:
        return {
            'seed': self.seed_node,
            'n_total': self.n_nodes_total,
            'n_edges_total': self.n_edges_total,
            'n_revealed': len(self.V_revealed),
            'n_frontier_remaining': len(self.frontier_edges),
            'n_revealed_edge_feats': len(self.revealed_edge_keys),
            'budget_used': self._budget_used,
            'reveal_ratio': round(self.get_revealed_ratio(), 4),
            'reveal_history': self.reveal_history.copy(),
            'has_edge_feat': self.has_edge_feat,
            'edge_feat_dim': self.edge_feat_dim,
        }
