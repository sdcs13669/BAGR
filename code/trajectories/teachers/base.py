"""Trajectory dataclass and the Teacher abstract base class.

Class/field names are stable so old trajectory files stay loadable
(legacy deserialization handled by tools/trajectory_io.py)."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

import torch
import dgl

from active_graph_reader.engine.graph_state import GraphState
from active_graph_reader.engine.reveal_action import reveal_node


@dataclass
class Trajectory:
    """One teacher reading trajectory; is_correct is always True (filtered)."""

    graph_idx: int
    seed_node: int
    reveal_ratio: float
    reveal_order: List[int]
    final_pred: int
    is_correct: bool
    teacher_type: str
    ce: float = 0.0


LEGACY_PICKLE_MODULE = 'm4.teachers.base'


class Teacher(ABC):
    """Teacher ABC: score the full graph, then greedily reveal a trajectory."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Teacher name."""
        ...

    @abstractmethod
    def score_nodes(
        self,
        graph: dgl.DGLGraph,
        classifier: torch.nn.Module,
        label: Optional[int] = None,
    ) -> torch.Tensor:
        """Full-graph node importance scores, shape (|V|,)."""
        ...

    def select_frontier(
        self,
        state: GraphState,
        scores: torch.Tensor,
        exclude: Optional[Set[int]] = None,
    ) -> Optional[int]:
        """Pick the highest-scored frontier node; exclude tried nodes."""
        best_node = None
        best_score = -float('inf')
        for u, v in state.frontier_edges:
            node = v if v not in state.V_revealed else u
            if exclude and node in exclude:
                continue
            s = float(scores[node])
            if s > best_score:
                best_score = s
                best_node = node
        return best_node

    def generate_trajectory(
        self,
        graph: dgl.DGLGraph,
        scores: torch.Tensor,
        seed_node: int,
        budget: int,
        edge_key_to_id: Optional[dict] = None,
    ) -> Tuple[Trajectory, GraphState]:
        """Greedy reveal from the seed using precomputed scores.

        Returns (trajectory, final_state); the state is returned for direct
        validation so the caller never rebuilds it."""
        state = GraphState(graph, seed_node, edge_key_to_id=edge_key_to_id)
        reveal_order: List[int] = []

        for _step in range(budget):
            if state.is_frontier_empty:
                break
            next_node = self.select_frontier(state, scores)
            if next_node is None:
                break
            reveal_order.append(next_node)
            reveal_node(state, next_node)

        n_revealed = len(state.V_revealed)
        reveal_ratio = n_revealed / max(state.n_nodes_total, 1)

        traj = Trajectory(
            graph_idx=-1,  # caller fills in
            seed_node=seed_node,
            reveal_ratio=reveal_ratio,
            reveal_order=reveal_order,
            final_pred=-1,     # caller fills in after classifier eval
            is_correct=True,   # filtered by caller
            teacher_type=self.name,
        )
        return traj, state
