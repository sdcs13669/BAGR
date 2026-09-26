"""T2 info-gain teacher: simulate revealing every frontier candidate at each
step and pick the node with the largest entropy drop."""

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
import dgl

from active_graph_reader.engine.graph_state import GraphState
from active_graph_reader.engine.reveal_action import reveal_node
from .base import Teacher, Trajectory

logger = logging.getLogger(__name__)


class InfoGainTeacher(Teacher):
    """T2 (info-gain): pick the frontier node that most reduces classifier
    entropy.

    Each step simulates every frontier candidate (build subgraph, batched
    classifier inference) and maximizes IG(v) = H(current) - H(after_reveal_v);
    candidates with IG <= 0 are excluded.
    """

    def __init__(
        self,
        classifier,
        subgraph_builder,
        device: str = 'cpu',
    ):
        self._classifier = classifier
        self._subgraph_builder = subgraph_builder
        self._device = device

    @property
    def name(self) -> str:
        return 'info-gain'

    def score_nodes(
        self,
        graph,
        classifier: Optional[torch.nn.Module] = None,
        label: Optional[int] = None,
    ) -> torch.Tensor:
        """T2 uses no static scores."""
        return torch.zeros(graph.num_nodes())

    def generate_trajectory(
        self,
        graph,
        scores: torch.Tensor,  # ignored
        seed_node: int,
        budget: int,
        edge_key_to_id: Optional[dict] = None,
    ) -> Tuple[Trajectory, GraphState]:
        state = GraphState(graph, seed_node, edge_key_to_id=edge_key_to_id)
        reveal_order: List[int] = []

        for _step in range(budget):
            if state.is_frontier_empty:
                break

            frontier_nodes = state.get_frontier_nodes()
            if not frontier_nodes:
                break

            if len(state.V_revealed) == 1 and len(state.revealed_edge_keys) == 0:
                next_node = frontier_nodes[0]
                reveal_order.append(next_node)
                reveal_node(state, next_node)
                continue

            cur_g, _ = self._subgraph_builder.build(state)
            H_current = self._entropy(cur_g)

            candidate_gs = []
            for v in frontier_nodes:
                cg = self._build_candidate_subgraph(state, v)
                candidate_gs.append(cg)

            batched = dgl.batch(candidate_gs).to(self._device)
            node_ids = torch.zeros(batched.num_nodes(), dtype=torch.long,
                                   device=self._device)
            with torch.no_grad():
                logits = self._classifier(batched, node_ids)
            probs = F.softmax(logits, dim=-1)
            entropies = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)

            best_node = None
            best_ig = -float('inf')
            for i, v in enumerate(frontier_nodes):
                ig = H_current - float(entropies[i])
                if ig > best_ig:
                    best_ig = ig
                    best_node = v

            if best_node is None or best_ig <= 0:
                break

            reveal_order.append(best_node)
            reveal_node(state, best_node)

        n_revealed = len(state.V_revealed)
        traj = Trajectory(
            graph_idx=-1,
            seed_node=seed_node,
            reveal_ratio=n_revealed / max(state.n_nodes_total, 1),
            reveal_order=reveal_order,
            final_pred=-1,
            is_correct=True,
            teacher_type=self.name,
        )
        return traj, state

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _entropy(self, sub_g) -> float:
        sub_g = sub_g.to(self._device)
        node_ids = torch.zeros(sub_g.num_nodes(), dtype=torch.long,
                               device=self._device)
        with torch.no_grad():
            logits = self._classifier(sub_g, node_ids)
        probs = F.softmax(logits, dim=-1)
        return float(-(probs * torch.log(probs + 1e-8)).sum())

    def _build_candidate_subgraph(
        self, state: GraphState, v: int,
    ):
        """Build the simulated subgraph V_revealed + {v} (state untouched)."""
        nodes = sorted(set(state.V_revealed) | {v})
        n_nodes = len(nodes)

        edge_keys = set(state.revealed_edge_keys)
        for u, w in state.frontier_edges:
            if (u == v and w in state.V_revealed) or (w == v and u in state.V_revealed):
                edge_keys.add((min(u, w), max(u, w)))

        edge_keys = list(edge_keys)
        if not edge_keys:
            sub_g = dgl.graph(([], []), num_nodes=n_nodes)
            if state.has_node_feat and state.node_feat is not None:
                sub_g.ndata['feat'] = state.node_feat[nodes]
            sub_g = dgl.add_self_loop(sub_g)
            return sub_g

        edge_ids = [state._edge_key_to_id[k] for k in edge_keys]
        edge_feats = state.edge_feat[edge_ids]
        device = edge_feats.device

        edge_pairs = torch.tensor(edge_keys, dtype=torch.long, device=device)
        max_old = max(nodes)
        old_to_new = torch.full((max_old + 1,), -1, dtype=torch.long, device=device)
        old_to_new[nodes] = torch.arange(n_nodes, device=device)

        u_new = old_to_new[edge_pairs[:, 0]]
        v_new = old_to_new[edge_pairs[:, 1]]
        K = u_new.shape[0]
        src = torch.empty(2 * K, dtype=torch.long, device=device)
        dst = torch.empty(2 * K, dtype=torch.long, device=device)
        src[0::2], src[1::2] = u_new, v_new
        dst[0::2], dst[1::2] = v_new, u_new

        sub_g = dgl.graph((src, dst), num_nodes=n_nodes)
        sub_g.edata['feat'] = edge_feats.repeat_interleave(2, dim=0)
        if state.has_node_feat and state.node_feat is not None:
            sub_g.ndata['feat'] = state.node_feat[nodes]
        sub_g = dgl.add_self_loop(sub_g)
        return sub_g
