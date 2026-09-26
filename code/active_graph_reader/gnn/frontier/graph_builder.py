"""FrontierGraphBuilder: build the extended subgraph from a GraphState.

Extended subgraph = V_revealed + frontier nodes, revealed-revealed edges
(both directions) and revealed->frontier edges (one-way). Frontier nodes are
unrevealed, so their identity must not leak back into revealed node
representations."""

from dataclasses import dataclass
from typing import List, Tuple

import dgl
import torch

from ...engine.graph_state import GraphState


@dataclass
class FrontierSubgraph:
    """Extended subgraph: the encoder input.

    Attributes
    ----------
    g : dgl.DGLGraph
    g: the built subgraph with edata[feat]; the first n_revealed nodes are
    revealed, the rest frontier.
    n_revealed : int
    frontier_original_ids : List[int]
    """
    g: dgl.DGLGraph
    n_revealed: int
    frontier_original_ids: List[int]


class FrontierGraphBuilder:
    """Convert a GraphState into the extended subgraph (encoder input).

    Contents:
    - nodes: V_revealed + frontier (ascending)
    - edges: revealed<->revealed (both directions) + revealed->frontier
    - edge features copied from the original graph into g.edata[feat]
    - no node features (the encoder assigns trainable type embeddings)
    """

    def build(self, state: GraphState) -> FrontierSubgraph:
        """Build one extended subgraph from a GraphState."""
        revealed = state.get_revealed_sorted()
        frontier = state.get_frontier_nodes()

        n_revealed = len(revealed)
        n_total = n_revealed + len(frontier)

        if n_total == 0:
            g = dgl.graph(([], []), num_nodes=0)
            return FrontierSubgraph(g=g, n_revealed=0, frontier_original_ids=[])

        device = (state.edge_feat.device if state.edge_feat is not None
                  else state.node_feat.device)

        max_old = revealed[-1]
        if frontier:
            max_old = max(max_old, frontier[-1])
        old_to_new = torch.full((max_old + 1,), -1, dtype=torch.long, device=device)
        old_to_new[revealed] = torch.arange(n_revealed, device=device)
        if frontier:
            old_to_new[frontier] = torch.arange(n_revealed, n_total, device=device)

        src_parts = []
        dst_parts = []
        feat_parts: List[torch.Tensor] = []

        if state.revealed_edge_keys:
            rk_list = list(state.revealed_edge_keys)
            rk_t = torch.tensor(rk_list, dtype=torch.long, device=device)
            edge_ids_r = [state._edge_key_to_id[k] for k in rk_list]
            feat_r = state.edge_feat[edge_ids_r]

            u_r = old_to_new[rk_t[:, 0]]
            v_r = old_to_new[rk_t[:, 1]]

            src_parts.append(u_r)
            src_parts.append(v_r)  # reverse
            dst_parts.append(v_r)
            dst_parts.append(u_r)  # reverse
            feat_parts.append(feat_r)
            feat_parts.append(feat_r)  # repeat for reverse

        if state.frontier_edges:
            fr_keys = list(state.frontier_edges)
            fr_ids = [state._edge_key_to_id[(min(u, v), max(u, v))] for u, v in fr_keys]
            feat_f = state.edge_feat[fr_ids]

            fr_t = torch.tensor(fr_keys, dtype=torch.long, device=device)
            u_f = old_to_new[fr_t[:, 0]]  # revealed side
            v_f = old_to_new[fr_t[:, 1]]  # frontier side

            src_parts.append(u_f)
            dst_parts.append(v_f)
            feat_parts.append(feat_f)

        if len(src_parts) == 0:
            g = dgl.graph(([], []), num_nodes=n_total)
            edge_feat = torch.empty(0, state.edge_feat_dim, device=device)
        else:
            src = torch.cat(src_parts, dim=0)
            dst = torch.cat(dst_parts, dim=0)
            edge_feat = torch.cat(feat_parts, dim=0)
            g = dgl.graph((src, dst), num_nodes=n_total)

        g.edata['feat'] = edge_feat
        g = dgl.add_self_loop(g)

        if state.has_node_feat and state.node_feat is not None:
            feat_dim = state.node_feat_dim
            feat_revealed = state.node_feat[revealed]
            feat_frontier = torch.zeros(len(frontier), feat_dim, device=device)
            g.ndata['feat'] = torch.cat([feat_revealed, feat_frontier], dim=0)
            g.ndata['has_orig_feat'] = torch.cat([
                torch.ones(n_revealed, dtype=torch.bool, device=device),
                torch.zeros(len(frontier), dtype=torch.bool, device=device),
            ])

        return FrontierSubgraph(g=g, n_revealed=n_revealed, frontier_original_ids=frontier)

    def build_batch(self, states: List[GraphState]) -> List[FrontierSubgraph]:
        """Build extended subgraphs in batch."""
        return [self.build(state) for state in states]
