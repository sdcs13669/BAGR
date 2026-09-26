"""Final subgraph builder: GraphState -> DGLGraph with revealed nodes/edges
only (classifier input).

Unlike the extended subgraph inside FrontierNodesEncoder (which also includes
frontier nodes/edges), the final subgraph contains only revealed content, with
edges in both directions to support message passing."""

import logging
from typing import Dict, List, Tuple
from ..engine.graph_state import GraphState
import dgl
import torch

logger = logging.getLogger(__name__)


class SubgraphBuilder:
    """Convert a finished GraphState into a classifier-consumable DGLGraph.

    Nodes: V_revealed only. Edges: only edges with both endpoints revealed
    (both directions). Edge features copied from the original graph; node
    features copied for the revealed range when the graph has them.
    """

    def build(
        self,
        state: GraphState,
    ) -> Tuple[dgl.DGLGraph, List[int]]:
        """Build the final subgraph (classifier input).

        original_ids: subgraph id -> original node id (ascending)."""
        original_ids = sorted(state.V_revealed)
        n_nodes = len(original_ids)
        if n_nodes == 0:
            raise ValueError("V_revealed is empty; cannot build a subgraph")

        if state.has_node_feat and state.node_feat is not None:
            node_feat_t = state.node_feat[original_ids]
        else:
            node_feat_t = None

        n_edges = len(state.revealed_edge_keys)
        if n_edges == 0:
            sub_g = dgl.graph(([], []), num_nodes=n_nodes)
            if node_feat_t is not None:
                sub_g.ndata['feat'] = node_feat_t
            sub_g = dgl.add_self_loop(sub_g)
            return sub_g, original_ids

        edge_keys = list(state.revealed_edge_keys)
        edge_ids = [state._edge_key_to_id[k] for k in edge_keys]
        edge_feats_t = state.edge_feat[edge_ids]

        device = edge_feats_t.device
        edge_pairs = torch.tensor(edge_keys, dtype=torch.long, device=device)

        max_old_id = max(original_ids)
        old_to_new = torch.full((max_old_id + 1,), -1, dtype=torch.long, device=device)
        old_to_new[original_ids] = torch.arange(n_nodes, device=device)

        u_new = old_to_new[edge_pairs[:, 0]]
        v_new = old_to_new[edge_pairs[:, 1]]

        K = u_new.shape[0]
        src = torch.empty(2 * K, dtype=torch.long, device=device)
        dst = torch.empty(2 * K, dtype=torch.long, device=device)
        src[0::2] = u_new
        src[1::2] = v_new
        dst[0::2] = v_new
        dst[1::2] = u_new
        edge_feats_bidir = edge_feats_t.repeat_interleave(2, dim=0)

        sub_g = dgl.graph((src, dst), num_nodes=n_nodes)
        sub_g.edata['feat'] = edge_feats_bidir
        if node_feat_t is not None:
            sub_g.ndata['feat'] = node_feat_t

        sub_g = dgl.add_self_loop(sub_g)
        return sub_g, original_ids

    def build_batch(
        self,
        states: List[GraphState],
    ) -> Tuple[dgl.DGLGraph, List[List[int]]]:
        """Build final subgraphs in batch and dgl.batch them."""
        sub_graphs = []
        all_original_ids = []
        for state in states:
            sub_g, original_ids = self.build(state)
            sub_graphs.append(sub_g)
            all_original_ids.append(original_ids)
        batched_g = dgl.batch(sub_graphs)
        return batched_g, all_original_ids
