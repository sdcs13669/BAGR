"""FrontierNodesEncoder: run the GNN on the extended subgraph to encode
frontier nodes.

Extended subgraph = V_revealed + frontier nodes, revealed-revealed edges
(both directions) and revealed->frontier edges (one-way). Node features come
from a trainable type embedding (revealed=0, frontier=1); edge features come
from the original graph (the key signal distinguishing frontier nodes).
Graph construction lives in FrontierGraphBuilder. Per-layer hidden_dims/
dropouts are supported (legacy scalars are expanded; state-dict keys
unchanged). node_type_emb dim = hidden_dims[0]; an empty frontier returns
width hidden_dims[-1]."""

from typing import List, Optional, Tuple

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from classifier.layers.gin import GINELayer
from .graph_builder import FrontierSubgraph


class FrontierNodesEncoder(nn.Module):
    """Run the GNN on the extended subgraph to encode frontier nodes.

    Parameters
    ----------
    edge_feat_dim : int  edge feature dim (PPA: 7)
    hidden_dim : int     hidden width (default 128, uniform form)
    n_layers : int       GNN layers (default 2)
    dropout : float      dropout (default 0.1)
    node_feat_dim : int  raw node feature dim (0 = type embedding only)
    hidden_dims/dropouts : per-layer lists (override the scalars)
    """

    def __init__(
        self,
        edge_feat_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
        node_feat_dim: int = 0,
        hidden_dims: Optional[List[int]] = None,
        dropouts: Optional[List[float]] = None,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [int(hidden_dim)] * int(n_layers)
        if dropouts is None:
            dropouts = [float(dropout)] * len(hidden_dims)
        if len(dropouts) != len(hidden_dims):
            raise ValueError("dropouts and hidden_dims length mismatch")

        self.hidden_dims = [int(d) for d in hidden_dims]
        self.dropouts = [float(p) for p in dropouts]
        self.n_layers = len(self.hidden_dims)
        self.hidden_dim = self.hidden_dims[-1]
        self.dropout = self.dropouts[0]

        self.node_type_emb = nn.Embedding(2, self.hidden_dims[0])
        self.node_feat_proj = (
            nn.Linear(node_feat_dim, self.hidden_dims[0]) if node_feat_dim > 0 else None
        )

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(self.n_layers):
            self.convs.append(GINELayer(self.hidden_dims[i], edge_feat_dim))
            self.bns.append(nn.BatchNorm1d(self.hidden_dims[i]))

    def _gnn_layers(self, g, h, edge_feat):
        """GNN layer loop (shared by forward / forward_batch / forward_all_batch)."""
        for i in range(self.n_layers):
            h = self.convs[i](g, h, edge_feat)
            h = self.bns[i](h)
            if i < self.n_layers - 1:
                h = F.dropout(F.relu(h), p=self.dropouts[i], training=self.training)
            else:
                h = F.dropout(h, p=self.dropouts[i], training=self.training)
        return h

    def forward(
        self, subgraph: FrontierSubgraph
    ) -> Tuple[torch.Tensor, List[int]]:
        """Encode frontier nodes; returns (frontier_embeds, frontier_nodes)."""
        g = subgraph.g
        n_revealed = subgraph.n_revealed
        frontier = subgraph.frontier_original_ids

        if len(frontier) == 0:
            return torch.empty(0, self.hidden_dim), []

        device = self.node_type_emb.weight.device
        g = subgraph.g.to(device)
        n_total = g.num_nodes()
        edge_feat = g.edata['feat']

        if 'feat' in g.ndata and self.node_feat_proj is not None:
            h = self.node_feat_proj(g.ndata['feat'].to(device))
            h[n_revealed:] = 0
        else:
            node_types = torch.zeros(n_total, dtype=torch.long, device=device)
            node_types[n_revealed:] = 1
            h = self.node_type_emb(node_types)
            h[n_revealed:] = 0

        h = self._gnn_layers(g, h, edge_feat)

        frontier_embeds = h[n_revealed:]
        return frontier_embeds, frontier

    def forward_batch(
        self, subgraphs: List[FrontierSubgraph],
    ) -> List[Tuple[torch.Tensor, List[int]]]:
        """Encode several extended subgraphs in one GNN forward; returns each
        subgraph's frontier embeddings.

        Merges many small graph launches into one to cut kernel overhead.
        """
        if not subgraphs:
            return []

        device = self.node_type_emb.weight.device
        batch_g = dgl.batch([sg.g.to(device) for sg in subgraphs])
        n_total = batch_g.num_nodes()

        node_counts = [sg.g.num_nodes() for sg in subgraphs]
        offsets = [0]
        for c in node_counts:
            offsets.append(offsets[-1] + c)

        frontier_starts: List[int] = []
        frontier_counts: List[int] = []

        if 'feat' in batch_g.ndata and self.node_feat_proj is not None:
            h = self.node_feat_proj(batch_g.ndata['feat'].to(device))
            for i, sg in enumerate(subgraphs):
                start = offsets[i]
                n_rev = sg.n_revealed
                n_fr = len(sg.frontier_original_ids)
                frontier_starts.append(start + n_rev)
                frontier_counts.append(n_fr)
                if n_fr > 0:
                    h[start + n_rev:start + n_rev + n_fr] = 0
        else:
            node_types = torch.zeros(n_total, dtype=torch.long, device=device)
            for i, sg in enumerate(subgraphs):
                start = offsets[i]
                n_rev = sg.n_revealed
                n_fr = len(sg.frontier_original_ids)
                frontier_starts.append(start + n_rev)
                frontier_counts.append(n_fr)
                if n_fr > 0:
                    node_types[start + n_rev:start + n_rev + n_fr] = 1

            h = self.node_type_emb(node_types)
            for start, cnt in zip(frontier_starts, frontier_counts):
                if cnt > 0:
                    h[start:start + cnt] = 0

        edge_feat = batch_g.edata.get('feat',
                      torch.empty(0, self.convs[0].edge_encoder.in_features,
                                  device=device))

        h = self._gnn_layers(batch_g, h, edge_feat)

        results: List[Tuple[torch.Tensor, List[int]]] = []
        for i, sg in enumerate(subgraphs):
            start = frontier_starts[i]
            cnt = frontier_counts[i]
            if cnt == 0:
                results.append((torch.empty(0, self.hidden_dim, device=device), []))
            else:
                results.append((h[start:start + cnt], sg.frontier_original_ids))

        return results

    def forward_all_batch(
        self, subgraphs: List[FrontierSubgraph],
    ) -> List[torch.Tensor]:
        """One GNN forward returning ALL node embeddings per subgraph
        (including revealed ones); used by the evidence reader.

        Identical feature construction and GNN as forward_batch: h[i] has
        shape (n_total_i, hidden_dim) with the first n_revealed rows the
        revealed nodes (ascending) and the rest frontier. Callers slice by row.
        """
        if not subgraphs:
            return []

        device = self.node_type_emb.weight.device
        batch_g = dgl.batch([sg.g.to(device) for sg in subgraphs])
        n_total = batch_g.num_nodes()

        node_counts = [sg.g.num_nodes() for sg in subgraphs]
        offsets = [0]
        for c in node_counts:
            offsets.append(offsets[-1] + c)

        if 'feat' in batch_g.ndata and self.node_feat_proj is not None:
            h = self.node_feat_proj(batch_g.ndata['feat'].to(device))
            for i, sg in enumerate(subgraphs):
                start = offsets[i]
                n_rev = sg.n_revealed
                n_fr = len(sg.frontier_original_ids)
                if n_fr > 0:
                    h[start + n_rev:start + n_rev + n_fr] = 0
        else:
            node_types = torch.zeros(n_total, dtype=torch.long, device=device)
            for i, sg in enumerate(subgraphs):
                start = offsets[i]
                n_rev = sg.n_revealed
                n_fr = len(sg.frontier_original_ids)
                if n_fr > 0:
                    node_types[start + n_rev:start + n_rev + n_fr] = 1
            h = self.node_type_emb(node_types)
            for i, sg in enumerate(subgraphs):
                start = offsets[i]
                n_rev = sg.n_revealed
                n_fr = len(sg.frontier_original_ids)
                if n_fr > 0:
                    h[start + n_rev:start + n_rev + n_fr] = 0

        edge_feat = batch_g.edata.get('feat',
                      torch.empty(0, self.convs[0].edge_encoder.in_features,
                                  device=device))

        h = self._gnn_layers(batch_g, h, edge_feat)

        return [h[offsets[i]:offsets[i + 1]] for i in range(len(subgraphs))]
