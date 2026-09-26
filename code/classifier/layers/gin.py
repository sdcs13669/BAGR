"""GIN backbone (OGB-style) with the GNN+ per-layer pipeline (arXiv 2502.09263,
Eq.11).

Per-layer pipeline: MP (with edge integration) -> inter-layer norm ->
activation -> dropout -> residual -> FFN epilogue; positional encoding
injects at the input; readout after the last layer. Blocks are toggled by
ClassifierConfig fields; defaults are bit-identical to legacy checkpoints
(values and state-dict keys). Per-layer hidden_dims/dropouts lists expand
from legacy scalars with unchanged key names. virtual_node / residual require
uniform width across layers (dimension alignment)."""

import logging

import dgl
import dgl.nn as dglnn
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from ..plus.edge import EDGE_ENCODER_KINDS, make_edge_encoder
from ..plus.ffn import FFN
from ..plus.norm import make_norm
from ..plus.pos_enc import random_walk_structural_encoding
from ..plus.readout import make_readout

logger = logging.getLogger(__name__)


class GINELayer(nn.Module):
    """Single GIN layer with edge features (per-layer edge encoder).

    Matches OGB's GINConv:
      edge_emb      = edge_encoder(edge_feat)      <- plus/edge.py block
      message       = ReLU(src_h + edge_emb)       <- edge integration (Eq.6)
      aggregation   = sum
      h_new         = MLP((1+eps) * h + aggregated_messages)
    """

    def __init__(self, hidden_dim: int, edge_feat_dim: int, mlp_hidden_mult: int = 2,
                 edge_encoder_kind: str = 'linear'):
        super().__init__()
        bottleneck_dim = hidden_dim * mlp_hidden_mult
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, hidden_dim),
        )
        self.eps = nn.Parameter(torch.zeros(1))
        self.edge_encoder = make_edge_encoder(edge_feat_dim, hidden_dim,
                                              edge_encoder_kind)

    def forward(
        self,
        g: dgl.DGLGraph,
        h: torch.Tensor,
        edge_feat: torch.Tensor,
    ) -> torch.Tensor:
        with g.local_scope():
            g.ndata['h'] = h
            edge_emb = self.edge_encoder(edge_feat)
            g.edata['e'] = edge_emb

            g.apply_edges(
                lambda edges: {'m': F.relu(edges.src['h'] + edges.data['e'])}
            )
            g.update_all(fn.copy_e('m', 'm'), fn.sum('m', 'neigh'))

            h_new = (1.0 + self.eps) * h + g.ndata['neigh']
            return self.mlp(h_new)


class GINEEncoder(nn.Module):
    """GIN encoder with the GNN+ per-layer pipeline skeleton.

    Two argument forms (lists take precedence):
    - legacy scalars: hidden_dim + n_layers + dropout (uniform)
    - lists: hidden_dims / dropouts (per-layer)

    plus/ block kwargs (appended, all defaulting to original behavior):
      readout / edge_encoder_kind / ffn / ffn_dropout / pos_enc / pos_enc_ksteps

    State-dict key names match legacy checkpoints (no new keys by default).
    """

    def __init__(
        self,
        edge_feat_dim: int,
        hidden_dim: int = 300,
        n_layers: int = 5,
        dropout: float = 0.5,
        virtual_node: bool = False,
        residual: bool = False,
        node_feat_dim: int = 0,
        hidden_dims: Optional[List[int]] = None,
        dropouts: Optional[List[float]] = None,
        gnn_norm: Optional[str] = None,
        readout: str = 'mean',
        edge_encoder_kind: str = 'linear',
        ffn: bool = False,
        ffn_dropout: float = 0.0,
        pos_enc: str = 'none',
        pos_enc_ksteps: int = 8,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [int(hidden_dim)] * int(n_layers)
        if dropouts is None:
            dropouts = [float(dropout)] * len(hidden_dims)
        if len(dropouts) != len(hidden_dims):
            raise ValueError(
                f"dropouts length {len(dropouts)} != hidden_dims length {len(hidden_dims)}")
        if (virtual_node or residual) and len(set(hidden_dims)) > 1:
            raise ValueError("virtual_node / residual require uniform layer width")
        if edge_encoder_kind not in EDGE_ENCODER_KINDS:
            raise ValueError(
                f"unknown edge encoder '{edge_encoder_kind}', choose: {EDGE_ENCODER_KINDS}")
        if pos_enc not in ('none', 'rwse'):
            raise ValueError(f"unknown pos_enc '{pos_enc}', choose: none, rwse")

        self.hidden_dims = [int(d) for d in hidden_dims]
        self.dropouts = [float(p) for p in dropouts]
        self.n_layers = len(self.hidden_dims)
        self.hidden_dim = self.hidden_dims[0]
        self.dropout = self.dropouts[0]
        self.virtual_node = virtual_node
        self.residual = residual
        self.norm_type = gnn_norm if gnn_norm is not None else 'batchnorm'

        self.pos_enc = pos_enc
        self.pos_enc_ksteps = int(pos_enc_ksteps)
        self.node_feat_dim = int(node_feat_dim)
        proj_in = self.node_feat_dim + (self.pos_enc_ksteps
                                        if pos_enc == 'rwse' else 0)
        if pos_enc == 'rwse' or self.node_feat_dim > 0:
            self.node_proj = nn.Linear(proj_in, self.hidden_dims[0])
        else:
            self.node_emb = nn.Embedding(1, self.hidden_dims[0])

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(self.n_layers):
            self.convs.append(GINELayer(self.hidden_dims[i], edge_feat_dim,
                                        edge_encoder_kind=edge_encoder_kind))
            self.bns.append(make_norm(self.hidden_dims[i], self.norm_type))

        if ffn:
            self.ffns = nn.ModuleList([FFN(d, ffn_dropout)
                                       for d in self.hidden_dims])
        else:
            self.ffns = nn.ModuleList([nn.Identity()
                                       for _ in range(self.n_layers)])

        # Virtual Node (optional, matches OGB gin-virtual)
        if virtual_node:
            self.vn_embedding = nn.Embedding(1, self.hidden_dims[0])
            nn.init.constant_(self.vn_embedding.weight.data, 0)
            self.vn_mlps = nn.ModuleList()
            for _ in range(self.n_layers - 1):
                h = self.hidden_dims[0]
                self.vn_mlps.append(nn.Sequential(
                    nn.Linear(h, 2 * h),
                    nn.BatchNorm1d(2 * h),
                    nn.ReLU(),
                    nn.Linear(2 * h, h),
                    nn.BatchNorm1d(h),
                    nn.ReLU(),
                ))

        self.pools = nn.ModuleList(make_readout(readout)[0])
        self.out_dim = self.hidden_dims[-1] * make_readout(readout)[1]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        g: dgl.DGLGraph,
        node_ids: torch.Tensor,
    ) -> torch.Tensor:
        if self.pos_enc == 'rwse':
            pe = random_walk_structural_encoding(
                g, self.pos_enc_ksteps).to(node_ids.device)
            if self.node_feat_dim > 0:
                x = torch.cat([g.ndata['feat'].to(node_ids.device), pe], dim=-1)
            else:
                x = torch.cat([torch.ones(pe.shape[0], 1,
                                          device=node_ids.device), pe], dim=-1)
            h = self.node_proj(x)
        elif self.node_feat_dim > 0:
            h = self.node_proj(g.ndata['feat'].to(node_ids.device))
        else:
            h = self.node_emb(node_ids)

        edge_feat = g.edata.get('feat', None)
        if edge_feat is None:
            edge_feat_dim = self.convs[0].edge_encoder.in_features \
                if hasattr(self.convs[0].edge_encoder, 'in_features') \
                else self.convs[0].edge_encoder[0].in_features
            edge_feat = torch.zeros(g.num_edges(), edge_feat_dim,
                                    device=h.device)

        # Virtual Node init
        if self.virtual_node:
            num_graphs = g.batch_size
            vn_h = self.vn_embedding(
                torch.zeros(num_graphs, dtype=torch.long, device=h.device)
            )
            batch_num_nodes = g.batch_num_nodes()
            node_to_graph = torch.arange(
                num_graphs, device=h.device
            ).repeat_interleave(batch_num_nodes)

        h_list = [h]
        for i in range(self.n_layers):
            # Add Virtual Node embedding to node features
            if self.virtual_node:
                h_list[i] = h_list[i] + vn_h[node_to_graph]

            h = self.convs[i](g, h_list[i], edge_feat)
            h = self.bns[i](h)
            h = h if i == self.n_layers - 1 else F.relu(h)
            h = F.dropout(h, p=self.dropouts[i],            # ③ Dropout
                          training=self.training)
            if self.residual:
                h = h + h_list[i]
            h = self.ffns[i](h)

            h_list.append(h)

            # Update Virtual Node
            if self.virtual_node and i < self.n_layers - 1:
                g.ndata['_vn_tmp'] = h
                vn_agg = dgl.sum_nodes(g, '_vn_tmp')
                vn_h = vn_h + F.dropout(
                    self.vn_mlps[i](vn_agg + vn_h),
                    p=self.dropouts[i], training=self.training,
                )

        pooled = [pool(g, h) for pool in self.pools]
        return torch.cat(pooled, dim=-1) if len(pooled) > 1 else pooled[0]
