"""GNN+ block 1: edge feature integration.

Per arXiv 2502.09263 Eq.6, the projected edge vector is added inside the
message aggregation (GINELayer h + edge_emb). The edge encoder is a
replaceable block: linear (single Linear, default, state-dict keys identical
to legacy checkpoints) or mlp (two-layer bottleneck, new keys only with the
new config)."""

import torch.nn as nn

EDGE_ENCODER_KINDS = ('linear', 'mlp')


def make_edge_encoder(edge_feat_dim: int, hidden_dim: int, kind: str = 'linear') -> nn.Module:
    if kind == 'linear':
        return nn.Linear(edge_feat_dim, hidden_dim)
    if kind == 'mlp':
        return nn.Sequential(
            nn.Linear(edge_feat_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
    raise ValueError(f"unknown edge encoder '{kind}', choose: {EDGE_ENCODER_KINDS}")
