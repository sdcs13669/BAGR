"""GNN+ block 2: inter-layer norm factory (distinct from the BatchNorm inside
the GINELayer MLP).

'none' returns nn.Identity - no params, no state-dict keys - so default
configs keep legacy checkpoint key sets unchanged."""

import torch.nn as nn

NORM_TYPES = ('batchnorm', 'layernorm', 'none')


def make_norm(dim: int, norm_type: str) -> nn.Module:
    if norm_type == 'batchnorm':
        return nn.BatchNorm1d(dim)
    if norm_type == 'layernorm':
        return nn.LayerNorm(dim)
    if norm_type == 'none':
        return nn.Identity()
    raise ValueError(f"unknown norm type '{norm_type}', choose: {NORM_TYPES}")
