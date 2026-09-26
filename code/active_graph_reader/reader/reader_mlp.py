"""ReaderMLP: map frontier embeddings to scalar scores.

Supports per-layer hidden/dropout lists; the scalar form (hidden_dim + 3
layers) keeps legacy state-dict key names (mlp.0 / mlp.3 / mlp.6)."""

from typing import List, Optional

import torch.nn as nn


class ReaderMLP(nn.Module):
    """MLP scorer over frontier embeddings.


    Two forms: legacy scalars (hidden_dim + dropout, 3 uniform hidden
    layers) or per-layer lists hidden_dims[i] / dropouts[i].
    Structure: [Linear(hi) ReLU Dropout]*k -> Linear(->1).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1,
                 hidden_dims: Optional[List[int]] = None,
                 dropouts: Optional[List[float]] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [hidden_dim] * 3
        if dropouts is None:
            dropouts = [dropout] * len(hidden_dims)
        if len(dropouts) != len(hidden_dims):
            raise ValueError("dropouts and hidden_dims length mismatch")

        layers: list = []
        prev = input_dim
        for i, (h, p) in enumerate(zip(hidden_dims, dropouts)):
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(p=p)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)
        self.hidden_dims = list(hidden_dims)

    def forward(self, frontier_embeds):
        """(N, input_dim) -> (N,)"""
        return self.mlp(frontier_embeds).squeeze(-1)
