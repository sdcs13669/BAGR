"""GNN+ block 5: feed-forward network.

Faithful to Eq.10: FFN(h) = BN(sigma(h W1) W2 + h) - two transformations,
built-in residual, trailing BatchNorm; dropout applies after the hidden
activation (before W2). Inserted as the last stage of the per-layer pipeline.
When disabled the encoder uses nn.Identity (no params, no keys)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import make_norm

FFN_HIDDEN_MULT = 2


class FFN(nn.Module):
    """h → BN( σ(h·W₁)·W₂ + h )，W₁: dim→mult·dim，W₂: mult·dim→dim。"""

    def __init__(self, dim: int, dropout: float = 0.0,
                 hidden_mult: int = FFN_HIDDEN_MULT):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_mult * dim)
        self.fc2 = nn.Linear(hidden_mult * dim, dim)
        self.bn = make_norm(dim, 'batchnorm')
        self.dropout = float(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h2 = self.fc2(F.dropout(F.relu(self.fc1(h)),
                                p=self.dropout, training=self.training))
        return self.bn(h2 + h)
