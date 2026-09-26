"""Shared probability helper (single stable_probs implementation)."""

import torch
import torch.nn.functional as F


def stable_probs(scores: torch.Tensor, temperature: float = 1.0,
                 min_prob: float = 1e-3) -> torch.Tensor:
    """Floor probabilities after softmax to avoid exact underflow to 0
    (log(0) = -inf would NaN the loss).

    In normal training all probabilities exceed min_prob and the clamp is
    inert; it only catches the underflow edge.
    """
    probs = F.softmax(scores / temperature, dim=0)
    probs = probs.clamp(min=min_prob)
    return probs / probs.sum()
