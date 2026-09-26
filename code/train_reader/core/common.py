"""Shared training utilities: rewards, advantage normalization, freezing, teleport."""

import math
import random
from typing import List

import torch
import torch.nn.functional as F

from active_graph_reader.engine.reveal_action import teleport_if_stuck


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def compute_reward(logits_i: torch.Tensor, label: torch.Tensor,
                   reward_type: str, ln_k: float,
                   gate_logits: torch.Tensor = None) -> float:
    """Per-trajectory reward: binary / tanh / asymmetric.

    With gate_logits (reward_source='both'), a trajectory counts as correct
    only if both primary and gate argmax are correct; tanh clamps to
    non-positive on miss. With gate_logits=None the behavior is unchanged.
    """
    correct = logits_i.argmax(dim=-1).item() == label.item()
    if gate_logits is not None:
        correct = correct and (gate_logits.argmax(dim=-1).item() == label.item())
    if reward_type == 'tanh':
        ce = F.cross_entropy(logits_i, label).item()
        r = math.tanh(1.0 - ce / ln_k)
        if gate_logits is not None and not correct:
            r = min(r, 0.0)
        return r
    if reward_type == 'asymmetric':
        return 1.0 if correct else -0.1
    return float(correct)  # binary


def compute_soft_both_reward(clf_logits: torch.Tensor, reader_logits: torch.Tensor,
                             label: torch.Tensor) -> float:
    """Soft-both reward: probability product instead of a hard joint test.

    R = p_clf(y_true) * p_reader(y_true) in [0, 1]; the reward decays
    proportionally when either model assigns low probability to the true
    class (partial credit, unlike hard both). --reward transform does not
    apply to this source.
    """
    y = label.item()
    p_clf = torch.softmax(clf_logits.float().flatten(), dim=-1)[y]
    p_reader = torch.softmax(reader_logits.float().flatten(), dim=-1)[y]
    return float(p_clf * p_reader)


def normalize_advantages(rewards, neg_scale: float) -> List[float]:
    """Batch advantage normalization with negative-side down-weighting.

    A = (R - mean)/(std + eps); negative A is scaled by neg_scale.
    Centering fixes the tanh anchor problem (near-all-negative rewards);
    neg_scale suppresses negative-sample noise.
    """
    if len(rewards) < 2:
        return list(rewards)
    rs = torch.tensor(rewards, dtype=torch.float)
    mean = rs.mean()
    std = rs.std()
    A = (rs - mean) / (std + 1e-6)
    A = torch.where(A >= 0, A, A * neg_scale)
    return A.tolist()


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def freeze_shared_layers(reader, enabled: bool) -> None:
    """--freeze: freeze everything except the feature projection layers."""
    if not enabled:
        return
    frozen = 0
    for name, p in reader.named_parameters():
        if 'edge_encoder' not in name and 'node_feat_proj' not in name:
            p.requires_grad = False
            frozen += 1
    print(f"  --freeze: {frozen} tensors frozen (only feature projections trainable)")


def teleport_or_deactivate(state, active: list, j: int, rng: random.Random = None) -> None:
    """Teleport handling when the frontier is empty but budget remains
    (seed in an isolated small component).

    Marks the state inactive if the budget is exhausted after teleporting.
    """
    st = state
    if teleport_if_stuck(st, rng=rng) is None:
        return
    if st.get_budget_used() >= st._budget_limit:
        active[j] = False
