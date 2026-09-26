"""Seed utilities: the deterministic seed formula (single implementation)
and global RNG seeding.

The seed formula (graph_idx*31 + seed_i*17 + 7) % n_nodes is part of the
protocol; all previously scattered copies converge here."""

import random

import numpy as np
import torch

SEED_METHOD_STR = 'deterministic formula: (idx*31 + seed_i*17 + 7) % n_nodes'


def deterministic_seed(graph_idx: int, seed_i: int, n_nodes: int) -> int:
    """Protocol deterministic seed-node selection."""
    return (graph_idx * 31 + seed_i * 17 + 7) % n_nodes


def set_all_seeds(seed: int) -> None:
    """Seed python / numpy / torch / cuda RNGs in one call."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
