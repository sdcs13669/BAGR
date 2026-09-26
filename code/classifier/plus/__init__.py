"""GNN+ block package: the six components of the ICML 2025 unified
enhancement framework (arXiv 2502.09263, "Can Classic GNNs Be Strong
Baselines for Graph-level Tasks?") plus readout utilities.

Per-layer pipeline (Eq.11): MP (with edge integration) -> norm -> activation
-> dropout -> residual -> FFN (epilogue). Blocks are toggled via
ClassifierConfig fields and all default to off (= original backbone behavior,
identical state-dict keys).

Component <-> module map:
  edge integration  plus/edge.py    make_edge_encoder  -> config.edge_encoder
  norm              plus/norm.py    make_norm          -> config.gnn_norm
  dropout           (encoder loop stage)               -> config.dropouts
  residual          (encoder loop stage)               -> config.residual
  FFN               plus/ffn.py     FFN                -> config.ffn / ffn_dropout
  pos encoding      plus/pos_enc.py RWSE               -> config.pos_enc / pos_enc_ksteps
  readout           plus/readout.py make_readout       -> config.readout

This package may only import torch/dgl (never config/layers), keeping the
dependency direction plus <- layers <- graph_classifier <- factory."""

from .edge import EDGE_ENCODER_KINDS, make_edge_encoder
from .ffn import FFN, FFN_HIDDEN_MULT
from .norm import NORM_TYPES, make_norm
from .pos_enc import random_walk_structural_encoding
from .readout import READOUT_KINDS, READOUT_OUT_MULT, make_readout

__all__ = [
    'EDGE_ENCODER_KINDS', 'make_edge_encoder',
    'FFN', 'FFN_HIDDEN_MULT',
    'NORM_TYPES', 'make_norm',
    'random_walk_structural_encoding',
    'READOUT_KINDS', 'READOUT_OUT_MULT', 'make_readout',
]
