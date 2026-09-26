"""Model (re)building: rebuild reader and classifier from checkpoint weights.

Reader structure always follows the checkpoint weight shapes (the embedded
config only fills non-structural hyperparameters); classifier loading is
delegated to classifier.factory.
"""

import os
import torch

from active_graph_reader import build_reader as _build_reader
from active_graph_reader.config import reader_config_from_checkpoint
from classifier import GraphClassifier, classifier_config_from_ckpt
from tools.ckpt_io import strip_classifier_keys


def load_classifier_frozen(ckpt_or_path, device, requires_grad: bool = False):
    """Load the frozen upper-bound classifier (eval mode).

    requires_grad=False: frozen (RL reward / reference head);
    requires_grad=True: keep the autograd graph (gradient teacher)."""
    ckpt = ckpt_or_path
    if isinstance(ckpt_or_path, (str, os.PathLike)):
        ckpt = torch.load(str(ckpt_or_path), map_location=device, weights_only=False)
    clf = GraphClassifier(config=classifier_config_from_ckpt(ckpt)).to(device)
    clf.load_state_dict(ckpt['model_state_dict'], strict=True)
    clf.eval()
    if not requires_grad:
        for p in clf.parameters():
            p.requires_grad = False
    return clf


def load_reader_from_checkpoint(ckpt_or_path, device, stochastic: bool = None,
                                temperature: float = None, strict: bool = True):
    """Rebuild a reader from a checkpoint (structure from weight shapes).

    stochastic/temperature: eval/RL overrides (None = keep config values)."""
    ckpt = ckpt_or_path
    if isinstance(ckpt_or_path, (str, os.PathLike)):
        ckpt = torch.load(str(ckpt_or_path), map_location='cpu', weights_only=False)
    sd = ckpt.get('reader_state_dict', ckpt.get('state_dict'))
    if sd is None:
        raise KeyError("reader_state_dict/state_dict not found in checkpoint")
    sd = strip_classifier_keys(dict(sd))
    cfg = reader_config_from_checkpoint(ckpt, sd)
    if stochastic is not None:
        cfg.stochastic = bool(stochastic)
    if temperature is not None:
        cfg.temperature = float(temperature)
    reader = _build_reader(cfg).to(device)
    reader.load_state_dict(sd, strict=strict)
    return reader


def bootstrap_reader(src_ckpt, device, stochastic: bool = True,
                     temperature: float = None):
    """Bootstrap reader: build by the source checkpoint structure and
    partially transfer weights.

    Returns: (reader, (n_transferred, n_padded, skipped))."""
    from tools.ckpt_io import transfer_weights
    sd = src_ckpt.get('reader_state_dict', src_ckpt.get('state_dict'))
    if sd is None:
        raise KeyError("reader_state_dict not found in bootstrap checkpoint")
    sd = strip_classifier_keys(dict(sd))
    cfg = reader_config_from_checkpoint(src_ckpt, sd)
    if stochastic:
        cfg.stochastic = True
    if temperature is not None:
        cfg.temperature = float(temperature)
    reader = _build_reader(cfg).to(device)
    stats = transfer_weights(sd, reader)
    return reader, stats
