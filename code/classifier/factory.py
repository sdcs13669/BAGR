"""Classifier factory: build from config, rebuild from checkpoints.

Rebuilds always follow the checkpoint embedded config; strict=True state
loading doubles as the config/weight consistency check and fails loudly on
mismatch (no silent correction)."""

import logging
from typing import Union

import torch

from .config import ClassifierConfig
from .graph_classifier import GraphClassifier

logger = logging.getLogger(__name__)


def classifier_config_from_ckpt(ckpt: dict) -> ClassifierConfig:
    """Checkpoint embedded config -> ClassifierConfig.

    Supports both {classifier: {...}, ...} and a bare legacy config dict.
    Raises if no embedded config is present.
    """
    embedded = ckpt.get('config')
    if isinstance(embedded, dict) and isinstance(embedded.get('classifier'), dict):
        embedded = embedded['classifier']
    if not isinstance(embedded, dict) or not embedded:
        raise ValueError(
            "checkpoint has no embedded classifier config; "
            "for legacy checkpoints run "
            "python code/tools/migrate_legacy_classifier_ckpt.py <ckpt.pt> --write")
    return ClassifierConfig.from_dict(embedded)


def build_classifier(config: ClassifierConfig) -> GraphClassifier:
    """ClassifierConfig -> untrained model."""
    return GraphClassifier(config=config)


def load_classifier(
    ckpt_path: str,
    device: Union[str, torch.device] = 'cpu',
    freeze: bool = True,
    strict: bool = True,
) -> GraphClassifier:
    """Load a classifier from a checkpoint.

    freeze=True (default): eval mode + no grad, the frozen reward/reference
    classifier; freeze=False for continued training.

    A shape mismatch raises RuntimeError by design (the config is the single
    source of truth; no silent correction).
    """
    device = torch.device(device)
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    sd = ckpt.get('model_state_dict', ckpt.get('state_dict'))
    if sd is None:
        raise KeyError(f"model_state_dict/state_dict not found in {ckpt_path}")

    model = GraphClassifier(config=classifier_config_from_ckpt(ckpt)).to(device)
    try:
        model.load_state_dict(sd, strict=strict)
    except RuntimeError as e:
        raise RuntimeError(
            f"{ckpt_path}: weights do not match the embedded config ({e}). "
            f"The config is authoritative; retrain or rebuild the config for "
            f"legacy or hand-edited checkpoints") from e
    if freeze:
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
    logger.info("classifier loaded from %s (val_acc=%s, freeze=%s)",
                ckpt_path, ckpt.get('val_acc', float('nan')), freeze)
    return model


def load_fclass(ckpt_path: str, device: Union[str, torch.device] = 'cpu',
                **_ignored) -> GraphClassifier:
    return load_classifier(ckpt_path, device=device, freeze=True)
