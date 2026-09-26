"""Checkpoint IO: uniform per-epoch saving, training-state resume, legacy
checkpoint compatibility.

Uniform checkpoint format: config (the effective config, embedded so
resume/eval can rebuild the model) and training_log (list of per-epoch
metrics). Legacy handling: checkpoints without a config get their reader
structure inferred from weights; v0/v1 reader_mlp input widths are zero-padded;
belief-classifier keys from early experiments are stripped."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

LEGACY_MLP_INPUT_DIMS = (128, 129)
CURRENT_MLP_INPUT_EXTRA = 4          # budget(1) + struct(3)


def strip_classifier_keys(state_dict: dict) -> dict:
    """Strip belief_state.classifier.* keys from early checkpoints."""
    return {k: v for k, v in state_dict.items()
            if not k.startswith('belief_state.classifier.')}


def pad_reader_mlp_input(state_dict: dict, target_dim: int = 132) -> Tuple[dict, bool]:
    """Zero-pad reader_mlp.mlp.0.weight input dim to target_dim (old ckpts).


    Missing columns are zero-initialized. Returns (state_dict, was_patched).
    """
    was_patched = False
    key = 'reader_mlp.mlp.0.weight'
    if key in state_dict:
        w = state_dict[key]
        if w.dim() == 2 and w.shape[1] < target_dim:
            gap = target_dim - w.shape[1]
            pad = torch.zeros(w.shape[0], gap, dtype=w.dtype, device=w.device)
            state_dict[key] = torch.cat([w, pad], dim=1)
            was_patched = True
    return state_dict, was_patched


def transfer_weights(source_state_dict: dict,
                     target_reader: nn.Module) -> Tuple[int, int, List[str]]:
    """Generic partial weight transfer.

    Same key+shape: copy; only a narrower last dim: zero-pad then copy;
    anything else keeps its random initialization.
    Returns: (n_transferred, n_padded, skipped_keys)
    """
    target_sd = target_reader.state_dict()
    n_transferred = 0
    n_padded = 0
    skipped: List[str] = []

    for k in target_sd:
        if k not in source_state_dict:
            skipped.append(k)
            continue
        src_w = source_state_dict[k]
        tgt_w = target_sd[k]
        if src_w.shape == tgt_w.shape:
            target_sd[k] = src_w.clone()
            n_transferred += 1
        elif (src_w.dim() >= 1
              and src_w.dim() == tgt_w.dim()
              and src_w.shape[:-1] == tgt_w.shape[:-1]
              and tgt_w.shape[-1] > src_w.shape[-1]):
            gap = tgt_w.shape[-1] - src_w.shape[-1]
            pad = torch.zeros(src_w.shape[:-1] + (gap,),
                              dtype=src_w.dtype, device=src_w.device)
            target_sd[k] = torch.cat([src_w, pad], dim=-1)
            n_padded += 1
        else:
            skipped.append(k)

    target_reader.load_state_dict(target_sd, strict=True)
    return n_transferred, n_padded, skipped


def save_epoch_checkpoint(
    path,
    epoch: int,
    reader,
    optimizer: Optional[torch.optim.Optimizer],
    config: Dict,
    training_log: List[Dict],
) -> None:
    """Uniform per-epoch checkpoint save."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        'epoch': epoch,
        'reader_state_dict': reader.state_dict(),
        'config': config,
        'training_log': training_log,
    }
    if optimizer is not None:
        ckpt['optimizer_state_dict'] = optimizer.state_dict()
    torch.save(ckpt, str(path))


def load_training_state(ckpt: dict) -> Tuple[int, List[Dict]]:
    """Restore (start_epoch, training_log) from a checkpoint.

    Tolerates old checkpoints that stored the hparam dict in training_log.
    """
    start_epoch = int(ckpt.get('epoch', -1)) + 1
    tl = ckpt.get('training_log', [])
    if isinstance(tl, dict):
        tl = []
    return start_epoch, list(tl)


def load_optimizer(optimizer: Optional[torch.optim.Optimizer],
                   ckpt: dict, device: str = 'cpu') -> bool:
    """Best-effort optimizer state restore; keep fresh init on failure."""
    sd = ckpt.get('optimizer_state_dict')
    if optimizer is None or not sd:
        return False
    try:
        optimizer.load_state_dict(sd)
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning("optimizer state restore failed (using fresh init): %s", e)
        return False
