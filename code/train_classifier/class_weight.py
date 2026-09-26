"""Class weights for classifier training (--class-weight).

Datasets such as COLLAB have a majority class above 50%; unweighted CE
collapses to predicting the majority class. mode='balanced' uses the sklearn
convention w_c = N / (K * count_c) to equalize total class weights.
mode='none' returns None (unweighted, default)."""

from typing import Optional

import torch


CLASS_WEIGHT_MODES = ('none', 'balanced')


def compute_class_weight(mode: str, labels: torch.Tensor, n_classes: int,
                         device=None) -> Optional[torch.Tensor]:
    """Per-class weights from training labels; mode='none' -> None.

    labels: (N,) integer class tensor. Classes with zero training samples
    get weight 1.0 with a warning (the weight never enters any loss term).
    """
    if mode not in CLASS_WEIGHT_MODES:
        raise ValueError(
            f"unknown class_weight mode '{mode}', choose: {CLASS_WEIGHT_MODES}")
    if mode == 'none':
        return None

    counts = torch.bincount(labels.long(), minlength=n_classes)
    weights = torch.ones(n_classes, dtype=torch.float32)
    n_total = int(counts.sum())
    for c in range(n_classes):
        if counts[c] > 0:
            weights[c] = n_total / (n_classes * counts[c])
        else:
            print(f"  WARNING: class {c} has 0 training samples; weight set to 1.0")
    print(f"  class_weight='balanced': counts={counts.tolist()} → "
          f"weights={[round(w, 4) for w in weights.tolist()]}")
    if device is not None:
        weights = weights.to(device)
    return weights
