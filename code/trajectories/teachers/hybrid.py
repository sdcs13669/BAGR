"""T4 hybrid teacher: alpha * gradient + (1 - alpha) * structural."""

import logging
from typing import Optional

import torch

from .base import Teacher
from .gradient import GradientTeacher
from .structural import StructuralTeacher

logger = logging.getLogger(__name__)


class HybridTeacher(Teacher):
    """T4 (hybrid): weighted combination of gradient attribution and
    structural importance.

    score = α · norm(grad_scores) + (1-α) · norm(struct_scores)
    """

    def __init__(self, alpha: float = 0.5, struct_method: str = 'degree'):
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got: {alpha}")
        self.alpha = alpha
        self._grad_teacher = GradientTeacher()
        self._struct_teacher = StructuralTeacher(method=struct_method)

    @property
    def name(self) -> str:
        return f'hybrid-a{self.alpha:.2f}'

    def score_nodes(
        self,
        graph,
        classifier,
        label: Optional[int] = None,
    ) -> torch.Tensor:
        grad_scores = self._grad_teacher.score_nodes(graph, classifier, label)
        struct_scores = self._struct_teacher.score_nodes(graph)

        grad_norm = self._normalize(grad_scores)
        struct_norm = self._normalize(struct_scores)

        return self.alpha * grad_norm + (1.0 - self.alpha) * struct_norm

    @staticmethod
    def _normalize(t: torch.Tensor) -> torch.Tensor:
        t_max = t.max()
        t_min = t.min()
        denom = t_max - t_min
        if denom > 0:
            return (t - t_min) / denom
        return torch.zeros_like(t) if t_max == 0 else torch.ones_like(t)
