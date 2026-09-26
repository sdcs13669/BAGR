"""T1 gradient teacher: frozen-classifier backprop gradients as node importance."""

import logging
from typing import Optional

import torch

from .base import Teacher

logger = logging.getLogger(__name__)


class GradientTeacher(Teacher):
    """T1 (gradient): pick the frontier node that most reduces classifier
    loss.

    Backpropagates on the full graph, takes the gradient norm of each conv
    layer output w.r.t. node embeddings, and aggregates across layers into a
    per-node importance score."""

    def __init__(self):
        self._hooks = []

    @property
    def name(self) -> str:
        return 'gradient'

    def score_nodes(
        self,
        graph,
        classifier,
        label: Optional[int] = None,
    ) -> torch.Tensor:
        if label is None:
            raise ValueError("GradientTeacher needs a label to compute the correct-class gradient")

        device = next(classifier.parameters()).device
        graph = graph.to(device)

        activations = []

        def make_hook(act_list):
            def hook(module, _input, output):
                output.retain_grad()
                act_list.append(output)
            return hook

        encoder = classifier.encoder
        hooks = []
        for conv in encoder.convs:
            hooks.append(conv.register_forward_hook(make_hook(activations)))

        try:
            node_ids = torch.zeros(graph.num_nodes(), dtype=torch.long, device=device)
            logits = classifier(graph, node_ids)

            classifier.zero_grad()
            correct_logit = logits[0, label]
            correct_logit.backward()

            n_nodes = graph.num_nodes()
            node_scores = torch.zeros(n_nodes, device=device)
            for act in activations:
                grad = act.grad
                if grad is not None:
                    node_scores += grad.norm(dim=1)

            score_sum = node_scores.sum()
            if score_sum == 0:
                node_scores = torch.ones(n_nodes, device=device) / n_nodes
        finally:
            for h in hooks:
                h.remove()

        return node_scores.detach().cpu()
