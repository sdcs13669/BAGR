"""Standalone classifier package: GIN backbone (registry in layers/),
GNN+ pluggable components (plus/), classification head, config/factory.

Decoupled from active_graph_reader: this package never imports reader code.
"""

from .config import ClassifierConfig
from .graph_classifier import GraphClassifier
from .layers import (BACKBONES, BACKBONE_KWARGS, GINEEncoder, GINELayer,
                     backbone_init_kwargs, get_backbone)
from .factory import (
    build_classifier,
    load_classifier,
    load_fclass,
    classifier_config_from_ckpt,
)

__all__ = [
    'ClassifierConfig', 'GraphClassifier',
    'GINEEncoder', 'GINELayer',
    'BACKBONES', 'BACKBONE_KWARGS', 'backbone_init_kwargs', 'get_backbone',
    'build_classifier', 'load_classifier', 'load_fclass',
    'classifier_config_from_ckpt',
]
