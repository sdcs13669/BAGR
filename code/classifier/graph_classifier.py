"""GraphClassifier: GINE backbone (registry in layers/) + pluggable GNN+
blocks + classification head (linear/ffn).

A backbone only needs to implement the encoder protocol (constructor kwargs
whitelisted in layers.BACKBONE_KWARGS, an out_dim attribute, and
forward(g, node_ids)); block kwargs are filtered by
layers.backbone_init_kwargs and unknown enabled blocks fail loudly. The
legacy scalar constructor signature and forward(g, node_ids) are preserved;
config=ClassifierConfig takes precedence."""

import logging
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

from .config import ClassifierConfig
from .layers import BACKBONE_KWARGS, backbone_init_kwargs, get_backbone

logger = logging.getLogger(__name__)


class GraphClassifier(nn.Module):
    """Graph-level classifier = backbone encoder + readout + Linear/FFN head.


    """

    def __init__(
        self,
        edge_feat_dim: int = 7,
        n_classes: int = 37,
        hidden_dim: int = 300,
        n_layers: int = 5,
        dropout: float = 0.5,
        virtual_node: bool = False,
        residual: bool = False,
        node_feat_dim: int = 0,
        config: Optional[ClassifierConfig] = None,
    ):
        super().__init__()

        if config is not None:
            self.config = config
        else:
            self.config = ClassifierConfig.from_legacy(
                edge_feat_dim=edge_feat_dim, n_classes=n_classes,
                hidden_dim=hidden_dim, n_layers=n_layers, dropout=dropout,
                virtual_node=virtual_node, residual=residual,
                node_feat_dim=node_feat_dim)

        backbone_cls = get_backbone(self.config.backbone)
        supported = BACKBONE_KWARGS[self.config.backbone]
        enabled = (
            ('readout', self.config.readout != 'mean'),
            ('edge_encoder_kind', self.config.edge_encoder != 'linear'),
            ('ffn', self.config.ffn),
            ('pos_enc', self.config.pos_enc != 'none'),
        )
        for block, on in enabled:
            if on and block not in supported:
                raise ValueError(
                    f"backbone '{self.config.backbone}' does not support GNN+ "
                    f"block '{block}' (not declared in layers.BACKBONE_KWARGS)")
        self.encoder = backbone_cls(
            **backbone_init_kwargs(self.config.backbone, self.config))

        embed_dim = getattr(self.encoder, 'out_dim', self.config.hidden_dims[-1])
        if self.config.head == 'ffn':
            self.head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, self.config.n_classes),
            )
        else:
            self.head = nn.Linear(embed_dim, self.config.n_classes)

    # ------------------------------------------------------------------
    # Pretrained weight loading
    # ------------------------------------------------------------------

    def load_pretrained(self, path: str, device: str = 'cpu') -> Dict[str, Any]:
        """Load pretrained weights from checkpoint (needs 'model_state_dict').

        Returns the checkpoint metadata dict.
        """
        ckpt = torch.load(path, map_location=device, weights_only=False)
        missing, unexpected = self.load_state_dict(ckpt['model_state_dict'], strict=True)
        if missing:
            logger.warning("Missing keys in pretrained weights: %s", missing)
        if unexpected:
            logger.warning("Unexpected keys in pretrained weights: %s", unexpected)
        logger.info("Loaded pretrained weights from %s (val_acc=%.4f)",
                    path, ckpt.get('val_acc', 0.0))
        return ckpt

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, g, node_ids: torch.Tensor) -> torch.Tensor:
        g_emb = self.encoder(g, node_ids)
        return self.head(g_emb)

    def get_embedding(self, g, node_ids: torch.Tensor) -> torch.Tensor:
        return self.encoder(g, node_ids)
