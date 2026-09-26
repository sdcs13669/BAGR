"""Classifier config: per-layer hidden/dropout, JSON-serializable for config
files and checkpoints.

Serialization folds to scalars (hidden_dim/n_layers/dropout) when all layers
are equal, otherwise uses lists (hidden_dims/dropouts); from_dict accepts both
forms."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .plus import EDGE_ENCODER_KINDS


def _as_list(value, n: Optional[int] = None) -> List:
    """Scalar -> [v]*n; list unchanged (normalization helper)."""
    if isinstance(value, (list, tuple)):
        return list(value)
    if n is None:
        raise ValueError("scalar form requires the layer count n")
    return [value] * n


@dataclass
class ClassifierConfig:
    """Full-graph classifier structure config.

    hidden_dims[i] / dropouts[i] correspond to layer i; backbone 'gin'

    head: 'linear' (single Linear, default) or 'ffn' (two-layer MLP).
    gnn_norm: inter-layer normalization (not the BatchNorm inside the
    GINELayer MLP); None keeps each backbone default (GIN: BatchNorm),
    or explicitly 'batchnorm' / 'layernorm' / 'none'.
    readout: graph pooling, 'mean' (default, AvgPooling) | 'meanmax'
    (concat of mean and max, doubles the output dim; neither scales with
    node count, so full-graph and revealed-subgraph magnitudes match) |
    'sum' (SumPooling; scales with node count, avoid on subgraphs).

    GNN+ block fields (arXiv 2502.09263; defaults reproduce the original
    behavior and state-dict keys; see plus/__init__.py):
    edge_encoder: edge feature integration (plus/edge.py), 'linear'
    (default) or 'mlp'.
    ffn / ffn_dropout: per-layer FFN (plus/ffn.py, residual + BN), off by
    default.
    pos_enc / pos_enc_ksteps: positional encoding (plus/pos_enc.py, RWSE);
    off by default; input dim = node_feat_dim + pos_enc_ksteps when on.
    """

    backbone: str = 'gin'
    edge_feat_dim: int = 7
    node_feat_dim: int = 0
    n_classes: int = 37
    hidden_dims: List[int] = field(default_factory=lambda: [300] * 5)
    dropouts: List[float] = field(default_factory=lambda: [0.5] * 5)
    virtual_node: bool = False
    residual: bool = False
    n_heads: Optional[List[int]] = None
    head: str = 'linear'
    gnn_norm: Optional[str] = None
    readout: str = 'mean'
    edge_encoder: str = 'linear'
    ffn: bool = False
    ffn_dropout: float = 0.0
    pos_enc: str = 'none'
    pos_enc_ksteps: int = 8

    @property
    def hidden_dim(self) -> int:
        return self.hidden_dims[0]

    @property
    def n_layers(self) -> int:
        return len(self.hidden_dims)

    @property
    def dropout(self) -> float:
        return self.dropouts[0]

    def to_dict(self) -> Dict:
        """JSON-friendly dict; folds to scalar + n_layers when uniform."""
        d: Dict = {'backbone': self.backbone,
                   'edge_feat_dim': self.edge_feat_dim,
                   'node_feat_dim': self.node_feat_dim,
                   'n_classes': self.n_classes,
                   'virtual_node': self.virtual_node,
                   'residual': self.residual}
        if len(set(self.hidden_dims)) == 1:
            d['hidden_dim'] = self.hidden_dims[0]
            d['n_layers'] = len(self.hidden_dims)
        else:
            d['hidden_dims'] = list(self.hidden_dims)
        if len(set(self.dropouts)) == 1:
            d['dropout'] = self.dropouts[0]
        else:
            d['dropouts'] = list(self.dropouts)
        if self.backbone == 'gat' and self.n_heads is not None:
            if len(set(self.n_heads)) == 1:
                d['n_heads'] = self.n_heads[0]
            else:
                d['n_heads'] = list(self.n_heads)
        if self.head != 'linear':
            d['head'] = self.head
        if self.gnn_norm is not None:
            d['gnn_norm'] = self.gnn_norm
        if self.readout != 'mean':
            d['readout'] = self.readout
        if self.edge_encoder != 'linear':
            d['edge_encoder'] = self.edge_encoder
        if self.ffn:
            d['ffn'] = True
        if self.ffn_dropout != 0.0:
            d['ffn_dropout'] = self.ffn_dropout
        if self.pos_enc != 'none':
            d['pos_enc'] = self.pos_enc
        if self.pos_enc != 'none' and self.pos_enc_ksteps != 8:
            d['pos_enc_ksteps'] = self.pos_enc_ksteps
        return d

    @staticmethod
    def from_dict(d: Dict) -> 'ClassifierConfig':
        """Accept to_dict output or the legacy scalar form."""
        if 'hidden_dims' in d:
            hidden_dims = list(d['hidden_dims'])
        else:
            hidden_dims = [int(d.get('hidden_dim', 300))] * int(d.get('n_layers', 5))
        if 'dropouts' in d:
            dropouts = list(d['dropouts'])
        else:
            dropouts = [float(d.get('dropout', 0.5))] * len(hidden_dims)
        if len(dropouts) != len(hidden_dims):
            raise ValueError(f"dropouts length {len(dropouts)} != hidden_dims length {len(hidden_dims)}")
        head = d.get('head', 'linear')
        if head not in ('linear', 'ffn'):
            raise ValueError(f"unknown head '{head}', choose: linear, ffn")
        gnn_norm = d.get('gnn_norm')
        if gnn_norm is not None and gnn_norm not in ('batchnorm', 'layernorm', 'none'):
            raise ValueError(
                f"unknown gnn_norm '{gnn_norm}', choose: batchnorm, layernorm, none")
        readout = d.get('readout', 'mean')
        if readout not in ('mean', 'meanmax', 'sum'):
            raise ValueError(f"unknown readout '{readout}', choose: mean, meanmax, sum")
        edge_encoder = d.get('edge_encoder', 'linear')
        if edge_encoder not in EDGE_ENCODER_KINDS:
            raise ValueError(
                f"unknown edge encoder '{edge_encoder}', choose: {EDGE_ENCODER_KINDS}")
        pos_enc = d.get('pos_enc', 'none')
        if pos_enc not in ('none', 'rwse'):
            raise ValueError(f"unknown pos_enc '{pos_enc}', choose: none, rwse")
        return ClassifierConfig(
            backbone=d.get('backbone', 'gin'),
            edge_feat_dim=int(d.get('edge_feat_dim', 7)),
            node_feat_dim=int(d.get('node_feat_dim', 0)),
            n_classes=int(d.get('n_classes', 37)),
            hidden_dims=hidden_dims,
            dropouts=dropouts,
            virtual_node=bool(d.get('virtual_node', False)),
            residual=bool(d.get('residual', False)),
            n_heads=(d.get('n_heads') if isinstance(d.get('n_heads'), (list, tuple))
                     else ([int(d['n_heads'])] if 'n_heads' in d else None)),
            head=head,
            gnn_norm=gnn_norm,
            readout=readout,
            edge_encoder=edge_encoder,
            ffn=bool(d.get('ffn', False)),
            ffn_dropout=float(d.get('ffn_dropout', 0.0)),
            pos_enc=pos_enc,
            pos_enc_ksteps=int(d.get('pos_enc_ksteps', 8)),
        )

    @staticmethod
    def from_legacy(edge_feat_dim: int, n_classes: int, hidden_dim: int = 300,
                    n_layers: int = 5, dropout: float = 0.5,
                    virtual_node: bool = False, residual: bool = False,
                    node_feat_dim: int = 0, backbone: str = 'gin',
                    n_heads: Optional[List[int]] = None,
                    head: str = 'linear',
                    gnn_norm: Optional[str] = None,
                    readout: str = 'mean',
                    edge_encoder: str = 'linear',
                    ffn: bool = False,
                    ffn_dropout: float = 0.0,
                    pos_enc: str = 'none',
                    pos_enc_ksteps: int = 8) -> 'ClassifierConfig':
        """Legacy scalar arguments -> config."""
        return ClassifierConfig(
            backbone=backbone,
            edge_feat_dim=edge_feat_dim, n_classes=n_classes,
            node_feat_dim=node_feat_dim, virtual_node=virtual_node,
            residual=residual, n_heads=n_heads, head=head, gnn_norm=gnn_norm,
            readout=readout, edge_encoder=edge_encoder, ffn=ffn,
            ffn_dropout=ffn_dropout, pos_enc=pos_enc,
            pos_enc_ksteps=pos_enc_ksteps,
            hidden_dims=[hidden_dim] * n_layers, dropouts=[dropout] * n_layers,
        )
