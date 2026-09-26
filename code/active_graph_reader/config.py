"""ReaderConfig: reader structure config (per-layer) + factory + ckpt inference.


Supports per-layer width/dropout lists (beyond uniform scalars); a single
build_reader(config) entry point; tolerant checkpoint-config parsing.

Serialization folds per-layer lists to scalars when all layers are equal
(same convention as classifier.ClassifierConfig).
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch


def _fold(d: Dict, values: List, base: str) -> None:
    """Fold to {base: v, n_{base}s: len} if all equal, else {base+'s': list}."""
    if len(set(values)) == 1:
        d[base] = values[0]
        d['n_' + base + 's'] = len(values)
    else:
        d[base + 's'] = list(values)


def _unfold(d: Dict, base: str, default: List) -> List:
    """Inverse of to_dict: accepts {base, n_bases} / {base+'s'} / missing;
    None counts as missing (nargs="+" args serialize to null)."""
    vals = d.get(base + 's')
    if vals is not None:
        return list(vals)
    scalar = d.get(base)
    if scalar is not None:
        n = _to_int(d.get('n_' + base + 's'), len(default))
        return [scalar] * n
    return list(default)


def _to_int(v, default):
    return default if v is None else int(v)


def _to_float(v, default):
    return default if v is None else float(v)


@dataclass
class ReaderConfig:
    """Reader structure config.

    encoder_*: GIN layers of FrontierNodesEncoder (per-layer hidden/dropout)
    scorer_*: selection scorer MLP (per-layer hidden/dropout)
    n_classes: evidence only (evidence-head class count, must be > 0)
    """

    kind: str = 'evidence'                # reader kind
    edge_feat_dim: int = 7
    node_feat_dim: int = 0
    encoder_hidden_dims: List[int] = field(default_factory=lambda: [128, 128])
    encoder_dropouts: List[float] = field(default_factory=lambda: [0.1, 0.1])
    scorer_hidden_dims: List[int] = field(default_factory=lambda: [128, 128])
    scorer_dropouts: List[float] = field(default_factory=lambda: [0.1, 0.1])
    stochastic: bool = False
    temperature: float = 1.0
    n_classes: int = 0
    evidence_variant: str = 'full'

    @property
    def hidden_dim(self) -> int:
        return self.encoder_hidden_dims[-1]

    @property
    def n_layers(self) -> int:
        return len(self.encoder_hidden_dims)

    @property
    def dropout(self) -> float:
        return self.encoder_dropouts[0]

    def validate(self) -> None:
        if self.kind != 'evidence':
            raise ValueError(f"kind must be 'evidence', got '{self.kind}'")
        if len(self.encoder_dropouts) != len(self.encoder_hidden_dims):
            raise ValueError("encoder_dropouts and encoder_hidden_dims length mismatch")
        if len(self.scorer_dropouts) != len(self.scorer_hidden_dims):
            raise ValueError("scorer_dropouts and scorer_hidden_dims length mismatch")
        if self.kind == 'evidence' and self.n_classes <= 0:
            raise ValueError("evidence reader requires n_classes > 0")
        if self.kind == 'evidence' and \
                self.evidence_variant not in ('full', 'no_pool', 'no_lstm', 'v1'):
            raise ValueError(f"invalid evidence_variant '{self.evidence_variant}'"
                             f" (choose full/no_pool/no_lstm/v1)")
        if self.kind != 'evidence' and self.evidence_variant != 'full':
            raise ValueError("evidence_variant only applies to kind='evidence'")

    def to_dict(self) -> Dict:
        self.validate()
        d: Dict = {'kind': self.kind,
                   'edge_feat_dim': self.edge_feat_dim,
                   'node_feat_dim': self.node_feat_dim,
                   'stochastic': self.stochastic,
                   'temperature': self.temperature}
        _fold(d, self.encoder_hidden_dims, 'hidden_dim')
        _fold(d, self.encoder_dropouts, 'dropout')
        if self.kind == 'evidence':
            _fold(d, self.scorer_hidden_dims, 'scorer_hidden_dim')
            _fold(d, self.scorer_dropouts, 'scorer_dropout')
        if self.kind == 'evidence':
            d['n_classes'] = self.n_classes
            d['evidence_variant'] = self.evidence_variant
        return d

    @staticmethod
    def from_dict(d: Dict) -> 'ReaderConfig':
        kind = d.get('kind', d.get('reader_kind', 'evidence'))
        encoder_hidden = _unfold(d, 'hidden_dim', [128, 128])
        encoder_drop = _unfold(d, 'dropout', [0.1] * len(encoder_hidden))
        if len(encoder_drop) != len(encoder_hidden):
            encoder_drop = (encoder_drop * len(encoder_hidden))[:len(encoder_hidden)]
        cfg = ReaderConfig(
            kind=kind,
            edge_feat_dim=_to_int(d.get('edge_feat_dim'), 7),
            node_feat_dim=_to_int(d.get('node_feat_dim'), 0),
            encoder_hidden_dims=[int(x) for x in encoder_hidden],
            encoder_dropouts=[float(x) for x in encoder_drop],
            scorer_hidden_dims=[int(x) for x in
                                _unfold(d, 'scorer_hidden_dim',
                                        [encoder_hidden[-1]] * 2)],
            scorer_dropouts=[float(x) for x in
                             _unfold(d, 'scorer_dropout',
                                     encoder_drop[:1] * 2)],
            stochastic=bool(d.get('stochastic', False)),
            temperature=_to_float(d.get('temperature'), 1.0),
            n_classes=_to_int(d.get('n_classes'), 0),
            evidence_variant=str(d.get('evidence_variant') or 'full'),
        )
        cfg.validate()
        return cfg

    @staticmethod
    def from_legacy(kind: str = 'evidence', edge_feat_dim: int = 7,
                    node_feat_dim: int = 0, hidden_dim: int = 128,
                    n_layers: int = 2, dropout: float = 0.1,
                    stochastic: bool = False, temperature: float = 1.0,
                    n_classes: int = 0,
                    ) -> 'ReaderConfig':
        """Legacy scalar-argument construction."""
        cfg = ReaderConfig(
            kind=kind, edge_feat_dim=edge_feat_dim, node_feat_dim=node_feat_dim,
            encoder_hidden_dims=[hidden_dim] * n_layers,
            encoder_dropouts=[dropout] * n_layers,
            scorer_hidden_dims=[hidden_dim] * 2,
            scorer_dropouts=[dropout] * 2,
            stochastic=stochastic, temperature=temperature,
            n_classes=n_classes,
        )
        cfg.validate()
        return cfg


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

_CONV_RE = re.compile(r'^frontier_encoder\.convs\.(\d+)\.mlp\.0\.weight$')
_SCORER_HIDDEN_RE = re.compile(r'^reader_mlp\.mlp\.(\d+)\.weight$')


def infer_reader_config(state_dict: Dict[str, torch.Tensor]) -> ReaderConfig:
    """Infer reader structure from weight shapes (shapes are ground truth).

    """
    conv_idx = {}
    for key, tensor in state_dict.items():
        m = _CONV_RE.match(key)
        if m:
            conv_idx[int(m.group(1))] = int(tensor.shape[0] // 2)  # bottleneck=2h
    if not conv_idx:
        raise ValueError("frontier_encoder.convs.* not found in state_dict")
    encoder_hidden = [conv_idx[i] for i in sorted(conv_idx)]

    if 'frontier_encoder.node_feat_proj.weight' in state_dict:
        node_feat_dim = int(state_dict['frontier_encoder.node_feat_proj.weight'].shape[1])
    else:
        node_feat_dim = 0

    edge_feat_dim = int(
        state_dict['frontier_encoder.convs.0.edge_encoder.weight'].shape[1])

    kind = 'evidence'
    if any(k.startswith(('evidence_head', 'evidence_lstm'))
           for k in state_dict):
        kind = 'evidence'

    n_classes = 0
    if kind == 'evidence':
        head_layers = [(int(k.split('.')[1]), int(t.shape[0]))
                       for k, t in state_dict.items()
                       if k.startswith('evidence_head.') and
                       k.endswith('.weight')]
        if head_layers:
            n_classes = max(head_layers)[1]

    scorer_hidden = [128, 128]
    if kind == 'evidence':
        linears = []                             # (block_idx, out_dim)
        for key, tensor in state_dict.items():
            mm = _SCORER_HIDDEN_RE.match(key)
            if mm and int(mm.group(1)) % 3 == 0:
                linears.append((int(mm.group(1)) // 3, int(tensor.shape[0])))
        if linears:
            linears.sort()
            scorer_hidden = [h for _, h in linears[:-1]]

    cfg = ReaderConfig(
        kind=kind,
        edge_feat_dim=edge_feat_dim,
        node_feat_dim=node_feat_dim,
        encoder_hidden_dims=encoder_hidden,
        encoder_dropouts=[0.1] * len(encoder_hidden),
        scorer_hidden_dims=scorer_hidden,
        scorer_dropouts=[0.1] * len(scorer_hidden),
        n_classes=n_classes,
    )
    return cfg


def reader_config_from_checkpoint(ckpt: dict,
                                  state_dict: Optional[dict] = None) -> ReaderConfig:
    """checkpoint -> ReaderConfig: structure from weight shapes; non-structural
    hyperparameters (temperature/stochastic) from the embedded config."""
    sd = state_dict if state_dict is not None else ckpt.get('reader_state_dict', {})
    cfg = infer_reader_config(sd)
    embedded = ckpt.get('config')
    if isinstance(embedded, dict):
        embedded = dict(embedded)
        embedded = {k: v for k, v in embedded.items() if v is not None}
        for k in ('hidden_dim', 'hidden_dims', 'n_hidden_dims',
                  'n_layers', 'n_layers_reader',
                  'n_dropouts', 'n_scorer_hidden_dims', 'n_scorer_dropouts',
                  'edge_feat_dim', 'node_feat_dim'):
            embedded.pop(k, None)
        merged = cfg.to_dict()
        merged.update(embedded)
        cfg = ReaderConfig.from_dict(merged)
    return cfg


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def build_reader(config: ReaderConfig):
    """ReaderConfig -> reader instance."""
    config.validate()
    if config.kind == 'evidence':
        from .reader.evidence import EvidenceReader
        return EvidenceReader(config=config)
    raise ValueError(f"unsupported reader kind: {config.kind!r}")
