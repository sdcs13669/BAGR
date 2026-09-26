"""Active graph reading (AGR): budget-limited node revelation on graphs.

Subpackages:
- core    - BaseReader interface and logging
- engine  - core engine (GraphState, reveal, RevealLoop)
- gnn     - subgraph construction and frontier encoding
- reader  - reader implementations (EvidenceReader, fixed strategies)
- config  - ReaderConfig (per-layer structure) + build_reader factory
            + checkpoint inference

The GIN classifier (GraphClassifier/GINEEncoder) lives in the separate
``classifier`` package.
"""

from .reader import (
    BaseReader,
    FixedStrategyReader,
    EvidenceReader,
    FrontierNodesEncoder,
    reveal_one,
    reveal_batch,
)
from .config import ReaderConfig, build_reader

__all__ = [
    'BaseReader',
    'EvidenceReader',
    'FixedStrategyReader',
    'FrontierNodesEncoder',
    'reveal_one',
    'reveal_batch',
    'ReaderConfig',
    'build_reader',
]
