"""Dataset registry package.

Note on shadowing: this top-level package named "datasets" shadows the
HuggingFace datasets package, and the dgl->torchdata init chain imports
"datasets" implicitly. Hence registry.py deliberately does not depend on
dgl, and dataset modules register lazily on first get_dataset()."""

from .registry import (
    DatasetSpec,
    Dataset,
    DATASETS,
    get_dataset,
    register,
    has_self_loops,
    release_dataset,
)

__all__ = [
    'DatasetSpec', 'Dataset', 'DATASETS', 'get_dataset', 'register',
    'has_self_loops', 'release_dataset',
]
