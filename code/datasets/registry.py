"""Dataset registry: name -> DatasetSpec (loader + metadata).

Adding a dataset = implement a load function and register a DatasetSpec; all
training/generation/eval scripts switch via --dataset. DatasetSpec supports
max_graphs truncation per split and selective loading by original index
(load_graph_subset); dataset modules are lazily registered on first
get_dataset to avoid a dgl import cycle (this module never imports dgl)."""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from typing import Callable, List, Optional, Tuple

import torch

SUBSET_SEED = 20260909


@dataclass
class DatasetSpec:
    name: str
    n_classes: int
    node_feat_dim: int
    edge_feat_dim: int
    needs_self_loop: bool
    budget_options: Tuple[float, ...] = (0.05, 0.10, 0.15)
    root: Optional[str] = None
    splits: Tuple[str, ...] = ('train', 'valid', 'test')
    load: Optional[Callable] = None   # load(spec) -> Dataset
    max_graphs: int = 0
    load_graph_subset: Optional[Callable] = None
    stratified: bool = False


@dataclass
class Dataset:
    spec: DatasetSpec
    train_graphs: List
    train_labels: torch.Tensor
    valid_graphs: List
    valid_labels: torch.Tensor
    test_graphs: List
    test_labels: torch.Tensor
    train_indices: List[int]
    test_indices: List[int]
    valid_indices: List[int] = field(default_factory=list)


DATASETS: dict = {}
_CACHE: dict = {}
_REGISTERED = False


def _ensure_registered() -> None:
    """Load dataset modules on first get_dataset (avoids a dgl import cycle)."""
    global _REGISTERED
    if not _REGISTERED:
        from . import cifar10, collab, ppa
        _REGISTERED = True


class _LazyDataset(Dataset):
    """Lazy wrapper: get_dataset only reads the spec; spec.load runs on first

    access to a data field and the result is cached.
    """

    _DATA_FIELDS = frozenset({
        'train_graphs', 'train_labels', 'valid_graphs', 'valid_labels',
        'test_graphs', 'test_labels', 'train_indices', 'test_indices',
        'valid_indices',
    })

    def __init__(self, spec: DatasetSpec) -> None:
        self.spec = spec
        self._real: Optional[Dataset] = None

    def _ensure_loaded(self) -> Dataset:
        if self._real is None:
            self._real = self.spec.load(self.spec)
        return self._real

    def __getattr__(self, name: str):
        if name in _LazyDataset._DATA_FIELDS:
            return getattr(self._ensure_loaded(), name)
        raise AttributeError(f"{type(self).__name__!r} has no attribute {name!r}")

    def __repr__(self) -> str:
        return (f"_LazyDataset(spec={self.spec.name!r}, "
                f"loaded={self._real is not None})")

    def __eq__(self, other) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)


def register(spec: DatasetSpec) -> None:
    DATASETS[spec.name] = spec


def stratified_take(indices, label_of, n: int,
                    seed: int = SUBSET_SEED) -> List[int]:
    """Take n indices from candidates stratified by class (deterministic);
    returns an ascending list.

    Per-class quotas use the largest-remainder method; within a class the
    RNG shuffles then takes the first k. Returns everything (ascending)
    when n <= 0 or n >= len(indices).
    """
    indices = list(indices)
    if n <= 0 or n >= len(indices):
        return sorted(indices)

    def _label(i):
        return label_of(i) if callable(label_of) else label_of[i]

    by_class: dict = {}
    for i in indices:
        by_class.setdefault(_label(i), []).append(i)
    classes = sorted(by_class)
    total = len(indices)
    exact = {c: n * len(by_class[c]) / total for c in classes}
    ks = {c: int(exact[c]) for c in classes}
    leftover = n - sum(ks.values())
    for c in sorted(classes, key=lambda c: (-(exact[c] - ks[c]), c))[:leftover]:
        ks[c] += 1
    rng = random.Random(seed)
    picked: List[int] = []
    for c in classes:
        pool = by_class[c][:]
        rng.shuffle(pool)
        picked += pool[:ks[c]]
    return sorted(picked)


def get_dataset(
    name: str,
    root: Optional[str] = None,
    splits=('train', 'valid', 'test'),
    max_graphs: int = 0,
    stratified: bool = False,
) -> Dataset:
    _ensure_registered()
    if name not in DATASETS:
        raise ValueError(f"unknown dataset '{name}', available: {sorted(DATASETS)}")
    splits = tuple(splits)
    cache_key = (name, root, splits, int(max_graphs), bool(stratified))
    if cache_key not in _CACHE:
        spec = DATASETS[name]
        if spec.load is None:
            raise ValueError(f"dataset '{name}' does not implement a load function")
        spec = replace(spec, root=root, splits=splits, max_graphs=int(max_graphs),
                       stratified=bool(stratified))
        _CACHE[cache_key] = _LazyDataset(spec)
    return _CACHE[cache_key]


def has_self_loops(g) -> bool:
    u, v = g.edges()
    return bool(torch.any(u == v).item())


def release_dataset(ds) -> None:
    """Drop the cached reference and loaded data (per-split load/eval/free).

    Removes the instance from _CACHE and clears its _real; the caller must
    also drop its own references (graphs/labels) for the memory to be freed.
    """
    for key in [k for k, v in _CACHE.items() if v is ds]:
        del _CACHE[key]
    if isinstance(ds, _LazyDataset):
        ds._real = None
