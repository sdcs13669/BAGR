"""COLLAB dataset (TUDataset, DGL built-in).

Scientific-collaboration graph classification: 5,000 graphs / 3 classes,
average 74.5 nodes / ~2,458 edges (dense). No node features, no edge
features, no official split. This loader: injects a 1-dim zero edge feature
(the protocol pipeline requires g.edata[feat]); dgl.add_self_loop then fills
self-loop edges with feat=1.0, so original edges carry no collaboration
information and frontier discrimination is purely topological. Splits are a
stratified 80/10/10 (fixed seed 42) with a final global shuffle so
--max-graphs truncation stays class-balanced. Self-loops are added once per
the protocol. graph_idx = local position in ds.train_graphs.

--root points at the TUDataset raw_dir (DGL generates COLLAB_<hash>/ under
it)."""

import random
from typing import List, Tuple

import torch
import dgl
from dgl.data import TUDataset
from tqdm import tqdm

from .registry import DatasetSpec, Dataset, register, stratified_take

SPLIT_SEED = 42
SPLIT_RATIOS = (0.8, 0.1, 0.1)   # train / valid / test


def _load_all(raw_dir: str) -> Tuple[List, List[int]]:
    ds = TUDataset(name='COLLAB', raw_dir=raw_dir)
    graphs, labels = [], []
    for item in tqdm(ds, desc='Loading COLLAB'):
        g, y = item if isinstance(item, tuple) else (item, None)
        if y is None:
            y = int(g.ndata['label'][0].item()) if 'label' in g.ndata else 0
        if 'feat' not in g.edata:
            g.edata['feat'] = torch.zeros(g.num_edges(), 1)
        graphs.append(g)
        labels.append(int(y))
    return graphs, labels


def _split_indices(labels: List[int]) -> Tuple[List[int], List[int], List[int]]:
    """Stratified 80/10/10 split with a fixed seed (deterministic).

    Each class is shuffled then split by ratio so all splits share the same
    class distribution; after concatenating splits, one more global shuffle
    keeps --max-graphs truncation class-balanced (TUDataset order clusters
    by class).
    """
    by_class: dict = {}
    for i, y in enumerate(labels):
        by_class.setdefault(y, []).append(i)

    rng = random.Random(SPLIT_SEED)
    train: List[int] = []
    valid: List[int] = []
    test: List[int] = []
    for cls in sorted(by_class):
        idx = by_class[cls]
        rng.shuffle(idx)
        n_train = int(len(idx) * SPLIT_RATIOS[0])
        n_valid = int(len(idx) * SPLIT_RATIOS[1])
        train += idx[:n_train]
        valid += idx[n_train:n_train + n_valid]
        test += idx[n_train + n_valid:]

    rng.shuffle(train)
    rng.shuffle(valid)
    rng.shuffle(test)
    return train, valid, test


def _finalize(graphs: List, labels: List[int], needs_self_loop: bool,
              desc: str) -> Tuple[List, torch.Tensor]:
    if needs_self_loop and graphs:
        graphs = [dgl.add_self_loop(dgl.remove_self_loop(g))
                  for g in tqdm(graphs, desc=desc)]
    return graphs, torch.tensor(labels)


def load_collab(spec: DatasetSpec) -> Dataset:
    if not spec.root:
        raise ValueError(
            f"dataset '{spec.name}' root not set: pass --root"
            " (e.g. --root D:\\data\\COLLAB)")

    all_graphs, all_labels = _load_all(spec.root)
    train_idx, valid_idx, test_idx = _split_indices(all_labels)

    def _pick(indices):
        """Return (graphs, labels), selected.

        selected holds the split-local positions chosen by stratified sampling
        (ascending; the trajectory graph_idx space); None when not stratified."""
        if spec.max_graphs > 0 and spec.stratified:
            local = stratified_take(range(len(indices)),
                                    [all_labels[i] for i in indices],
                                    spec.max_graphs)
            sel = [indices[i] for i in local]
            return (([all_graphs[i] for i in sel],
                     [all_labels[i] for i in sel]), local)
        if spec.max_graphs > 0:
            indices = indices[:spec.max_graphs]
        return (([all_graphs[i] for i in indices],
                 [all_labels[i] for i in indices]), None)

    parts = {}
    for split, indices in (('train', train_idx), ('valid', valid_idx),
                           ('test', test_idx)):
        if split in spec.splits:
            (gs, ls), sel = _pick(indices)
            parts[split] = (_finalize(gs, ls, spec.needs_self_loop,
                                      f'Self-loop {split}'), sel)
        else:
            parts[split] = (([], torch.tensor([])), None)

    (train_graphs, train_labels), train_sel = parts['train']
    (valid_graphs, valid_labels), valid_sel = parts['valid']
    (test_graphs, test_labels), test_sel = parts['test']

    def _split_out(split_idx, sel):
        if sel is not None:
            return sel
        return split_idx[:spec.max_graphs if spec.max_graphs > 0 else None]

    return Dataset(
        spec=spec,
        train_graphs=train_graphs,
        train_labels=train_labels,
        valid_graphs=valid_graphs,
        valid_labels=valid_labels,
        test_graphs=test_graphs,
        test_labels=test_labels,
        train_indices=_split_out(train_idx, train_sel),
        valid_indices=_split_out(valid_idx, valid_sel),
        test_indices=_split_out(test_idx, test_sel),
    )


register(DatasetSpec(
    name='collab',
    n_classes=3,
    node_feat_dim=0,
    edge_feat_dim=1,
    needs_self_loop=True,
    budget_options=(0.10, 0.15, 0.20),
    load=load_collab,
))
