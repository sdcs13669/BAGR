"""CIFAR10-Superpixel dataset (DGL built-in, use_feature=True).

Node features are 5-dim (2 position + 3 color), edge features 1-dim, 10
classes. No validation split: training uses the official DGL train split,
valid is always empty, evaluation uses test. Data directory comes from
DatasetSpec.root (--root). max_graphs truncates after loading."""

from typing import List, Tuple

import torch
import dgl
from tqdm import tqdm

from .registry import DatasetSpec, Dataset, register, has_self_loops, \
    stratified_take


def _load_split(split: str, raw_dir: str) -> Tuple[List[dgl.DGLGraph], List[int]]:
    from dgl.data import CIFAR10SuperPixelDataset
    ds = CIFAR10SuperPixelDataset(split=split, use_feature=True, raw_dir=raw_dir)
    graphs = []
    labels = []
    for g, y in tqdm(ds, desc=f"Loading {split}"):
        graphs.append(g)
        labels.append(int(y))
    return graphs, labels


def load_cifar10(spec: DatasetSpec) -> Dataset:
    if not spec.root:
        raise ValueError(
            f"dataset '{spec.name}' root not set: pass --root"
            " (e.g. --root D:\\superpixels)")

    train_graphs, train_labels = [], []
    test_graphs, test_labels = [], []
    if 'train' in spec.splits:
        train_graphs, train_labels = _load_split('train', spec.root)
    if 'test' in spec.splits:
        test_graphs, test_labels = _load_split('test', spec.root)

    train_indices = list(range(len(train_graphs)))
    test_indices = list(range(len(test_graphs)))

    if spec.max_graphs > 0:
        if spec.stratified:
            sel = stratified_take(range(len(train_graphs)), train_labels,
                                  spec.max_graphs)
            train_graphs = [train_graphs[i] for i in sel]
            train_labels = [train_labels[i] for i in sel]
            train_indices = sel
            sel = stratified_take(range(len(test_graphs)), test_labels,
                                  spec.max_graphs)
            test_graphs = [test_graphs[i] for i in sel]
            test_labels = [test_labels[i] for i in sel]
            test_indices = sel
        else:
            if 'train' in spec.splits:
                train_graphs = train_graphs[:spec.max_graphs]
                train_labels = train_labels[:spec.max_graphs]
            if 'test' in spec.splits:
                test_graphs = test_graphs[:spec.max_graphs]
                test_labels = test_labels[:spec.max_graphs]

    if spec.needs_self_loop:
        if 'train' in spec.splits and train_graphs and not has_self_loops(train_graphs[0]):
            train_graphs = [dgl.add_self_loop(g)
                            for g in tqdm(train_graphs, desc="Self-loop train")]
        if 'test' in spec.splits and test_graphs and not has_self_loops(test_graphs[0]):
            test_graphs = [dgl.add_self_loop(g)
                           for g in tqdm(test_graphs, desc="Self-loop test")]

    return Dataset(
        spec=spec,
        train_graphs=train_graphs,
        train_labels=torch.tensor(train_labels),
        valid_graphs=[],
        valid_labels=torch.tensor([]),
        test_graphs=test_graphs,
        test_labels=torch.tensor(test_labels),
        train_indices=train_indices,
        test_indices=test_indices,
    )


register(DatasetSpec(
    name='cifar10',
    n_classes=10,
    node_feat_dim=5,
    edge_feat_dim=1,
    needs_self_loop=True,
    budget_options=(0.05, 0.10, 0.15),
    load=load_cifar10,
))
