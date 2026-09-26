"""OGBG-PPA dataset (selective loading; 158k graphs never fully in memory).

Data layout (DGL processed format): <root>/ogbg_ppa/processed/
dgl_data_processed. graph_idx uses the OGB global index; train_indices lists
the loaded OGB indices (ascending). Self-loops are added once after loading
(per protocol). With max_graphs > 0, each split keeps N indices (ascending,
deterministic; stratified by class when stratified=True). Always pair PPA
with max_graphs: the full train split is ~158k graphs (>5GB)."""

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import torch
import dgl
from tqdm import tqdm

from .registry import DatasetSpec, Dataset, register, stratified_take


def _ppa_paths(root: str):
    data_dir = Path(root) / 'ogbg_ppa'
    return (data_dir / 'processed' / 'dgl_data_processed',
            data_dir / 'split' / 'species')


def _read_split_ids(split_dir: Path, split: str) -> List[int]:
    df = pd.read_csv(split_dir / f'{split}.csv.gz', compression='gzip', header=None)
    return df.values.flatten().tolist()


def _load_graphs(proc_path: Path, ids: List[int], needs_self_loop: bool,
                 desc: str = 'Loading PPA graphs'):
    """Load graphs/labels by OGB global index; returns (graphs, labels)."""
    graphs, label_dict = dgl.load_graphs(str(proc_path), idx_list=list(ids))
    labels = torch.tensor([int(label_dict['labels'][i].item()) for i in ids])
    if needs_self_loop:
        graphs = [dgl.add_self_loop(g) for g in tqdm(graphs, desc=desc)]
    return graphs, labels


def _ppa_all_labels(spec, min_len: int) -> torch.Tensor:
    """Full label array aligned to OGB global indices (needed before
    stratified sampling).


    Labels are file-level storage in DGL; read cheaply with idx_list=[0]
    and cache to labels_cache.pt; fall back to a single full load when the
    returned length is shorter than min_len (old DGL slices labels).
    """
    proc_path, _ = _ppa_paths(spec.root)
    cache = proc_path.parent / 'labels_cache.pt'
    labels = None
    if cache.exists():
        labels = torch.load(str(cache), weights_only=True).get('labels')
    if labels is not None and len(labels) >= min_len:
        return labels
    _, label_dict = dgl.load_graphs(str(proc_path), idx_list=[0])
    cand = label_dict.get('labels')
    if not isinstance(cand, torch.Tensor):
        cand = torch.as_tensor(cand)
    if len(cand) >= min_len:
        labels = cand
    else:
        print(f'  PPA labels: idx_list read too short ({len(cand)} < {min_len}); '
              'falling back to a single full load')
        _, label_dict = dgl.load_graphs(str(proc_path))
        labels = label_dict['labels']
    torch.save({'labels': labels}, str(cache))
    return labels


def load_ppa(spec: DatasetSpec) -> Dataset:
    if not spec.root:
        raise ValueError(
            f"dataset '{spec.name}' root not set: pass --root"
            " (e.g. --root D:\\ogb-master\\dataset)")

    proc_path, split_dir = _ppa_paths(spec.root)

    out = {'train': ([], []), 'valid': ([], []), 'test': ([], [])}
    kept_ids = {}
    for split in spec.splits:
        ids = sorted(_read_split_ids(split_dir, split))
        if spec.max_graphs > 0 and ids:
            if spec.stratified:
                all_labels = _ppa_all_labels(spec, min_len=max(ids) + 1)
                ids = stratified_take(ids, lambda i: int(all_labels[i]),
                                      spec.max_graphs)
            else:
                ids = ids[:spec.max_graphs]
        graphs, labels = _load_graphs(proc_path, ids, spec.needs_self_loop,
                                      desc=f"PPA {split}")
        out[split] = (graphs, labels)
        kept_ids[split] = ids

    train_graphs, train_labels = out['train']
    valid_graphs, valid_labels = out['valid']
    test_graphs, test_labels = out['test']
    return Dataset(
        spec=spec,
        train_graphs=train_graphs,
        train_labels=train_labels,
        valid_graphs=valid_graphs,
        valid_labels=valid_labels,
        test_graphs=test_graphs,
        test_labels=test_labels,
        train_indices=kept_ids.get('train', []),
        valid_indices=kept_ids.get('valid', []),
        test_indices=kept_ids.get('test', []),
    )


def load_ppa_subset(spec: DatasetSpec, needed: Iterable[int]) -> Dict[int, Tuple[dgl.DGLGraph, int]]:
    """Selectively load by OGB train index; returns {idx: (graph, label)}.

    Only train-split indices load; others warn and skip.
    """
    if not spec.root:
        raise ValueError(
            f"dataset '{spec.name}' root not set: pass --root")
    proc_path, split_dir = _ppa_paths(spec.root)

    train_set = set(_read_split_ids(split_dir, 'train'))
    needed = set(int(i) for i in needed)
    to_load = sorted(needed & train_set)
    skipped = needed - train_set
    if skipped:
        print(f"  WARNING: {len(skipped)} graph_idx not in the train split, ignored")

    graph_cache: Dict[int, Tuple[dgl.DGLGraph, int]] = {}
    if not to_load:
        return graph_cache

    graphs, label_dict = dgl.load_graphs(str(proc_path), idx_list=to_load)
    for i, ogb_idx in enumerate(tqdm(to_load, desc="Loading PPA subset")):
        g = dgl.add_self_loop(graphs[i]) if spec.needs_self_loop else graphs[i]
        graph_cache[ogb_idx] = (g, int(label_dict['labels'][ogb_idx].item()))
    return graph_cache


register(DatasetSpec(
    name='ppa',
    n_classes=37,
    node_feat_dim=0,
    edge_feat_dim=7,
    needs_self_loop=True,
    budget_options=(0.05, 0.10, 0.15),
    load=load_ppa,
    load_graph_subset=load_ppa_subset,
))
