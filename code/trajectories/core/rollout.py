"""Rollout helpers: model building from checkpoint config, batched BC reader
rollout, and candidate validation."""

import random
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
import dgl

from active_graph_reader import build_reader as _build_reader
from active_graph_reader.config import reader_config_from_checkpoint
from active_graph_reader.engine.reveal_action import reveal_node
from classifier import GraphClassifier, classifier_config_from_ckpt
from tools.ckpt_io import pad_reader_mlp_input, strip_classifier_keys
from .trajectory import Trajectory


def build_classifier(ckpt: dict, device, requires_grad: bool = False) -> GraphClassifier:
    """Rebuild the upper-bound classifier from a checkpoint.

    requires_grad=False (default): frozen weights; requires_grad=True keeps
    the autograd graph (needed by GradientTeacher). Frozen means weights
    only, no optimizer updates.
    """
    clf = GraphClassifier(config=classifier_config_from_ckpt(ckpt)).to(device)
    clf.load_state_dict(ckpt['model_state_dict'], strict=True)
    clf.eval()
    if not requires_grad:
        for p in clf.parameters():
            p.requires_grad = False
    return clf


def build_reader(ckpt: dict, device, stochastic: bool = False,
                 temperature: float = None):
    """Rebuild a reader from a checkpoint.

    """
    sd = ckpt.get('reader_state_dict', ckpt.get('state_dict'))
    sd = strip_classifier_keys(dict(sd))
    cfg = reader_config_from_checkpoint(ckpt, sd)
    if temperature is not None:
        cfg.temperature = float(temperature)
    if stochastic:
        cfg.stochastic = True
    reader = _build_reader(cfg).to(device)
    target_dim = reader.reader_mlp.mlp[0].in_features if cfg.kind == 'base' else None
    if target_dim is not None:
        sd, _ = pad_reader_mlp_input(sd, target_dim)
    reader.load_state_dict(sd, strict=True)
    reader.eval()
    return reader


def build_edge_key_map(g):
    """Per-graph canonical edge key -> edge id map (GraphState interface).

    Without it every GraphState rebuilds the map internally, which at tens
    of thousands of states per round exhausts memory; build once per graph
    and share across its states.
    """
    if 'feat' not in g.edata:
        return None
    src, dst = g.edges()
    m = {}
    for eid in range(g.num_edges()):
        u, v = int(src[eid]), int(dst[eid])
        key = (min(u, v), max(u, v))
        if key not in m:
            m[key] = eid
    return m


def rollout_batch(reader, states: list, rng: random.Random) -> list:
    """Cross-graph lockstep batched rollout (argmax greedy reading).

    states: GraphStates with seed and _budget_limit set. Returns one
    (reveal_order, final_state) per input; no-action states give ([], state).
    """
    n_total = len(states)
    orders = [[] for _ in range(n_total)]
    active = [True] * n_total
    max_budget = max(s._budget_limit for s in states)
    for _step in range(max_budget):
        active_idx = [j for j in range(n_total)
                      if active[j] and not states[j].is_frontier_empty]
        if not active_idx:
            break
        batch = reader.forward_states([states[j] for j in active_idx],
                                      temperature=reader.temperature)
        for k, j in enumerate(active_idx):
            _scores, probs, frontier_nodes = batch[k]
            if not frontier_nodes:
                active[j] = False
                continue
            node = frontier_nodes[int(probs.argmax().item())]
            orders[j].append(node)
            reveal_node(states[j], node)
            if states[j].get_budget_used() >= states[j]._budget_limit:
                active[j] = False
    return [(o, s) for o, s in zip(orders, states)]


def validate_candidates(cands, builder, classifier, device, train_labels,
                        batch_size):
    """Validate candidates with the upper-bound classifier (argmax == label).

    cands: [(graph_idx, ratio, traj, fin_state)]; returns the passing
    [(graph_idx, ratio, traj)]; final states are released on return.
    """
    accepted = []
    for c_start in range(0, len(cands), batch_size):
        chunk = cands[c_start:c_start + batch_size]
        sub_graphs = []
        for _gi, _ratio, _traj, state in chunk:
            sub_g, _ = builder.build(state)
            sub_graphs.append(sub_g)
        batch_g = dgl.batch(sub_graphs).to(device)
        nids = torch.zeros(batch_g.num_nodes(), dtype=torch.long, device=device)
        with torch.no_grad():
            logits = classifier(batch_g, nids)
        for k, (gi, ratio, traj, state) in enumerate(chunk):
            logits_i = logits[k:k + 1]
            pred = int(logits_i.argmax(dim=-1).item())
            ce = float(torch.nn.functional.cross_entropy(
                logits_i, torch.tensor([int(train_labels[gi])],
                                       device=device)).item())
            traj.ce = ce
            if pred == int(train_labels[gi]):
                traj.final_pred = pred
                accepted.append((gi, ratio, traj))
    return accepted


def count_labels(trajs: list, train_labels) -> Dict[int, int]:
    """Count trajectories per label (graph_idx -> local position -> train_labels)."""
    cnt = defaultdict(int)
    for t in trajs:
        cnt[int(train_labels[t.graph_idx])] += 1
    return dict(cnt)
