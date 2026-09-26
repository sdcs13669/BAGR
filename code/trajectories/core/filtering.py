"""Trajectory filtering: similarity dedup + per-graph top-k + teacher diversity.

Stage 1: within a graph, greedily dedup by ascending CE (cosine similarity
|A n B| / sqrt(|A||B|) above the threshold counts as a duplicate). Stage 2:
keep per-graph top-k by ascending CE, trying to cover each teacher type."""

import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

from tools.trajectory_io import load_trajectory_file
from .trajectory import Trajectory


def similarity(set1: Set[int], set2: Set[int]) -> float:
    """Cosine similarity: |A n B| / sqrt(|A| * |B|)."""
    n_inter = len(set1 & set2)
    if n_inter == 0:
        return 0.0
    return n_inter / math.sqrt(len(set1) * len(set2))


def dedup_group(records: List[tuple], threshold: float) -> List[tuple]:
    """Greedy dedup: walk in ascending CE (quality) order and drop any

    trajectory whose similarity to a kept one exceeds the threshold.

    records: (quality, teacher, traj) with quality = -ce; output keeps
    descending quality order.
    """
    if len(records) <= 1:
        return records

    records = sorted(records, key=lambda r: r[0], reverse=True)
    kept = []
    kept_sets = []
    for rec in records:
        traj_set = set(rec[2].reveal_order)
        if any(similarity(traj_set, ks) > threshold for ks in kept_sets):
            continue
        kept.append(rec)
        kept_sets.append(traj_set)
    return kept


def select_top_k(records: List[tuple], top_k: int, require_diversity: bool) -> List[Trajectory]:
    """Per-graph top-k selection.

    records: (quality, teacher, traj), descending quality. Strategy: always
    keep the best one, then try to include one per teacher type, then fill by
    quality.
    """
    if len(records) <= top_k:
        return [r[2] for r in records]

    selected = []
    used_teachers: Set[str] = set()
    remaining = list(records)

    selected.append(remaining.pop(0))
    used_teachers.add(selected[0][1])

    if require_diversity:
        all_types = sorted({r[1] for r in remaining})
        for ttype in all_types:
            if len(selected) >= top_k:
                break
            if ttype in used_teachers:
                continue
            for i, r in enumerate(remaining):
                if r[1] == ttype:
                    selected.append(remaining.pop(i))
                    used_teachers.add(ttype)
                    break

    while len(selected) < top_k and remaining:
        selected.append(remaining.pop(0))

    return [r[2] for r in selected]


def filter_file(
    fpath: Path, teachers: List[str], top_k: int, require_diversity: bool,
    sim_threshold: float,
) -> Dict:
    """Filter one trajectory file; returns {data: save dict, report: stats}."""
    data = load_trajectory_file(fpath)
    orig_config = data.get('config', {})
    teacher_names = teachers or list(orig_config.get('teachers', [])
                                     or data['trajectories'].keys())

    records: List[tuple] = []  # (quality, teacher, traj)
    for tname in teacher_names:
        for per_graph in data['trajectories'].get(tname, []):
            for traj in per_graph:
                records.append((-traj.ce, tname, traj))

    n_before = len(records)

    by_graph: Dict[int, List[tuple]] = defaultdict(list)
    for rec in records:
        by_graph[rec[2].graph_idx].append(rec)

    max_g_idx = max(by_graph, default=-1)
    result = {tname: [[] for _ in range(max_g_idx + 1)] for tname in teacher_names}
    n_after_dedup = 0
    diverse = single = 0
    for g_idx, recs in by_graph.items():
        deduped = dedup_group(recs, sim_threshold)
        n_after_dedup += len(deduped)
        kept = select_top_k(deduped, top_k, require_diversity)
        if len({t.teacher_type for t in kept}) > 1:
            diverse += 1
        else:
            single += 1
        for traj in kept:
            result[traj.teacher_type][g_idx].append(traj)

    report = {
        'n_before': n_before,
        'n_after_dedup': n_after_dedup,
        'dedup_removed': n_before - n_after_dedup,
        'n_after': sum(len(l) for lists in result.values() for l in lists),
        'n_graphs_after': len(by_graph),
        'diverse_graphs': diverse,
        'single_teacher_graphs': single,
    }
    save = {
        'trajectories': result,
        'config': {
            **orig_config,
            'top_k': top_k,
            'require_diversity': require_diversity,
            'sim_threshold': sim_threshold,
            'filter_report': report,
        },
    }
    return {'data': save, 'report': report}
