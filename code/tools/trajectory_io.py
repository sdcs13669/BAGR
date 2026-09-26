"""Trajectory file IO: pickle-compatible loading, directory merging, saving.

Legacy trajectory .pt files pickle-referenced m4.teachers.base.Trajectory;
a custom Unpickler remaps that path to trajectories.core.trajectory.Trajectory
so old files load unchanged under the new layout."""

import pickle
import re
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import torch

LEGACY_PICKLE_MODULE = 'm4.teachers.base'


def _trajectory_cls():
    """Import Trajectory lazily (avoids a tools <-> trajectories cycle)."""
    from trajectories.core.trajectory import Trajectory
    return Trajectory


class _TrajectoryUnpickler(pickle.Unpickler):
    """Remap the legacy m4.teachers.base path to the new Trajectory."""

    def find_class(self, module, name):
        if name == 'Trajectory' and module == LEGACY_PICKLE_MODULE:
            return _trajectory_cls()
        return super().find_class(module, name)


class _compat_pickle_module:
    """Minimal interface required by torch.load(pickle_module=...)."""

    Unpickler = _TrajectoryUnpickler


def load_trajectory_file(path) -> Dict:
    """Load one trajectory .pt (old and new pickle paths both work)."""
    return torch.load(str(path), map_location='cpu', weights_only=False,
                      pickle_module=_compat_pickle_module)


def normalize_trajectory(t):
    """Coerce a trajectory (new/old instance or dict) into the new Trajectory."""
    Trajectory = _trajectory_cls()
    if isinstance(t, Trajectory):
        return t
    return Trajectory(
        graph_idx=int(t.graph_idx),
        seed_node=int(t.seed_node),
        reveal_ratio=float(t.reveal_ratio),
        reveal_order=[int(v) for v in t.reveal_order],
        final_pred=int(t.final_pred),
        is_correct=bool(t.is_correct),
        teacher_type=str(t.teacher_type),
        ce=float(getattr(t, 'ce', 0.0) or 0.0),
    )


def ratio_key(ratio: float) -> str:
    """0.05 -> 0_050 (filename key)."""
    return f'0_{round(ratio * 1000):03d}'


def ratio_key_from_name(name: str) -> float:
    """0_050 -> 0.05 (inverse of ratio_key)."""
    m = re.search(r'0_(\d{3})', name)
    if not m:
        raise ValueError(f"cannot parse the budget ratio from filename: {name}")
    return int(m.group(1)) / 1000.0


def save_trajectory_file(path, trajs_by_teacher: Dict[str, list],
                         config: Dict) -> None:
    """Save a trajectory file, keeping the per-teacher/per-graph structure.

    trajs_by_teacher: {teacher_name: [Trajectory, ...]}; stored grouped by
    graph_idx as {teacher: [[t, ...] per graph]}.
    """
    Trajectory = _trajectory_cls()
    grouped: Dict[str, Dict[int, List]] = {}
    for teacher, trajs in trajs_by_teacher.items():
        per_graph: Dict[int, List[Trajectory]] = {}
        for t in trajs:
            per_graph.setdefault(t.graph_idx, []).append(normalize_trajectory(t))
        grouped[teacher] = [per_graph[k] for k in sorted(per_graph)]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'trajectories': grouped, 'config': config}, str(path))


def iter_file_trajectories(data: Dict) -> Iterator:
    """Iterate over all trajectories in a file (flattened)."""
    for per_graph in data.get('trajectories', {}).values():
        for trajs in per_graph:
            for t in trajs:
                yield t


def merge_trajectory_dir(data_dir, pattern: str = 'trajectories_*.pt',
                         verbose: bool = True) -> List:
    """Load every trajectory file in a directory and flatten.

    No hardcoded teacher list or budget tiers: whatever is in the directory
    gets merged.
    """
    Trajectory = _trajectory_cls()
    data_dir = Path(data_dir)
    all_trajs: List = []
    for fpath in sorted(data_dir.glob(pattern)):
        data = load_trajectory_file(fpath)
        n_before = len(all_trajs)
        for t in iter_file_trajectories(data):
            all_trajs.append(normalize_trajectory(t))
        if verbose:
            print(f"  {fpath.name}: +{len(all_trajs) - n_before} trajectories")
    if verbose:
        print(f"Total merged: {len(all_trajs)} trajectories")
    return all_trajs
