"""Teachers: T1 gradient / T2 info-gain / T3 structural / T4 hybrid.

build_teacher(name) resolves teacher names for generate_trajectories.py."""

from .base import Teacher, Trajectory, LEGACY_PICKLE_MODULE  # noqa: F401
from .gradient import GradientTeacher
from .structural import StructuralTeacher
from .hybrid import HybridTeacher
from .info_gain import InfoGainTeacher


def build_teacher(name: str, classifier=None, subgraph_builder=None,
                  device: str = 'cpu') -> Teacher:
    """Teacher name -> Teacher instance; info-gain needs classifier/builder/device."""
    if name == 'gradient':
        return GradientTeacher()
    if name.startswith('structural-'):
        return StructuralTeacher(method=name.split('-', 1)[1])
    if name.startswith('hybrid-a'):
        return HybridTeacher(alpha=float(name.split('-')[1].replace('a', '')))
    if name == 'info-gain':
        if classifier is None or subgraph_builder is None:
            raise ValueError("info-gain teacher requires classifier and subgraph_builder")
        return InfoGainTeacher(classifier, subgraph_builder, device=device)
    raise ValueError(f"unknown teacher '{name}', available: gradient, structural-degree, "
                     "structural-pagerank, info-gain, hybrid-a{α}")
