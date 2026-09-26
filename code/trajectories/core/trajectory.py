"""Trajectory dataclass and Teacher ABC (defined in teachers/base.py;
re-exported here to avoid a teachers.base -> core -> generator ->
teachers.gradient -> teachers.base import cycle)."""

from ..teachers.base import Teacher, Trajectory, LEGACY_PICKLE_MODULE  # noqa: F401
