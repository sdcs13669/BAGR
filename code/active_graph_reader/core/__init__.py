"""Core module — shared interfaces and utilities."""

from .base_reader import BaseReader
from .logging import get_logger, set_verbosity

__all__ = [
    'BaseReader',
    'get_logger',
    'set_verbosity',
]
