"""Compat shim: GINEEncoder/GINELayer live in the classifier package.

Prefer from classifier.layers.gin import ...; this shim keeps the old import
path working."""

from classifier.layers.gin import GINEEncoder, GINELayer  # noqa: F401
