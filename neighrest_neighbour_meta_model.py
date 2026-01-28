"""Compatibility shim: re-export package module for legacy top-level imports.

Tests expect to import `neighrest_neighbour_meta_model` from the top-level. The
project's implementation lives in the `surroptim` package; re-export the
public API here so tests continue to work.
"""
from surroptim.neighrest_neighbour_meta_model import *  # noqa: F401,F403

try:
    __all__ = __import__("surroptim.neighrest_neighbour_meta_model", fromlist=["*"]).__all__
except Exception:
    __all__ = None
