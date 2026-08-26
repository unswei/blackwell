"""Blackwell: JAX-native probabilistic robotics."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("blackwell")
except PackageNotFoundError:  # pragma: no cover - supports direct source imports.
    __version__ = "0.0.0"

__all__ = ["__version__"]
