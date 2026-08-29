"""Blackwell: JAX-native probabilistic robotics."""

from importlib.metadata import PackageNotFoundError, version

from blackwell.beliefs import GaussianBelief, ParticleBelief

try:
    __version__ = version("blackwell")
except PackageNotFoundError:  # pragma: no cover - supports direct source imports.
    __version__ = "0.0.1"

__all__ = ["GaussianBelief", "ParticleBelief", "__version__"]
