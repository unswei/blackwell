"""State-space operations for Euclidean vectors."""

from __future__ import annotations

from jax import Array


def retract(state: Array, tangent: Array) -> Array:
    """Apply a Euclidean tangent displacement to a state."""

    return state + tangent


def local_coordinates(reference: Array, state: Array) -> Array:
    """Express a Euclidean state as a displacement from ``reference``."""

    return state - reference


def transport(reference: Array, target: Array, covariance: Array) -> Array:
    """Transport a Euclidean covariance, which leaves it unchanged."""

    del reference, target
    return covariance
