"""State-space operations for Euclidean vectors."""

from __future__ import annotations

from jax import Array


def retract(state: Array, tangent: Array) -> Array:
    """Apply a Euclidean tangent displacement to a state.

    Args:
        state: Reference vector with shape ``(state_dim,)``.
        tangent: Displacement vector with the same shape.

    Returns:
        ``state + tangent``.
    """

    return state + tangent


def local_coordinates(reference: Array, state: Array) -> Array:
    """Express a Euclidean state as a displacement from ``reference``.

    Args:
        reference: Reference vector with shape ``(state_dim,)``.
        state: Target vector with the same shape.

    Returns:
        ``state - reference``.
    """

    return state - reference


def transport(reference: Array, target: Array, covariance: Array) -> Array:
    """Transport a Euclidean covariance, which leaves it unchanged.

    Args:
        reference: Source mean; unused in Euclidean coordinates.
        target: Target mean; unused in Euclidean coordinates.
        covariance: Covariance with shape ``(state_dim, state_dim)``.

    Returns:
        The input ``covariance`` unchanged.
    """

    del reference, target
    return covariance
