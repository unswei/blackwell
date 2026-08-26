"""Numerical estimation-error and consistency metrics."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def root_mean_square_error(errors: Array, axis: int | tuple[int, ...] = 0) -> Array:
    """Return component-wise RMSE over one or more leading sample axes."""

    return jnp.sqrt(jnp.mean(errors**2, axis=axis))


def position_rmse(errors: Array, axis: int | tuple[int, ...] = 0) -> Array:
    """Return planar position RMSE from tangent errors with ``x, y`` first."""

    squared_position_error = jnp.sum(errors[..., :2] ** 2, axis=-1)
    return jnp.sqrt(jnp.mean(squared_position_error, axis=axis))


def normalised_estimation_error_squared(errors: Array, covariances: Array) -> Array:
    """Return tangent-space NEES for matching errors and covariances.

    ``errors`` has shape ``(..., tangent_dim)`` and ``covariances`` has shape
    ``(..., tangent_dim, tangent_dim)``. Covariances must be nonsingular.
    """

    covariance_inverse_errors = jnp.linalg.solve(covariances, errors[..., None])[
        ..., 0
    ]
    return jnp.sum(errors * covariance_inverse_errors, axis=-1)


def mean_nees(
    errors: Array,
    covariances: Array,
    axis: int | tuple[int, ...] = 0,
) -> Array:
    """Return NEES averaged over one or more leading sample axes."""

    return jnp.mean(normalised_estimation_error_squared(errors, covariances), axis=axis)
