"""Numerical estimation-error and consistency metrics."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def root_mean_square_error(errors: Array, axis: int | tuple[int, ...] = 0) -> Array:
    """Return component-wise root mean square error.

    Args:
        errors: Error array. Remaining axes are preserved after reduction.
        axis: One axis or a tuple of axes over which samples are averaged.

    Returns:
        Square root of the mean squared error over ``axis``.
    """

    return jnp.sqrt(jnp.mean(errors**2, axis=axis))


def position_rmse(errors: Array, axis: int | tuple[int, ...] = 0) -> Array:
    """Return planar position RMSE from errors with position first.

    Args:
        errors: Tangent errors with shape ``(..., tangent_dim)`` and planar
            position in the first two coordinates.
        axis: One axis or a tuple of axes over which samples are averaged.

    Returns:
        Root mean squared Euclidean position norm.
    """

    squared_position_error = jnp.sum(errors[..., :2] ** 2, axis=-1)
    return jnp.sqrt(jnp.mean(squared_position_error, axis=axis))


def normalised_estimation_error_squared(errors: Array, covariances: Array) -> Array:
    """Return tangent-space NEES for matching errors and covariances.

    Args:
        errors: Local errors with shape ``(..., tangent_dim)``.
        covariances: Matching nonsingular covariances with shape
            ``(..., tangent_dim, tangent_dim)``.

    Returns:
        NEES values with shape ``errors.shape[:-1]``.
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
    """Return NEES averaged over one or more sample axes.

    Args:
        errors: Local errors with shape ``(..., tangent_dim)``.
        covariances: Matching nonsingular local covariances.
        axis: One axis or a tuple of axes over which NEES is averaged.

    Returns:
        Mean normalised estimation error squared over ``axis``.
    """

    return jnp.mean(normalised_estimation_error_squared(errors, covariances), axis=axis)
