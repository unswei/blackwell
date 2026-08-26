"""Known-landmark range-bearing observation models on SE(2)."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from blackwell.spaces import se2 as space


class KnownLandmarksRangeBearing(NamedTuple):
    """Range-bearing observations of fixed two-dimensional world landmarks."""

    landmarks: Array
    measurement_covariance: Array


def observe(state: Array, model: KnownLandmarksRangeBearing) -> Array:
    """Predict ``[range, bearing]`` observations at every known landmark."""

    delta = model.landmarks - state[:2]
    cosine = jnp.cos(state[2])
    sine = jnp.sin(state[2])
    local_x = cosine * delta[:, 0] + sine * delta[:, 1]
    local_y = -sine * delta[:, 0] + cosine * delta[:, 1]
    return jnp.stack(
        (jnp.hypot(local_x, local_y), space.wrap_angle(jnp.arctan2(local_y, local_x))),
        axis=-1,
    )


def observation_jacobian(state: Array, model: KnownLandmarksRangeBearing) -> Array:
    """Return local-tangent Jacobians with shape ``(landmark_count, 2, 3)``.

    The range-bearing observation is singular at a landmark coincident with the
    robot pose, so that configuration is intentionally not defined.
    """

    delta = model.landmarks - state[:2]
    cosine = jnp.cos(state[2])
    sine = jnp.sin(state[2])
    local_x = cosine * delta[:, 0] + sine * delta[:, 1]
    local_y = -sine * delta[:, 0] + cosine * delta[:, 1]
    squared_range = local_x**2 + local_y**2
    range_value = jnp.sqrt(squared_range)
    range_row = jnp.stack(
        (-local_x / range_value, -local_y / range_value, jnp.zeros_like(range_value)),
        axis=-1,
    )
    bearing_row = jnp.stack(
        (
            local_y / squared_range,
            -local_x / squared_range,
            -jnp.ones_like(range_value),
        ),
        axis=-1,
    )
    return jnp.stack((range_row, bearing_row), axis=1)


def measurement_covariance(
    state: Array, model: KnownLandmarksRangeBearing
) -> Array:
    """Return the block-diagonal covariance for all landmark observations."""

    del state
    landmark_count = model.landmarks.shape[0]
    return jnp.kron(
        jnp.eye(landmark_count, dtype=model.measurement_covariance.dtype),
        model.measurement_covariance,
    )


def measurement_residual(
    measurement: Array, expected: Array, model: KnownLandmarksRangeBearing
) -> Array:
    """Return residuals with bearings wrapped across the principal-angle cut."""

    del model
    residual = measurement - expected
    return residual.at[:, 1].set(space.wrap_angle(residual[:, 1]))
