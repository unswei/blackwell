"""SE(2) poses and local tangent-coordinate operations."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array


def compose(left: Array, right: Array) -> Array:
    """Compose SE(2) poses represented as ``[x, y, heading]``."""

    cosine = jnp.cos(left[2])
    sine = jnp.sin(left[2])
    translation = left[:2] + jnp.array(
        [
            cosine * right[0] - sine * right[1],
            sine * right[0] + cosine * right[1],
        ]
    )
    heading = jnp.atleast_1d(wrap_angle(left[2] + right[2]))
    return jnp.concatenate((translation, heading))


def inverse(pose: Array) -> Array:
    """Return the inverse of an SE(2) pose."""

    cosine = jnp.cos(pose[2])
    sine = jnp.sin(pose[2])
    translation = jnp.array(
        [
            -cosine * pose[0] - sine * pose[1],
            sine * pose[0] - cosine * pose[1],
        ]
    )
    heading = jnp.atleast_1d(wrap_angle(-pose[2]))
    return jnp.concatenate((translation, heading))


def exp(tangent: Array) -> Array:
    """Map an SE(2) tangent vector to a pose."""

    angle = tangent[2]
    sine_over_angle, one_minus_cosine_over_angle = _rotation_coefficients(angle)
    translation = jnp.array(
        [
            sine_over_angle * tangent[0]
            - one_minus_cosine_over_angle * tangent[1],
            one_minus_cosine_over_angle * tangent[0]
            + sine_over_angle * tangent[1],
        ]
    )
    heading = jnp.atleast_1d(wrap_angle(angle))
    return jnp.concatenate((translation, heading))


def log(pose: Array) -> Array:
    """Map an SE(2) pose to its principal tangent vector."""

    sine_over_angle, one_minus_cosine_over_angle = _rotation_coefficients(pose[2])
    rotation_scale = sine_over_angle**2 + one_minus_cosine_over_angle**2
    translation = jnp.array(
        [
            (
                sine_over_angle * pose[0]
                + one_minus_cosine_over_angle * pose[1]
            )
            / rotation_scale,
            (
                -one_minus_cosine_over_angle * pose[0]
                + sine_over_angle * pose[1]
            )
            / rotation_scale,
        ]
    )
    heading = jnp.atleast_1d(wrap_angle(pose[2]))
    return jnp.concatenate((translation, heading))


def retract(pose: Array, tangent: Array) -> Array:
    """Apply a right, body-frame tangent displacement to a pose."""

    return compose(pose, exp(tangent))


def local_coordinates(reference: Array, pose: Array) -> Array:
    """Express ``pose`` in the body-frame tangent coordinates at ``reference``."""

    return log(compose(inverse(reference), pose))


def transport(reference: Array, target: Array, covariance: Array) -> Array:
    """Transport covariance from ``reference`` to ``target`` tangent coordinates.

    The map is the exact first-order Jacobian of re-expressing perturbations at
    ``reference`` around ``target``. It is valid away from the SE(2) logarithm's
    unavoidable principal-angle branch cut.
    """

    def reexpress(tangent: Array) -> Array:
        return local_coordinates(target, retract(reference, tangent))

    jacobian = jax.jacfwd(reexpress)(jnp.zeros(3, dtype=reference.dtype))
    transported = jacobian @ covariance @ jacobian.T
    return (transported + transported.T) / 2


def adjoint(pose: Array) -> Array:
    """Return the SE(2) adjoint matrix for body-frame tangent coordinates."""

    cosine = jnp.cos(pose[2])
    sine = jnp.sin(pose[2])
    return jnp.array(
        [
            [cosine, -sine, pose[1]],
            [sine, cosine, -pose[0]],
            [0.0, 0.0, 1.0],
        ],
        dtype=pose.dtype,
    )


def wrap_angle(angle: Array) -> Array:
    """Wrap an angle to the principal interval ``[-pi, pi]``."""

    return jnp.arctan2(jnp.sin(angle), jnp.cos(angle))


def _rotation_coefficients(angle: Array) -> tuple[Array, Array]:
    """Evaluate SE(2) exponential-map coefficients stably near zero."""

    small_angle = jnp.abs(angle) < 1e-4
    safe_angle = jnp.where(small_angle, 1.0, angle)
    sine_over_angle = jnp.where(
        small_angle,
        1.0 - angle**2 / 6.0 + angle**4 / 120.0,
        jnp.sin(safe_angle) / safe_angle,
    )
    one_minus_cosine_over_angle = jnp.where(
        small_angle,
        angle / 2.0 - angle**3 / 24.0 + angle**5 / 720.0,
        (1.0 - jnp.cos(safe_angle)) / safe_angle,
    )
    return sine_over_angle, one_minus_cosine_over_angle
