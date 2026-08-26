"""A private SE(2) range-bearing EKF localisation experiment.

The belief covariance is expressed in the local tangent coordinates of its
nominal pose.  This keeps the experiment close to the manifold-aware interface
that Blackwell will expose publicly once the experiments have settled.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array


class SE2Belief(NamedTuple):
    """A Gaussian pose belief with covariance in local SE(2) coordinates."""

    pose: Array
    covariance: Array


class RangeBearingLocalisationModel(NamedTuple):
    """Known-landmark range-bearing localisation model.

    Motion controls are body-frame SE(2) tangent increments.  Every landmark
    supplies one range and one bearing observation with an independently
    repeated ``measurement_covariance``.
    """

    landmarks: Array
    process_covariance: Array
    measurement_covariance: Array


def compose(left: Array, right: Array) -> Array:
    """Compose two SE(2) poses represented as ``[x, y, heading]``."""

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
    return jnp.concatenate((translation, jnp.atleast_1d(wrap_angle(-pose[2]))))


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
    return jnp.concatenate((translation, jnp.atleast_1d(wrap_angle(angle))))


def log(pose: Array) -> Array:
    """Map an SE(2) pose to its tangent vector."""

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
    return jnp.concatenate((translation, jnp.atleast_1d(wrap_angle(pose[2]))))


def retract(pose: Array, tangent: Array) -> Array:
    """Apply a local tangent displacement to an SE(2) pose."""

    return compose(pose, exp(tangent))


def local_coordinates(reference: Array, pose: Array) -> Array:
    """Express ``pose`` as a tangent displacement from ``reference``."""

    return log(compose(inverse(reference), pose))


def range_bearing(pose: Array, landmarks: Array) -> Array:
    """Predict ``[range, bearing]`` observations for known world landmarks."""

    delta = landmarks - pose[:2]
    cosine = jnp.cos(pose[2])
    sine = jnp.sin(pose[2])
    local_x = cosine * delta[:, 0] + sine * delta[:, 1]
    local_y = -sine * delta[:, 0] + cosine * delta[:, 1]
    return jnp.stack(
        (jnp.hypot(local_x, local_y), wrap_angle(jnp.arctan2(local_y, local_x))),
        axis=-1,
    )


def range_bearing_jacobian(pose: Array, landmarks: Array) -> Array:
    """Return observation Jacobians with respect to local pose tangents.

    The output has shape ``(landmark_count, 2, 3)`` and is defined away from
    landmarks coincident with the robot pose, where range-bearing observations
    themselves are singular.
    """

    delta = landmarks - pose[:2]
    cosine = jnp.cos(pose[2])
    sine = jnp.sin(pose[2])
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


def predict(
    belief: SE2Belief,
    model: RangeBearingLocalisationModel,
    control: Array,
) -> SE2Belief:
    """Apply the motion update to an SE(2) tangent-space Gaussian belief."""

    control_pose = exp(control)
    pose = compose(belief.pose, control_pose)
    transition = adjoint(inverse(control_pose))
    covariance = (
        transition @ belief.covariance @ transition.T + model.process_covariance
    )
    return SE2Belief(pose=pose, covariance=_symmetrise(covariance))


def update(
    belief: SE2Belief,
    model: RangeBearingLocalisationModel,
    measurements: Array,
) -> SE2Belief:
    """Apply a range-bearing EKF update with wrapped bearing innovation."""

    expected_measurements = range_bearing(belief.pose, model.landmarks)
    innovation = measurements - expected_measurements
    innovation = innovation.at[:, 1].set(wrap_angle(innovation[:, 1]))
    observation = range_bearing_jacobian(belief.pose, model.landmarks).reshape(-1, 3)
    measurement_covariance = jnp.kron(
        jnp.eye(model.landmarks.shape[0], dtype=belief.covariance.dtype),
        model.measurement_covariance,
    )
    innovation_covariance = (
        observation @ belief.covariance @ observation.T + measurement_covariance
    )
    gain = jnp.linalg.solve(innovation_covariance, observation @ belief.covariance).T

    pose = retract(belief.pose, gain @ innovation.reshape(-1))
    identity = jnp.eye(3, dtype=belief.covariance.dtype)
    residual = identity - gain @ observation
    covariance = (
        residual @ belief.covariance @ residual.T
        + gain @ measurement_covariance @ gain.T
    )
    return SE2Belief(pose=pose, covariance=_symmetrise(covariance))


def step(
    belief: SE2Belief,
    model: RangeBearingLocalisationModel,
    control: Array,
    measurements: Array,
) -> SE2Belief:
    """Perform one motion prediction followed by a range-bearing EKF update."""

    return update(predict(belief, model, control), model, measurements)


def adjoint(pose: Array) -> Array:
    """Return the SE(2) adjoint matrix for local tangent coordinates."""

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
    """Evaluate the SE(2) exponential-map coefficients stably near zero."""

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


def _symmetrise(matrix: Array) -> Array:
    """Limit numerical skew in a covariance matrix."""

    return (matrix + matrix.T) / 2
