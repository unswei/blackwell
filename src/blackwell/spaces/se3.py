"""SE(3) poses with right, body-frame tangent coordinates.

States are ``[x, y, z, qx, qy, qz, qw]`` (shape ``(7,)``), using Hamilton
quaternions in scalar-last ``xyzw`` order. A pose maps a body-frame point to
the parent frame as ``R(q) @ point + translation``. Rotations are active and
right-handed. Tangents are ``[rho_x, rho_y, rho_z, phi_x, phi_y, phi_z]``
(shape ``(6,)``): body translation first, then a rotation vector in radians.

Inputs must be finite floating-point arrays with non-zero quaternions; each
operation normalises quaternions to tolerate numerical drift. A zero quaternion
is invalid and is not repaired. Quaternion signs are interchangeable, and pose
outputs are not sign-canonicalised. Only the logarithm selects a principal
rotation. Kernels operate on one pose/point; use :func:`jax.vmap` for batches.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array


def compose(left: Array, right: Array) -> Array:
    """Compose poses, applying ``right`` before ``left``.

    Args:
        left: Parent-from-intermediate pose, shape ``(7,)``.
        right: Intermediate-from-body pose, shape ``(7,)``.

    Returns:
        Parent-from-body pose, shape ``(7,)``, with a unit quaternion.
    """

    left_q = _normalise(left[3:])
    right_q = _normalise(right[3:])
    translation = left[:3] + _rotate(left_q, right[:3])
    quaternion = jnp.concatenate(
        (
            left_q[3] * right_q[:3]
            + right_q[3] * left_q[:3]
            + jnp.cross(left_q[:3], right_q[:3]),
            jnp.atleast_1d(left_q[3] * right_q[3] - left_q[:3] @ right_q[:3]),
        )
    )
    return jnp.concatenate((translation, _normalise(quaternion)))


def inverse(pose: Array) -> Array:
    """Return the inverse frame map of a pose with shape ``(7,)``.

    The returned pose has translation ``-R.T @ t`` and the conjugate unit
    quaternion, also in ``xyzw`` order.
    """

    quaternion = _normalise(pose[3:]) * jnp.array(
        [-1.0, -1.0, -1.0, 1.0], dtype=pose.dtype
    )
    return jnp.concatenate((-_rotate(quaternion, pose[:3]), quaternion))


def transform_point(pose: Array, point: Array) -> Array:
    """Map one body-frame point into the pose's parent frame.

    Args:
        pose: Parent-from-body pose with shape ``(7,)``.
        point: Body-frame Cartesian point with shape ``(3,)``.

    Returns:
        ``R(q) @ point + translation``, shape ``(3,)``.
    """

    return _rotate(_normalise(pose[3:]), point) + pose[:3]


def exp(tangent: Array) -> Array:
    """Map ``[rho, phi]`` (shape ``(6,)``) to an SE(3) pose.

    Translation is ``V(phi) @ rho``, where ``V`` is the SO(3) left Jacobian;
    it is generally different from ``rho``. The quaternion represents the
    right-handed rotation vector ``phi``. Series expansions give finite
    forward and reverse derivatives at zero, including pure translations.

    Returns:
        Pose with shape ``(7,)`` and a unit ``xyzw`` quaternion. Rotation
        vectors outside the principal ball are allowed, but ``log(exp(x))``
        only recovers ``x`` when ``norm(phi) < pi``.
    """

    rho, phi = tangent[:3], tangent[3:]
    squared_angle = phi @ phi
    small = squared_angle < 0.01
    # Mask before sqrt/division: inactive branches must also differentiate
    # safely under both jacrev and batched (vmap) evaluation.
    angle = jnp.sqrt(jnp.where(small, 1.0, squared_angle))
    half_sinc = jnp.where(
        small,
        0.5 - squared_angle / 48 + squared_angle**2 / 3840 - squared_angle**3 / 645120,
        jnp.sin(angle / 2) / angle,
    )
    cosine = jnp.where(
        small,
        1 - squared_angle / 8 + squared_angle**2 / 384 - squared_angle**3 / 46080,
        jnp.cos(angle / 2),
    )
    a = jnp.where(
        small,
        0.5 - squared_angle / 24 + squared_angle**2 / 720 - squared_angle**3 / 40320,
        2 * (jnp.sin(angle / 2) / angle) ** 2,
    )
    b = jnp.where(
        small,
        1 / 6
        - squared_angle / 120
        + squared_angle**2 / 5040
        - squared_angle**3 / 362880,
        (angle - jnp.sin(angle)) / angle**3,
    )
    cross = jnp.cross(phi, rho)
    translation = rho + a * cross + b * jnp.cross(phi, cross)
    quaternion = jnp.concatenate((half_sinc * phi, jnp.atleast_1d(cosine)))
    return jnp.concatenate((translation, _normalise(quaternion)))


def log(pose: Array) -> Array:
    """Return the principal tangent ``[rho, phi]``, shape ``(6,)``.

    The rotation norm lies in ``[0, pi]``. Equivalent quaternion signs give
    the same result. At exactly ``qw == 0``, the largest-magnitude vector
    component is chosen non-negative (ties use the first component). The
    principal logarithm is necessarily discontinuous at rotations of pi;
    derivatives and Gaussian linearisation are not valid on that cut.

    Translation is recovered using ``V(phi)^{-1}``, including its coupling
    to rotation. Series expansions preserve derivatives at the identity.

    Args:
        pose: Pose with shape ``(7,)`` and a non-zero quaternion.
    """

    quaternion = _normalise(pose[3:])
    vector, scalar = quaternion[:3], quaternion[3]
    tie_component = vector[jnp.argmax(jnp.abs(vector))]
    flip = (scalar < 0) | ((scalar == 0) & (tie_component < 0))
    quaternion = jnp.where(flip, -quaternion, quaternion)
    vector, scalar = quaternion[:3], quaternion[3]
    squared_sine = vector @ vector
    small = squared_sine < 0.0025
    sine = jnp.sqrt(jnp.where(small, 1.0, squared_sine))
    scale = jnp.where(
        small,
        2 + squared_sine / 3 + 3 * squared_sine**2 / 20 + 5 * squared_sine**3 / 56,
        2 * jnp.arctan2(sine, scalar) / sine,
    )
    phi = scale * vector
    squared_angle = phi @ phi
    small_angle = squared_angle < 0.01
    angle = jnp.sqrt(jnp.where(small_angle, 1.0, squared_angle))
    coefficient = jnp.where(
        small_angle,
        1 / 12
        + squared_angle / 720
        + squared_angle**2 / 30240
        + squared_angle**3 / 1209600,
        (1 - (angle / 2) / jnp.tan(angle / 2)) / angle**2,
    )
    cross = jnp.cross(phi, pose[:3])
    rho = pose[:3] - cross / 2 + coefficient * jnp.cross(phi, cross)
    return jnp.concatenate((rho, phi))


def retract(pose: Array, tangent: Array) -> Array:
    """Apply a right/body displacement: ``compose(pose, exp(tangent))``.

    Args:
        pose: Reference pose, shape ``(7,)``.
        tangent: Body displacement ``[rho, phi]``, shape ``(6,)``.

    Returns:
        Displaced pose with shape ``(7,)``.
    """

    return compose(pose, exp(tangent))


def local_coordinates(reference: Array, pose: Array) -> Array:
    """Return ``log(compose(inverse(reference), pose))``.

    Args:
        reference: Reference pose with shape ``(7,)``.
        pose: Target pose with shape ``(7,)``.

    Returns:
        Right/body tangent at ``reference``, shape ``(6,)``. Retraction
        recovers the target transform, possibly with the opposite quaternion
        sign. The principal-logarithm branch cut applies to relative rotation.
    """

    return log(compose(inverse(reference), pose))


def adjoint(pose: Array) -> Array:
    """Return ``Ad_T = [[R, skew(t) @ R], [0, R]]``, shape ``(6, 6)``.

    This maps body/right tangents to parent/left tangents, with translation
    before rotation, and satisfies ``T Exp(xi) T^-1 = Exp(Ad_T @ xi)``.
    To change perturbation convention, use ``Ad_T @ P @ Ad_T.T`` for a
    covariance. To re-express a distribution about a different mean, use
    :func:`transport` instead.

    Args:
        pose: Parent-from-body pose with shape ``(7,)``.
    """

    rotation = jax.vmap(_rotate, in_axes=(None, 1), out_axes=1)(
        _normalise(pose[3:]), jnp.eye(3, dtype=pose.dtype)
    )
    translation_cross = jax.vmap(
        lambda column: jnp.cross(pose[:3], column), in_axes=1, out_axes=1
    )(rotation)
    return jnp.concatenate(
        (
            jnp.concatenate((rotation, translation_cross), axis=1),
            jnp.concatenate((jnp.zeros_like(rotation), rotation), axis=1),
        )
    )


def transport(reference: Array, target: Array, covariance: Array) -> Array:
    """Re-express covariance from ``reference`` about ``target`` to first order.

    Uses the Jacobian at zero of
    ``local_coordinates(target, retract(reference, delta))``. This follows
    Blackwell's state-space contract and is not simply an adjoint frame change.
    It is valid away from relative rotations of pi. The chart displacement at
    zero need not vanish; transporting covariance alone does not move a mean.

    Args:
        reference: Mean where the input covariance lives, shape ``(7,)``.
        target: Mean defining the new local coordinates, shape ``(7,)``.
        covariance: Body tangent covariance with shape ``(6, 6)``.

    Returns:
        Symmetric ``(6, 6)`` first-order covariance in the target chart.
    """

    jacobian = jax.jacfwd(
        lambda delta: local_coordinates(target, retract(reference, delta))
    )(jnp.zeros(6, dtype=reference.dtype))
    result = jacobian @ covariance @ jacobian.T
    return (result + result.T) / 2


def _normalise(quaternion: Array) -> Array:
    return quaternion / jnp.linalg.norm(quaternion)


def _rotate(quaternion: Array, point: Array) -> Array:
    cross = 2 * jnp.cross(quaternion[:3], point)
    return point + quaternion[3] * cross + jnp.cross(quaternion[:3], cross)
