"""Independent geometry and differentiation checks in both supported precisions."""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from blackwell.spaces import se3


def skew(vector):
    x, y, z = vector
    return jnp.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def matrix(pose):
    """Independent quaternion-to-matrix formula for the reference tests."""
    x, y, z, w = pose[3:] / jnp.linalg.norm(pose[3:])
    rotation = jnp.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    return jnp.eye(4, dtype=pose.dtype).at[:3, :3].set(rotation).at[:3, 3].set(pose[:3])


def finite_difference(function, zero, step):
    basis = jnp.eye(zero.size, dtype=zero.dtype) * step
    return jax.vmap(
        lambda dx: (function(zero + dx) - function(zero - dx)) / (2 * step), out_axes=1
    )(basis)


def test_identity_translation_and_rotation_conventions(precision):
    dtype, tol = precision
    zero = jnp.zeros(6, dtype=dtype)
    identity = jnp.array([0, 0, 0, 0, 0, 0, 1], dtype=dtype)
    assert_allclose(se3.exp(zero), identity, atol=tol)
    assert_allclose(se3.log(identity), zero, atol=tol)
    translation = se3.exp(jnp.array([1, -2, 3, 0, 0, 0], dtype=dtype))
    assert_allclose(translation[:3], [1, -2, 3], atol=tol)
    quarter_turn = jnp.array([1, 2, 3, 0, 0, 2**-0.5, 2**-0.5], dtype=dtype)
    assert_allclose(
        se3.transform_point(quarter_turn, jnp.array([1, 0, 0])), [1, 3, 3], atol=tol
    )
    body_step = se3.retract(quarter_turn, zero.at[0].set(1))
    assert_allclose(body_step[:3], [1, 3, 3], atol=tol)
    assert body_step.dtype == dtype


@pytest.mark.parametrize("angle", [0, 1e-9, 1e-5, 0.099, 0.101, 1.4, 3.13])
def test_exponential_matches_matrix_exponential_and_log_round_trip(precision, angle):
    dtype, tol = precision
    axis = jnp.array([2, -3, 1], dtype=dtype) / jnp.sqrt(14.0)
    tangent = jnp.concatenate((jnp.array([0.7, -1.2, 0.3], dtype=dtype), axis * angle))
    algebra = jnp.zeros((4, 4), dtype=dtype).at[:3, :3].set(skew(tangent[3:]))
    algebra = algebra.at[:3, 3].set(tangent[:3])
    pose = jax.jit(se3.exp)(tangent)
    assert_allclose(matrix(pose), jsp.linalg.expm(algebra), atol=tol, rtol=tol)
    assert_allclose(jax.jit(se3.log)(pose), tangent, atol=tol, rtol=tol)
    assert_allclose(jnp.linalg.norm(pose[3:]), 1, atol=tol)
    assert pose.dtype == dtype


def test_group_operations_match_homogeneous_matrices(precision):
    dtype, tol = precision
    a = se3.exp(jnp.array([1, -2, 0.7, 0.3, -0.8, 0.2], dtype=dtype))
    b = se3.exp(jnp.array([-0.5, 1, 2, -0.4, 0.1, 0.9], dtype=dtype))
    point = jnp.array([2, -1, 4], dtype=dtype)
    assert_allclose(matrix(se3.compose(a, b)), matrix(a) @ matrix(b), atol=tol)
    assert_allclose(matrix(se3.inverse(a)), jnp.linalg.inv(matrix(a)), atol=tol)
    for product in [se3.compose(a, se3.inverse(a)), se3.compose(se3.inverse(a), a)]:
        assert_allclose(matrix(product), jnp.eye(4), atol=tol)
    assert_allclose(
        se3.transform_point(se3.compose(a, b), point),
        se3.transform_point(a, se3.transform_point(b, point)),
        atol=tol,
    )
    assert_allclose(
        se3.transform_point(se3.inverse(a), se3.transform_point(a, point)),
        point,
        atol=tol,
    )
    assert_allclose(
        se3.local_coordinates(a, se3.retract(a, se3.log(b))), se3.log(b), atol=tol
    )


@pytest.mark.parametrize(
    "angle", [0.0, 1e-8, 1.2, np.pi - 1e-4, np.pi + 1e-4, 2 * np.pi, 2 * np.pi + 0.3]
)
def test_principal_logarithm_and_equivalent_signs(precision, angle):
    dtype, tol = precision
    tangent = jnp.array([0.5, -1, 2, angle, 0, 0], dtype=dtype)
    pose = se3.exp(tangent)
    opposite = pose.at[3:].multiply(-1)
    result = se3.log(pose)
    assert jnp.linalg.norm(result[3:]) <= np.pi + tol
    assert_allclose(se3.log(opposite), result, atol=tol)
    assert_allclose(matrix(se3.exp(result)), matrix(pose), atol=tol)
    assert_allclose(se3.local_coordinates(pose, opposite), jnp.zeros(6), atol=tol)
    assert_allclose(se3.adjoint(opposite), se3.adjoint(pose), atol=tol)
    point = jnp.array([1, 2, -3], dtype=dtype)
    assert_allclose(
        se3.transform_point(opposite, point), se3.transform_point(pose, point), atol=tol
    )
    assert_allclose(matrix(se3.inverse(opposite)), matrix(se3.inverse(pose)), atol=tol)
    assert_allclose(
        matrix(se3.compose(pose, opposite)), matrix(se3.compose(pose, pose)), atol=tol
    )


def test_exact_half_turn_tie_break_and_branch_limits(precision):
    dtype, tol = precision
    for vector in [[-1, 0, 0], [0, -1, 0], [0, 0, -1], [-1, 1, 0]]:
        pose = jnp.array([0.2, -0.3, 1, *vector, 0], dtype=dtype)
        tangent = se3.log(pose)
        assert_allclose(jnp.linalg.norm(tangent[3:]), np.pi, atol=tol)
        assert tangent[3 + jnp.argmax(jnp.abs(tangent[3:]))] >= 0
        assert_allclose(se3.log(pose.at[3:].multiply(-1)), tangent, atol=tol)
        assert_allclose(matrix(se3.exp(tangent)), matrix(pose), atol=tol)
    below = se3.log(se3.exp(jnp.array([0, 0, 0, np.pi - 1e-4, 0, 0], dtype=dtype)))
    above = se3.log(se3.exp(jnp.array([0, 0, 0, np.pi + 1e-4, 0, 0], dtype=dtype)))
    assert below[3] > 3 and above[3] < -3


def test_adjoint_conjugation_and_composition(precision):
    dtype, tol = precision
    a = se3.exp(jnp.array([1, -2, 0.5, 0.2, 0.5, -0.8], dtype=dtype))
    delta = jnp.array([0.1, -0.03, 0.04, 0.05, -0.02, 0.03], dtype=dtype)
    b = se3.exp(delta)
    conjugated = se3.compose(se3.compose(a, b), se3.inverse(a))
    assert_allclose(
        matrix(se3.exp(se3.adjoint(a) @ delta)), matrix(conjugated), atol=tol
    )
    assert_allclose(
        se3.adjoint(se3.compose(a, b)), se3.adjoint(a) @ se3.adjoint(b), atol=tol
    )


@pytest.mark.parametrize("angle", [0, 1e-9, 0.099, 0.101, 0.8, np.pi - 1e-3])
def test_forward_reverse_and_finite_difference_jacobians(precision, angle):
    dtype, tol = precision
    tangent = jnp.array([0.2, -0.3, 1, angle, 0, 0], dtype=dtype)

    def function(x):
        return se3.log(se3.exp(x))

    assert_allclose(jax.jit(jax.jacfwd(function))(tangent), jnp.eye(6), atol=tol)
    assert_allclose(jax.jit(jax.jacrev(function))(tangent), jnp.eye(6), atol=tol)
    step = 1e-3 if dtype == jnp.float32 else 1e-5
    fd_tol = 4e-4 if dtype == jnp.float32 else 1e-8
    forward = jax.jacfwd(se3.exp)(tangent)
    assert_allclose(forward, finite_difference(se3.exp, tangent, step), atol=fd_tol)
    assert_allclose(jax.jacrev(se3.exp)(tangent), forward, atol=tol)


def test_jit_vmap_scan_and_covariance_transport(precision):
    dtype, tol = precision
    zero = jnp.zeros(6, dtype=dtype)
    reference = se3.exp(jnp.array([0.5, -1, 0.8, 0.1, -0.3, 0.2], dtype=dtype))
    target = se3.retract(
        reference, jnp.array([0.3, -0.2, 0.1, 0.2, 0.1, -0.4], dtype=dtype)
    )
    factor = jnp.arange(36, dtype=dtype).reshape(6, 6) / 100
    covariance = factor @ factor.T + 0.1 * jnp.eye(6, dtype=dtype)

    def fn(dx):
        return se3.local_coordinates(target, se3.retract(reference, dx))

    step = 1e-3 if dtype == jnp.float32 else 1e-5
    jacobian = finite_difference(fn, zero, step)
    result = jax.jit(se3.transport)(reference, target, covariance)
    assert_allclose(
        result,
        jacobian @ covariance @ jacobian.T,
        atol=2e-4 if dtype == jnp.float32 else 1e-9,
    )
    assert_allclose(result, result.T, atol=tol)
    assert jnp.linalg.eigvalsh(result).min() > 0
    assert_allclose(
        se3.transport(reference, reference, covariance), covariance, atol=tol
    )
    assert_allclose(
        se3.transport(reference, target, jnp.zeros_like(covariance)), 0, atol=tol
    )
    assert jnp.all(
        jnp.isfinite(
            jax.jacrev(
                lambda x: se3.transport(
                    se3.retract(reference, x), reference, covariance
                )
            )(zero)
        )
    )
    tangents = jnp.stack((zero, zero.at[0].set(0.1), zero.at[5].set(0.2)))
    poses = jax.jit(jax.vmap(se3.exp))(tangents)
    assert_allclose(jax.jit(jax.vmap(se3.log))(poses), tangents, atol=tol)

    def step_pose(pose, tangent):
        next_pose = se3.retract(pose, tangent)
        return next_pose, next_pose

    final, history = jax.jit(lambda x, dx: jax.lax.scan(step_pose, x, dx))(
        reference, tangents
    )
    assert history.shape == (3, 7)
    assert_allclose(
        matrix(final), matrix(reference) @ matrix(poses[1]) @ matrix(poses[2]), atol=tol
    )


def test_quaternion_normalisation_and_invalid_zero(precision):
    dtype, tol = precision
    pose = jnp.array([1, 2, 3, 0.1, -0.2, 0.3, 0.9], dtype=dtype)
    scaled = pose.at[3:].multiply(1.01)
    assert_allclose(se3.log(pose), se3.log(scaled), atol=tol)
    assert_allclose(jnp.linalg.norm(se3.inverse(pose)[3:]), 1, atol=tol)
    assert not jnp.all(jnp.isfinite(se3.log(jnp.zeros(7, dtype=dtype))))
