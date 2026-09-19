"""Tangent Jacobians, correlated covariance and the runnable public example."""

import runpy
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from blackwell import GaussianBelief
from blackwell.spaces import se3


@pytest.fixture(scope="module")
def example():
    return runpy.run_path(
        str(Path(__file__).parents[2] / "examples/se3_uncertainty.py")
    )


def test_tangent_jacobians_match_analytical_results_and_finite_differences(precision):
    dtype, _ = precision
    a = se3.exp(jnp.array([1, -2, 0.5, 0.3, -0.7, 0.4], dtype=dtype))
    b = se3.exp(jnp.array([-0.4, 1, 2, -0.1, 0.2, 0.8], dtype=dtype))
    p = jnp.array([2, -1, 3], dtype=dtype)
    composed = se3.compose(a, b)
    inverted = se3.inverse(a)
    rotation = se3.adjoint(a)[:3, :3]
    x, y, z = p
    skew = jnp.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=dtype)
    cases = [
        (
            lambda d: se3.local_coordinates(
                composed, se3.compose(se3.retract(a, d[:6]), se3.retract(b, d[6:]))
            ),
            jnp.concatenate((se3.adjoint(se3.inverse(b)), jnp.eye(6)), axis=1),
        ),
        (
            lambda d: se3.local_coordinates(inverted, se3.inverse(se3.retract(a, d))),
            -se3.adjoint(a),
        ),
        (
            lambda d: se3.transform_point(se3.retract(a, d[:6]), p + d[6:]),
            jnp.concatenate((rotation, -rotation @ skew, rotation), axis=1),
        ),
    ]
    for function, expected in cases:
        zero = jnp.zeros(expected.shape[1], dtype=dtype)
        tolerance = 3e-6 if dtype == jnp.float32 else 2e-10
        forward = jax.jit(jax.jacfwd(function))(zero)
        reverse = jax.jit(jax.jacrev(function))(zero)
        assert_allclose(forward, expected, atol=tolerance)
        assert_allclose(reverse, expected, atol=tolerance)
        step = 1e-3 if dtype == jnp.float32 else 1e-5
        differences = jax.vmap(
            lambda dx, fn=function, h=step: (fn(dx) - fn(-dx)) / (2 * h), out_axes=1
        )(jnp.eye(zero.size, dtype=dtype) * step)
        assert_allclose(
            forward, differences, atol=4e-4 if dtype == jnp.float32 else 1e-9
        )


def test_correlated_composition_and_inversion_covariances(example):
    a = se3.exp(jnp.array([1, 2, -1, 0.3, -0.2, 0.6]))
    b = se3.exp(jnp.array([-0.5, 1, 2, -0.2, 0.4, -0.3]))
    factor = jax.random.normal(jax.random.key(19), (12, 12)) * 0.01
    joint = factor @ factor.T
    cross = joint[:6, 6:]
    left = GaussianBelief(a, joint[:6, :6])
    right = GaussianBelief(b, joint[6:, 6:])
    result = jax.jit(example["compose_beliefs"])(left, right, cross)
    jacobian = se3.adjoint(se3.inverse(b))
    expected = (
        jacobian @ left.covariance @ jacobian.T
        + right.covariance
        + jacobian @ cross
        + cross.T @ jacobian.T
    )
    assert_allclose(result.covariance, expected, atol=1e-8)
    independent = example["compose_beliefs"](left, right, jnp.zeros((6, 6)))
    assert jnp.linalg.norm(result.covariance - independent.covariance) > 1e-4
    inverted = jax.jit(example["invert_belief"])(result)
    adjoint = se3.adjoint(result.mean)
    assert_allclose(inverted.covariance, adjoint @ expected @ adjoint.T, atol=1e-8)
    for belief in [result, inverted]:
        assert belief.mean.shape == (7,)
        assert belief.covariance.shape == (6, 6)
        assert_allclose(belief.covariance, belief.covariance.T, atol=1e-8)
        assert jnp.linalg.eigvalsh(belief.covariance).min() > 0


def test_perfect_correlation_cancels_pose_and_its_inverse(example):
    pose = se3.exp(jnp.array([1, -2, 0.4, 0.3, -0.5, 0.2]))
    covariance = jnp.eye(6) * 0.01
    inverse_jacobian = -se3.adjoint(pose)
    left = GaussianBelief(pose, covariance)
    right = example["invert_belief"](left)
    cross = covariance @ inverse_jacobian.T
    cancelled = example["compose_beliefs"](left, right, cross)
    assert_allclose(se3.log(cancelled.mean), jnp.zeros(6), atol=1e-6)
    assert_allclose(cancelled.covariance, jnp.zeros((6, 6)), atol=1e-8)
    independent = example["compose_beliefs"](left, right, jnp.zeros((6, 6)))
    assert jnp.trace(independent.covariance) > 0.1


def test_correlated_uncertain_points_match_monte_carlo(example):
    pose, point, joint = example["point_scenario"]()
    predicted = jax.jit(example["transform_point_beliefs"])(pose, point, joint[:6, 6:])
    cloud = jax.jit(
        example["sample_transformed_points"], static_argnames="sample_count"
    )(jax.random.key(2026), pose, point, joint, sample_count=50_000)
    empirical = example["empirical_gaussian"](cloud)
    error = jnp.linalg.norm(predicted.covariance - empirical.covariance)
    assert error / jnp.linalg.norm(predicted.covariance) < 0.035
    # Mean error includes sampling uncertainty and the small second-order bias.
    assert_allclose(predicted.mean, empirical.mean, atol=8e-4)
    independent = example["transform_point_beliefs"](pose, point, jnp.zeros((6, 3)))
    omitted_error = jnp.linalg.norm(independent.covariance - empirical.covariance)
    assert omitted_error > 5 * error
    assert predicted.mean.shape == (3,)
    assert cloud.particles.shape == (50_000, 3)
    assert_allclose(jnp.sum(cloud.weights), 1, atol=1e-6)
    assert_allclose(predicted.covariance, predicted.covariance.T, atol=1e-8)
    assert jnp.linalg.eigvalsh(predicted.covariance).min() > 0
