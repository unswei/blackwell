"""Tests for public generic extended Kalman filtering."""

import jax
import jax.numpy as jnp

from blackwell.beliefs import GaussianBelief
from blackwell.filters.ekf import ExtendedKalmanFilter
from blackwell.models import linear, range_bearing
from blackwell.models import se2 as se2_models
from blackwell.spaces import euclidean, se2


def test_ekf_matches_the_closed_form_linear_gaussian_update() -> None:
    filter_ = ExtendedKalmanFilter(euclidean, linear, linear)
    dynamics = linear.LinearDynamics(
        transition=jnp.eye(2),
        control=jnp.zeros((2, 0)),
        process_covariance=jnp.zeros((2, 2)),
    )
    observation = linear.LinearObservation(
        observation=jnp.array([[1.0, 0.0]]),
        measurement_covariance=jnp.array([[2.0]]),
    )
    belief = GaussianBelief(jnp.array([0.0, 0.0]), jnp.diag(jnp.array([2.0, 1.0])))

    result = filter_.step(
        belief,
        dynamics,
        observation,
        jnp.empty(0),
        jnp.array([4.0]),
    )

    assert jnp.allclose(result.mean, jnp.array([2.0, 0.0]))
    assert jnp.allclose(result.covariance, jnp.diag(jnp.array([1.0, 1.0])))


def test_se2_ekf_is_jittable_with_public_models_and_wrapped_bearings() -> None:
    filter_ = ExtendedKalmanFilter(se2, se2_models, range_bearing)
    dynamics = se2_models.BodyMotion(jnp.eye(3) * 0.01)
    observation = range_bearing.KnownLandmarksRangeBearing(
        landmarks=jnp.array([[5.0, 0.0], [0.0, 5.0]]),
        measurement_covariance=jnp.diag(jnp.array([0.1, 0.02])),
    )
    belief = GaussianBelief(
        jnp.array([0.0, 0.0, -jnp.pi + 0.01]),
        jnp.diag(jnp.array([1.0, 1.0, 0.5])),
    )
    measurement = range_bearing.observe(
        jnp.array([0.1, 0.0, jnp.pi - 0.03]), observation
    )

    result = jax.jit(filter_.step)(
        belief,
        dynamics,
        observation,
        jnp.array([0.0, 0.0, 0.0]),
        measurement,
    )

    assert result.mean.shape == (3,)
    assert jnp.allclose(result.covariance, result.covariance.T)
    assert jnp.all(jnp.linalg.eigvalsh(result.covariance) >= -1e-6)
