"""Tests for public SE(2) motion and range-bearing observation models."""

import jax
import jax.numpy as jnp

from blackwell.models import range_bearing, se2
from blackwell.spaces import se2 as space


def test_body_motion_jacobian_matches_automatic_differentiation() -> None:
    model = se2.BodyMotion(jnp.eye(3) * 0.1)
    state = jnp.array([1.0, -2.0, 0.3])
    control = jnp.array([0.5, -0.1, 0.2])
    predicted = se2.propagate(state, control, model)

    automatic_jacobian = jax.jacfwd(
        lambda tangent: space.local_coordinates(
            predicted, se2.propagate(space.retract(state, tangent), control, model)
        )
    )(jnp.zeros(3))

    assert jnp.allclose(
        se2.transition_jacobian(state, control, model), automatic_jacobian, atol=1e-6
    )


def test_range_bearing_model_wraps_residuals_and_matches_autodiff() -> None:
    model = range_bearing.KnownLandmarksRangeBearing(
        landmarks=jnp.array([[4.0, 2.0], [-1.0, 3.0]]),
        measurement_covariance=jnp.diag(jnp.array([0.2, 0.1])),
    )
    state = jnp.array([1.0, -0.5, 0.4])
    automatic_jacobian = jax.jacfwd(
        lambda tangent: range_bearing.observe(space.retract(state, tangent), model)
    )(jnp.zeros(3))
    expected = range_bearing.observe(state, model)
    measurement = expected.at[0, 1].set(-jnp.pi + 0.01)
    residual = range_bearing.measurement_residual(measurement, expected, model)

    assert jnp.allclose(
        range_bearing.observation_jacobian(state, model), automatic_jacobian, atol=1e-6
    )
    assert residual[0, 1] > 0.0
    assert jnp.allclose(
        range_bearing.measurement_covariance(state, model),
        jnp.kron(jnp.eye(2), model.measurement_covariance),
    )
