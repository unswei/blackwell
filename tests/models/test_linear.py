"""Tests for public linear dynamics and observation models."""

import jax.numpy as jnp

from blackwell.models import linear


def test_linear_model_operations_expose_dynamics_and_observation_terms() -> None:
    dynamics = linear.LinearDynamics(
        transition=jnp.array([[1.0, 1.0], [0.0, 1.0]]),
        control=jnp.array([[0.5], [1.0]]),
        process_covariance=jnp.diag(jnp.array([0.1, 0.2])),
    )
    observation = linear.LinearObservation(
        observation=jnp.array([[1.0, 0.0]]),
        measurement_covariance=jnp.array([[0.3]]),
    )
    state = jnp.array([2.0, 3.0])
    control = jnp.array([4.0])

    assert jnp.allclose(
        linear.propagate(state, control, dynamics), jnp.array([7.0, 7.0])
    )
    assert jnp.allclose(
        linear.transition_jacobian(state, control, dynamics), dynamics.transition
    )
    assert jnp.allclose(
        linear.process_covariance(state, control, dynamics),
        dynamics.process_covariance,
    )
    assert jnp.allclose(linear.observe(state, observation), jnp.array([2.0]))
    assert jnp.allclose(
        linear.observation_jacobian(state, observation), observation.observation
    )
    assert jnp.allclose(
        linear.measurement_residual(jnp.array([3.0]), jnp.array([2.0]), observation),
        jnp.array([1.0]),
    )
