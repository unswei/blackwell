"""Tests for public model-based simulation."""

import jax
import jax.numpy as jnp

from blackwell.models import linear
from blackwell.simulation import Simulator
from blackwell.spaces import euclidean


def test_simulator_rollout_is_jittable_and_supports_zero_noise() -> None:
    simulator = Simulator(euclidean, linear, linear)
    dynamics = linear.LinearDynamics(
        transition=jnp.array([[1.0]]),
        control=jnp.array([[1.0]]),
        process_covariance=jnp.zeros((1, 1)),
    )
    observation = linear.LinearObservation(
        observation=jnp.array([[1.0]]),
        measurement_covariance=jnp.zeros((1, 1)),
    )

    states, measurements = jax.jit(simulator.rollout)(
        jax.random.key(0),
        jnp.array([0.0]),
        dynamics,
        observation,
        jnp.array([[1.0], [2.0]]),
    )

    assert jnp.allclose(states, jnp.array([[1.0], [3.0]]))
    assert jnp.allclose(measurements, states)
