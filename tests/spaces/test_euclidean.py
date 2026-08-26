"""Tests for the public Euclidean state space."""

import jax.numpy as jnp

from blackwell.spaces import euclidean


def test_euclidean_operations_are_addition_subtraction_and_identity_transport() -> None:
    reference = jnp.array([1.0, -2.0])
    tangent = jnp.array([0.5, 3.0])
    covariance = jnp.array([[2.0, 0.4], [0.4, 1.0]])

    target = euclidean.retract(reference, tangent)

    assert jnp.allclose(target, jnp.array([1.5, 1.0]))
    assert jnp.allclose(euclidean.local_coordinates(reference, target), tangent)
    assert jnp.allclose(euclidean.transport(reference, target, covariance), covariance)
