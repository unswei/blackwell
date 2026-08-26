"""Tests for the public SE(2) state space."""

import jax
import jax.numpy as jnp

from blackwell.spaces import se2


def test_se2_retract_and_local_coordinates_are_inverse_locally() -> None:
    pose = jnp.array([1.0, -2.0, 0.5])
    tangent = jnp.array([0.3, -0.2, 0.1])

    target = se2.retract(pose, tangent)

    assert jnp.allclose(se2.local_coordinates(pose, target), tangent, atol=1e-6)


def test_se2_transport_matches_the_reexpression_jacobian() -> None:
    reference = jnp.array([0.5, -1.0, 0.2])
    target = jnp.array([1.2, -0.8, 0.5])
    covariance = jnp.array([[2.0, 0.3, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 0.5]])

    def reexpression(tangent: jax.Array) -> jax.Array:
        return se2.local_coordinates(target, se2.retract(reference, tangent))

    jacobian = jax.jacfwd(reexpression)(jnp.zeros(3))
    expected = jacobian @ covariance @ jacobian.T

    assert jnp.allclose(se2.transport(reference, target, covariance), expected)
