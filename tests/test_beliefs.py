"""Tests for public belief containers."""

import jax
import jax.numpy as jnp

import blackwell


def test_public_beliefs_are_jax_pytrees() -> None:
    gaussian = blackwell.GaussianBelief(jnp.zeros(2), jnp.eye(2))
    particles = blackwell.ParticleBelief(jnp.zeros((3, 2)), jnp.full(3, 1 / 3))

    doubled_gaussian = jax.jit(
        lambda belief: blackwell.GaussianBelief(
            belief.mean * 2, belief.covariance * 2
        )
    )(gaussian)
    doubled_particles = jax.jit(
        lambda belief: blackwell.ParticleBelief(
            belief.particles * 2, belief.weights
        )
    )(particles)

    assert jnp.allclose(doubled_gaussian.mean, jnp.zeros(2))
    assert jnp.allclose(doubled_gaussian.covariance, jnp.eye(2) * 2)
    assert jnp.allclose(doubled_particles.particles, jnp.zeros((3, 2)))
    assert jnp.allclose(doubled_particles.weights, particles.weights)
