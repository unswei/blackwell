"""Tests for the public generic bootstrap particle filter."""

import jax
import jax.numpy as jnp

from blackwell.beliefs import ParticleBelief
from blackwell.filters.particle import BootstrapParticleFilter
from blackwell.models import range_bearing
from blackwell.models import se2 as se2_models
from blackwell.spaces import se2


def _filter_and_models() -> tuple[
    BootstrapParticleFilter,
    se2_models.BodyMotion,
    range_bearing.KnownLandmarksRangeBearing,
]:
    filter_ = BootstrapParticleFilter(se2, se2_models, range_bearing)
    dynamics = se2_models.BodyMotion(jnp.zeros((3, 3)))
    observation = range_bearing.KnownLandmarksRangeBearing(
        landmarks=jnp.array([[5.0, 0.0], [0.0, 5.0]]),
        measurement_covariance=jnp.diag(jnp.array([0.05, 0.01])),
    )
    return filter_, dynamics, observation


def test_particle_update_prefers_the_particle_matching_measurements() -> None:
    filter_, _, observation = _filter_and_models()
    belief = ParticleBelief(
        particles=jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        weights=jnp.array([0.5, 0.5]),
    )
    measurement = range_bearing.observe(jnp.array([1.0, 0.0, 0.0]), observation)

    result = filter_.update(belief, observation, measurement)

    assert result.weights[1] > result.weights[0]
    assert jnp.isclose(jnp.sum(result.weights), 1.0)


def test_particle_filter_is_jittable_and_resampling_is_explicit() -> None:
    filter_, dynamics, observation = _filter_and_models()
    initialised = filter_.initialise(
        jax.random.key(0), jnp.zeros(3), jnp.zeros((3, 3)), 8
    )
    particle_positions = jnp.linspace(-0.5, 0.5, 8)
    belief = ParticleBelief(
        particles=jnp.stack(
            (particle_positions, jnp.zeros(8), jnp.zeros(8)), axis=-1
        ),
        weights=jnp.full(8, 1.0 / 8),
    )
    measurement = range_bearing.observe(jnp.array([0.2, -0.1, 0.05]), observation)

    weighted = jax.jit(filter_.step)(
        jax.random.key(1),
        belief,
        dynamics,
        observation,
        jnp.array([0.1, 0.0, 0.01]),
        measurement,
    )
    resampled = filter_.systematic_resample(jax.random.key(2), weighted)

    assert jnp.allclose(initialised.particles, jnp.zeros((8, 3)))
    assert jnp.allclose(initialised.weights, jnp.full(8, 1.0 / 8))
    assert jnp.isclose(jnp.sum(weighted.weights), 1.0)
    assert not jnp.allclose(weighted.weights, jnp.full(8, 1.0 / 8))
    assert jnp.allclose(resampled.weights, jnp.full(8, 1.0 / 8))
