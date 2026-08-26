"""Tests for the private bootstrap particle-localisation experiment."""

import jax
import jax.numpy as jnp

from blackwell._experiments.particle_localisation import (
    ParticleBelief,
    effective_sample_size,
    initialise,
    predict,
    step,
    systematic_resample,
    update,
)
from blackwell._experiments.se2_range_bearing import (
    RangeBearingLocalisationModel,
    range_bearing,
)


def _model(
    process_covariance: jax.Array | None = None,
) -> RangeBearingLocalisationModel:
    return RangeBearingLocalisationModel(
        landmarks=jnp.array([[5.0, 0.0], [0.0, 5.0]]),
        process_covariance=(
            jnp.zeros((3, 3))
            if process_covariance is None
            else process_covariance
        ),
        measurement_covariance=jnp.diag(jnp.array([0.05, 0.01])),
    )


def test_initialise_creates_equally_weighted_tangent_particles() -> None:
    pose = jnp.array([1.0, -2.0, 0.3])

    belief = initialise(jax.random.key(0), pose, jnp.zeros((3, 3)), particle_count=4)

    assert jnp.allclose(belief.particles, jnp.broadcast_to(pose, (4, 3)))
    assert jnp.allclose(belief.weights, jnp.full(4, 0.25))
    assert jnp.isclose(effective_sample_size(belief), 4.0)


def test_predict_reuses_the_se2_body_frame_motion_model() -> None:
    belief = ParticleBelief(
        particles=jnp.array([[0.0, 0.0, jnp.pi / 2], [1.0, 2.0, 0.0]]),
        weights=jnp.array([0.4, 0.6]),
    )

    result = predict(jax.random.key(1), belief, _model(), jnp.array([1.0, 0.0, 0.0]))

    assert jnp.allclose(
        result.particles,
        jnp.array([[0.0, 1.0, jnp.pi / 2], [2.0, 2.0, 0.0]]),
        atol=1e-6,
    )
    assert jnp.allclose(result.weights, belief.weights)


def test_update_assigns_more_weight_to_the_particle_matching_measurements() -> None:
    model = _model()
    belief = ParticleBelief(
        particles=jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        weights=jnp.array([0.5, 0.5]),
    )
    measurements = range_bearing(jnp.array([1.0, 0.0, 0.0]), model.landmarks)

    result = update(belief, model, measurements)

    assert result.weights[1] > result.weights[0]
    assert jnp.isclose(jnp.sum(result.weights), 1.0)


def test_systematic_resample_resets_uniform_weights() -> None:
    belief = ParticleBelief(
        particles=jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        weights=jnp.array([0.0, 1.0]),
    )

    result = systematic_resample(jax.random.key(2), belief)

    assert jnp.allclose(result.particles, jnp.array([[1.0, 0.0, 0.0]] * 2))
    assert jnp.allclose(result.weights, jnp.array([0.5, 0.5]))


def test_step_is_jittable() -> None:
    model = _model(jnp.eye(3) * 0.01)
    belief = initialise(jax.random.key(3), jnp.zeros(3), jnp.eye(3) * 0.1, 8)
    measurements = range_bearing(jnp.array([0.2, -0.1, 0.05]), model.landmarks)

    result = jax.jit(step)(
        jax.random.key(4), belief, model, jnp.array([0.1, 0.0, 0.01]), measurements
    )

    assert result.particles.shape == (8, 3)
    assert jnp.allclose(result.weights, jnp.full(8, 1.0 / 8))
