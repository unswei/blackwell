"""SE(3) uses the existing generic filters without changing belief contracts."""

from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp
from numpy.testing import assert_allclose

from blackwell import GaussianBelief
from blackwell.filters.ekf import ExtendedKalmanFilter
from blackwell.filters.particle import BootstrapParticleFilter
from blackwell.spaces import se3


class PointObservations(NamedTuple):
    points: jax.Array
    covariance: jax.Array


def observe(state, model):
    return jax.vmap(se3.transform_point, in_axes=(None, 0))(state, model.points)


def observation_jacobian(state, model):
    return jax.jacfwd(lambda delta: observe(se3.retract(state, delta), model))(
        jnp.zeros(6, dtype=state.dtype)
    )


dynamics = SimpleNamespace(
    propagate=lambda state, control, model: se3.retract(state, control),
    transition_jacobian=lambda state, control, model: se3.adjoint(
        se3.inverse(se3.exp(control))
    ),
    process_covariance=lambda state, control, model: model,
)
observation = SimpleNamespace(
    observe=observe,
    observation_jacobian=observation_jacobian,
    measurement_covariance=lambda state, model: model.covariance,
    measurement_residual=lambda measurement, expected, model: measurement - expected,
)


def test_se3_gaussian_filter_predict_update_scan_vmap_and_differentiate():
    filter_ = ExtendedKalmanFilter(se3, dynamics, observation)
    mean = se3.exp(jnp.array([0.3, -0.1, 0.5, 0.2, -0.3, 0.1]))
    initial = GaussianBelief(mean, jnp.eye(6) * 0.01)
    process = jnp.eye(6) * 1e-5
    sensor = PointObservations(jnp.eye(3), jnp.eye(9) * 0.001)
    control = jnp.array([0.1, 0, 0.02, 0.01, 0.02, -0.03])
    predicted = jax.jit(filter_.predict)(initial, process, control)
    transition = se3.adjoint(se3.inverse(se3.exp(control)))
    assert_allclose(
        predicted.covariance,
        transition @ initial.covariance @ transition.T + process,
        atol=1e-8,
    )
    truth = se3.retract(predicted.mean, jnp.array([0.02, -0.01, 0.01, 0.01, 0, -0.01]))
    measurement = observe(truth, sensor)
    corrected = jax.jit(filter_.update)(predicted, sensor, measurement)
    before = jnp.linalg.norm(se3.local_coordinates(predicted.mean, truth))
    after = jnp.linalg.norm(se3.local_coordinates(corrected.mean, truth))
    assert after < before
    assert corrected.mean.shape == (7,)
    assert corrected.covariance.shape == (6, 6)
    assert_allclose(corrected.covariance, corrected.covariance.T, atol=1e-8)
    assert jnp.linalg.eigvalsh(corrected.covariance).min() > 0
    assert_allclose(jnp.linalg.norm(corrected.mean[3:]), 1, atol=1e-6)

    def run(belief):
        def step(current, unused):
            result = filter_.step(current, process, sensor, control, measurement)
            return result, result

        return jax.lax.scan(step, belief, xs=None, length=3)[1]

    batched = jax.tree.map(lambda x: jnp.stack([x, x]), initial)
    histories = jax.jit(jax.vmap(run))(batched)
    assert histories.mean.shape == (2, 3, 7)
    assert histories.covariance.shape == (2, 3, 6, 6)

    # Differentiating through transport needs second derivatives of geometry.
    def loss(delta):
        moved = GaussianBelief(se3.retract(mean, delta), initial.covariance)
        result = filter_.step(moved, process, sensor, control, measurement)
        return jnp.sum(result.mean**2) + jnp.trace(result.covariance)

    assert jnp.all(jnp.isfinite(jax.jit(jax.grad(loss))(jnp.zeros(6))))


def test_se3_particle_filter_initialise_predict_update_resample():
    filter_ = BootstrapParticleFilter(se3, dynamics, observation)
    mean = se3.exp(jnp.array([0.3, -0.1, 0.5, 0.2, -0.3, 0.1]))
    key_init, key_step, key_resample = jax.random.split(jax.random.key(20), 3)
    initial = jax.jit(filter_.initialise, static_argnames="particle_count")(
        key_init, mean, jnp.eye(6) * 0.001, particle_count=256
    )
    sensor = PointObservations(jnp.eye(3), jnp.eye(9) * 0.01)
    control = jnp.array([0.1, 0, 0.02, 0.01, 0.02, -0.03])
    posterior = jax.jit(filter_.step)(
        key_step,
        initial,
        jnp.eye(6) * 1e-5,
        sensor,
        control,
        observe(se3.retract(mean, control), sensor),
    )
    resampled = jax.jit(filter_.systematic_resample)(key_resample, posterior)
    for belief in [initial, posterior, resampled]:
        assert belief.particles.shape == (256, 7)
        assert jnp.all(jnp.isfinite(belief.particles))
        assert_allclose(jnp.linalg.norm(belief.particles[:, 3:], axis=1), 1, atol=1e-6)
        assert_allclose(jnp.sum(belief.weights), 1, atol=1e-6)
        assert jnp.all(belief.weights >= 0)
    assert 1 <= filter_.effective_sample_size(posterior) <= 256
    assert_allclose(resampled.weights, jnp.full(256, 1 / 256), atol=1e-8)
