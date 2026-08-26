"""A private bootstrap particle-filter localisation experiment.

The filter deliberately reuses the SE(2) motion and range-bearing observation
model from :mod:`blackwell._experiments.se2_range_bearing`.  Its random keys are
explicit so the experiment stays pure and compatible with JAX transformations.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import Array

from blackwell._experiments.se2_range_bearing import (
    RangeBearingLocalisationModel,
    range_bearing,
    retract,
    wrap_angle,
)


class ParticleBelief(NamedTuple):
    """Weighted SE(2) particles used by the private localisation experiment.

    Attributes:
        particles: Poses with shape ``(particle_count, 3)``.
        weights: Normalised particle weights with shape ``(particle_count,)``.
    """

    particles: Array
    weights: Array


def initialise(
    key: Array,
    pose: Array,
    covariance: Array,
    particle_count: int,
) -> ParticleBelief:
    """Sample an equally weighted local-tangent particle belief."""

    tangent_samples = jax.random.multivariate_normal(
        key,
        mean=jnp.zeros(3, dtype=pose.dtype),
        cov=covariance,
        shape=(particle_count,),
        method="svd",
    )
    particles = jax.vmap(retract, in_axes=(None, 0))(pose, tangent_samples)
    weights = jnp.full((particle_count,), 1.0 / particle_count, dtype=pose.dtype)
    return ParticleBelief(particles=particles, weights=weights)


def predict(
    key: Array,
    belief: ParticleBelief,
    model: RangeBearingLocalisationModel,
    control: Array,
) -> ParticleBelief:
    """Propagate particles through noisy body-frame SE(2) motion.

    Process noise is applied after the control increment, matching the tangent
    covariance convention used by the EKF experiment.
    """

    noise = jax.random.multivariate_normal(
        key,
        mean=jnp.zeros(3, dtype=belief.particles.dtype),
        cov=model.process_covariance,
        shape=(belief.particles.shape[0],),
        method="svd",
    )
    controlled_particles = jax.vmap(retract, in_axes=(0, None))(
        belief.particles, control
    )
    particles = jax.vmap(retract)(controlled_particles, noise)
    return ParticleBelief(particles=particles, weights=belief.weights)


def update(
    belief: ParticleBelief,
    model: RangeBearingLocalisationModel,
    measurements: Array,
) -> ParticleBelief:
    """Reweight particles from independent landmark range-bearing likelihoods."""

    expected_measurements = jax.vmap(range_bearing, in_axes=(0, None))(
        belief.particles, model.landmarks
    )
    residual = measurements[None, :, :] - expected_measurements
    residual = residual.at[:, :, 1].set(wrap_angle(residual[:, :, 1]))
    whitened_residual = jnp.linalg.solve(
        model.measurement_covariance, residual[..., None]
    )[..., 0]
    squared_mahalanobis = jnp.sum(residual * whitened_residual, axis=(1, 2))
    _, log_determinant = jnp.linalg.slogdet(model.measurement_covariance)
    landmark_count = model.landmarks.shape[0]
    log_likelihood = -0.5 * (
        squared_mahalanobis
        + landmark_count * (2 * jnp.log(2 * jnp.pi) + log_determinant)
    )
    log_weights = jnp.log(belief.weights) + log_likelihood
    weights = jnp.exp(log_weights - jsp.special.logsumexp(log_weights))
    return ParticleBelief(particles=belief.particles, weights=weights)


def systematic_resample(key: Array, belief: ParticleBelief) -> ParticleBelief:
    """Systematically resample particles and reset weights to uniform."""

    particle_count = belief.weights.shape[0]
    offset = jax.random.uniform(key, (), dtype=belief.weights.dtype)
    positions = (offset + jnp.arange(particle_count, dtype=belief.weights.dtype))
    positions = positions / particle_count
    cumulative_weights = jnp.cumsum(belief.weights).at[-1].set(1.0)
    indices = jnp.searchsorted(cumulative_weights, positions, side="right")
    indices = jnp.minimum(indices, particle_count - 1)
    particles = belief.particles[indices]
    weights = jnp.full_like(belief.weights, 1.0 / particle_count)
    return ParticleBelief(particles=particles, weights=weights)


def effective_sample_size(belief: ParticleBelief) -> Array:
    """Return the standard effective sample size of a normalised belief."""

    return 1.0 / jnp.sum(belief.weights**2)


def step(
    key: Array,
    belief: ParticleBelief,
    model: RangeBearingLocalisationModel,
    control: Array,
    measurements: Array,
) -> ParticleBelief:
    """Perform bootstrap propagation, weighting, and systematic resampling."""

    prediction_key, resampling_key = jax.random.split(key)
    predicted = predict(prediction_key, belief, model, control)
    weighted = update(predicted, model, measurements)
    return systematic_resample(resampling_key, weighted)
