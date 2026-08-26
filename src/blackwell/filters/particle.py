"""Generic bootstrap particle filtering over Blackwell model operations."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import Array

from blackwell.beliefs import ParticleBelief
from blackwell.filters._protocols import (
    DynamicsOperations,
    ObservationOperations,
    StateSpaceOperations,
)


@dataclass(frozen=True)
class BootstrapParticleFilter:
    """A particle filter configured with static state-space and model operations.

    Capture an instance in a closure or bind one of its methods before applying
    :func:`jax.jit`. The ``step`` method intentionally does not resample; call
    :meth:`systematic_resample` explicitly when the effective sample size calls
    for it.
    """

    state_space: StateSpaceOperations
    dynamics: DynamicsOperations
    observation: ObservationOperations

    def initialise(
        self,
        key: Array,
        mean: Array,
        covariance: Array,
        particle_count: int,
    ) -> ParticleBelief:
        """Sample an equally weighted local-tangent particle belief.

        ``particle_count`` must be static when this method is JIT compiled.
        """

        tangent_dimension = covariance.shape[0]
        tangent_samples = jax.random.multivariate_normal(
            key,
            mean=jnp.zeros(tangent_dimension, dtype=mean.dtype),
            cov=covariance,
            shape=(particle_count,),
            method="svd",
        )
        particles = jax.vmap(self.state_space.retract, in_axes=(None, 0))(
            mean, tangent_samples
        )
        weights = jnp.full((particle_count,), 1.0 / particle_count, dtype=mean.dtype)
        return ParticleBelief(particles=particles, weights=weights)

    def predict(
        self,
        key: Array,
        belief: ParticleBelief,
        dynamics_model: object,
        control: Array,
    ) -> ParticleBelief:
        """Propagate particles through dynamics and sampled local process noise."""

        covariance = jax.vmap(
            lambda state: self.dynamics.process_covariance(
                state, control, dynamics_model
            )
        )(belief.particles)
        noise = _sample_tangent_noise(key, covariance)
        controlled_particles = jax.vmap(
            lambda state: self.dynamics.propagate(state, control, dynamics_model)
        )(belief.particles)
        particles = jax.vmap(self.state_space.retract)(controlled_particles, noise)
        return ParticleBelief(particles=particles, weights=belief.weights)

    def update(
        self,
        belief: ParticleBelief,
        observation_model: object,
        measurement: Array,
    ) -> ParticleBelief:
        """Reweight particles from normalised, model-specific log likelihoods."""

        residual = jax.vmap(
            lambda state: self.observation.measurement_residual(
                measurement,
                self.observation.observe(state, observation_model),
                observation_model,
            )
        )(belief.particles)
        flattened_residual = residual.reshape(residual.shape[0], -1)
        covariance = jax.vmap(
            lambda state: self.observation.measurement_covariance(
                state, observation_model
            )
        )(belief.particles)
        whitened_residual = jax.vmap(jnp.linalg.solve)(
            covariance, flattened_residual[..., None]
        )[..., 0]
        squared_mahalanobis = jnp.sum(
            flattened_residual * whitened_residual, axis=-1
        )
        _, log_determinant = jax.vmap(jnp.linalg.slogdet)(covariance)
        measurement_dimension = flattened_residual.shape[-1]
        log_likelihood = -0.5 * (
            squared_mahalanobis
            + measurement_dimension * jnp.log(2 * jnp.pi)
            + log_determinant
        )
        log_weights = jnp.log(belief.weights) + log_likelihood
        weights = jnp.exp(log_weights - jsp.special.logsumexp(log_weights))
        return ParticleBelief(particles=belief.particles, weights=weights)

    def systematic_resample(self, key: Array, belief: ParticleBelief) -> ParticleBelief:
        """Systematically resample particles and reset weights to uniform."""

        particle_count = belief.weights.shape[0]
        offset = jax.random.uniform(key, (), dtype=belief.weights.dtype)
        positions = offset + jnp.arange(particle_count, dtype=belief.weights.dtype)
        positions = positions / particle_count
        cumulative_weights = jnp.cumsum(belief.weights).at[-1].set(1.0)
        indices = jnp.searchsorted(cumulative_weights, positions, side="right")
        indices = jnp.minimum(indices, particle_count - 1)
        particles = belief.particles[indices]
        weights = jnp.full_like(belief.weights, 1.0 / particle_count)
        return ParticleBelief(particles=particles, weights=weights)

    def effective_sample_size(self, belief: ParticleBelief) -> Array:
        """Return the standard effective sample size of a normalised belief."""

        return 1.0 / jnp.sum(belief.weights**2)

    def step(
        self,
        key: Array,
        belief: ParticleBelief,
        dynamics_model: object,
        observation_model: object,
        control: Array,
        measurement: Array,
    ) -> ParticleBelief:
        """Perform bootstrap propagation and weighting without resampling."""

        return self.update(
            self.predict(key, belief, dynamics_model, control),
            observation_model,
            measurement,
        )


def _sample_tangent_noise(key: Array, covariance: Array) -> Array:
    """Sample one tangent Gaussian for every leading covariance entry."""

    particle_count = covariance.shape[0]
    tangent_dimension = covariance.shape[-1]
    keys = jax.random.split(key, particle_count)
    mean = jnp.zeros(tangent_dimension, dtype=covariance.dtype)
    return jax.vmap(
        lambda sample_key, sample_covariance: jax.random.multivariate_normal(
            sample_key,
            mean=mean,
            cov=sample_covariance,
            method="svd",
        )
    )(keys, covariance)
