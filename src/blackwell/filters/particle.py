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

    Attributes:
        state_space: Operations providing state retraction.
        dynamics: Operations providing propagation and process covariance.
        observation: Operations providing prediction, covariance and residual.
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

        Args:
            key: JAX random key used once for all initial samples.
            mean: Nominal state about which local samples are drawn.
            covariance: Local Gaussian covariance with shape
                ``(tangent_dim, tangent_dim)``.
            particle_count: Number of samples. This value must be static when
                the method is JIT compiled.

        Returns:
            Particles retracted onto the state space with uniform normalised
            weights.
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
        """Propagate particles and sample local process noise.

        Args:
            key: JAX random key used once for process-noise samples.
            belief: Prior normalised particle belief.
            dynamics_model: Parameters understood by ``dynamics``.
            control: Control array shared by every particle.

        Returns:
            Propagated particles with the prior weights unchanged.
        """

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
        """Reweight particles from model-specific Gaussian likelihoods.

        Args:
            belief: Predicted particle belief with finite normalised weights.
            observation_model: Parameters understood by ``observation``.
            measurement: Measurement shared by every particle.

        Returns:
            The same particle states with posterior weights normalised in log
            space.

        Note:
            Observation covariance must be positive definite for the linear
            solve and log determinant used by the Gaussian likelihood.
        """

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
        """Systematically resample particles and reset weights to uniform.

        Args:
            key: JAX random key used once for the systematic offset.
            belief: Normalised particle belief to resample.

        Returns:
            A belief with selected particle states and uniform weights.
        """

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
        """Return effective sample size ``1 / sum(weights**2)``.

        Args:
            belief: Particle belief with normalised weights.

        Returns:
            Scalar effective sample size in ``[1, particle_count]`` for a valid
            non-empty belief.
        """

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
        """Perform bootstrap propagation and weighting without resampling.

        Args:
            key: JAX random key for process-noise samples.
            belief: Prior particle belief.
            dynamics_model: Parameters understood by ``dynamics``.
            observation_model: Parameters understood by ``observation``.
            control: Control shared by every particle.
            measurement: Measurement shared by every particle.

        Returns:
            Predicted and reweighted particle belief. Call
            :meth:`effective_sample_size` and :meth:`systematic_resample`
            separately to apply an explicit resampling policy.
        """

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
