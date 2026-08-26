"""Simulation helpers for models with local additive Gaussian uncertainty."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from blackwell.filters._protocols import (
    DynamicsOperations,
    ObservationOperations,
    StateSpaceOperations,
)


@dataclass(frozen=True)
class Simulator:
    """Simulator configured with static state-space and model-family operations.

    The simulator first propagates a state, applies local process noise, then
    samples additive measurement noise in flattened measurement coordinates.
    Periodic observation models should handle equivalent representations in
    their residual operation.

    Attributes:
        state_space: Operations providing state retraction.
        dynamics: Operations providing propagation and process covariance.
        observation: Operations providing measurement prediction and
            covariance.
    """

    state_space: StateSpaceOperations
    dynamics: DynamicsOperations
    observation: ObservationOperations

    def step(
        self,
        key: Array,
        state: Array,
        dynamics_model: object,
        observation_model: object,
        control: Array,
    ) -> tuple[Array, Array]:
        """Simulate one noisy state transition and its measurement.

        Args:
            key: JAX random key, split once for process and measurement noise.
            state: Current state array.
            dynamics_model: Parameters understood by ``dynamics``.
            observation_model: Parameters understood by ``observation``.
            control: Control array for this transition.

        Returns:
            A pair ``(next_state, measurement)``. Process noise is sampled in
            local tangent coordinates and measurement noise in flattened
            measurement coordinates.
        """

        process_key, measurement_key = jax.random.split(key)
        propagated = self.dynamics.propagate(state, control, dynamics_model)
        process_covariance = self.dynamics.process_covariance(
            state, control, dynamics_model
        )
        process_noise = _sample_gaussian(process_key, process_covariance)
        next_state = self.state_space.retract(propagated, process_noise)
        expected_measurement = self.observation.observe(next_state, observation_model)
        measurement_covariance = self.observation.measurement_covariance(
            next_state, observation_model
        )
        measurement_noise = _sample_gaussian(measurement_key, measurement_covariance)
        measurement = expected_measurement + measurement_noise.reshape(
            expected_measurement.shape
        )
        return next_state, measurement

    def rollout(
        self,
        key: Array,
        initial_state: Array,
        dynamics_model: object,
        observation_model: object,
        controls: Array,
    ) -> tuple[Array, Array]:
        """Simulate one trajectory with shape-static controls and JAX scan.

        Args:
            key: JAX random key for the complete rollout.
            initial_state: State immediately before the first control.
            dynamics_model: Parameters understood by ``dynamics``.
            observation_model: Parameters understood by ``observation``.
            controls: Control sequence with leading shape ``(step_count,)``.

        Returns:
            ``(states, measurements)`` with one entry per control. The initial
            state itself is not included in ``states``.
        """

        keys = jax.random.split(key, controls.shape[0])

        def run_step(
            state: Array,
            inputs: tuple[Array, Array],
        ) -> tuple[Array, tuple[Array, Array]]:
            step_key, control = inputs
            next_state, measurement = self.step(
                step_key,
                state,
                dynamics_model,
                observation_model,
                control,
            )
            return next_state, (next_state, measurement)

        _, (states, measurements) = jax.lax.scan(
            run_step, initial_state, (keys, controls)
        )
        return states, measurements


def _sample_gaussian(key: Array, covariance: Array) -> Array:
    """Sample a local Gaussian, including positive-semidefinite covariances."""

    dimension = covariance.shape[0]
    return jax.random.multivariate_normal(
        key,
        mean=jnp.zeros(dimension, dtype=covariance.dtype),
        cov=covariance,
        method="svd",
    )
