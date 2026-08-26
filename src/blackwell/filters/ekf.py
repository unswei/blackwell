"""Generic extended Kalman filtering over Blackwell model operations."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from blackwell.beliefs import GaussianBelief
from blackwell.filters._protocols import (
    DynamicsOperations,
    ObservationOperations,
    StateSpaceOperations,
)


@dataclass(frozen=True)
class ExtendedKalmanFilter:
    """An EKF configured with static state-space and model-family operations.

    Capture an instance in a closure or bind one of its methods before applying
    :func:`jax.jit`. Model parameter objects remain function arguments and are
    therefore ordinary dynamic JAX PyTrees.
    """

    state_space: StateSpaceOperations
    dynamics: DynamicsOperations
    observation: ObservationOperations

    def predict(
        self,
        belief: GaussianBelief,
        dynamics_model: object,
        control: Array,
    ) -> GaussianBelief:
        """Apply one tangent-space Gaussian prediction step."""

        mean = self.dynamics.propagate(belief.mean, control, dynamics_model)
        transition = self.dynamics.transition_jacobian(
            belief.mean, control, dynamics_model
        )
        covariance = (
            transition @ belief.covariance @ transition.T
            + self.dynamics.process_covariance(
                belief.mean, control, dynamics_model
            )
        )
        return GaussianBelief(mean=mean, covariance=_symmetrise(covariance))

    def update(
        self,
        belief: GaussianBelief,
        observation_model: object,
        measurement: Array,
    ) -> GaussianBelief:
        """Apply a Joseph-form measurement update in local tangent coordinates."""

        expected = self.observation.observe(belief.mean, observation_model)
        innovation = self.observation.measurement_residual(
            measurement, expected, observation_model
        ).reshape(-1)
        tangent_dimension = belief.covariance.shape[0]
        observation = self.observation.observation_jacobian(
            belief.mean, observation_model
        ).reshape(-1, tangent_dimension)
        measurement_covariance = self.observation.measurement_covariance(
            belief.mean, observation_model
        )
        innovation_covariance = (
            observation @ belief.covariance @ observation.T + measurement_covariance
        )
        gain = jnp.linalg.solve(
            innovation_covariance, observation @ belief.covariance
        ).T

        mean = self.state_space.retract(belief.mean, gain @ innovation)
        identity = jnp.eye(tangent_dimension, dtype=belief.covariance.dtype)
        residual = identity - gain @ observation
        covariance = (
            residual @ belief.covariance @ residual.T
            + gain @ measurement_covariance @ gain.T
        )
        covariance = self.state_space.transport(belief.mean, mean, covariance)
        return GaussianBelief(mean=mean, covariance=_symmetrise(covariance))

    def step(
        self,
        belief: GaussianBelief,
        dynamics_model: object,
        observation_model: object,
        control: Array,
        measurement: Array,
    ) -> GaussianBelief:
        """Apply one prediction followed by one measurement update."""

        return self.update(
            self.predict(belief, dynamics_model, control),
            observation_model,
            measurement,
        )


def _symmetrise(matrix: Array) -> Array:
    """Limit numerical skew in a covariance matrix."""

    return (matrix + matrix.T) / 2
