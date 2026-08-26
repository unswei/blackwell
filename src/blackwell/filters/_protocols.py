"""Structural types for the static operations captured by filter kernels."""

from __future__ import annotations

from typing import Protocol

from jax import Array


class StateSpaceOperations(Protocol):
    """Operations a filter needs from a state space."""

    def retract(self, state: Array, tangent: Array) -> Array:
        """Apply a tangent correction to a state."""

    def transport(self, reference: Array, target: Array, covariance: Array) -> Array:
        """Transport covariance between local tangent coordinates."""


class DynamicsOperations(Protocol):
    """Operations a filter needs from a dynamics-model family."""

    def propagate(self, state: Array, control: Array, model: object) -> Array:
        """Propagate a state through deterministic dynamics."""

    def transition_jacobian(
        self, state: Array, control: Array, model: object
    ) -> Array:
        """Return the local tangent transition Jacobian."""

    def process_covariance(self, state: Array, control: Array, model: object) -> Array:
        """Return local process covariance."""


class ObservationOperations(Protocol):
    """Operations a filter needs from an observation-model family."""

    def observe(self, state: Array, model: object) -> Array:
        """Predict a measurement."""

    def observation_jacobian(self, state: Array, model: object) -> Array:
        """Return the measurement Jacobian with respect to a local tangent."""

    def measurement_covariance(self, state: Array, model: object) -> Array:
        """Return measurement covariance in flattened measurement coordinates."""

    def measurement_residual(
        self, measurement: Array, expected: Array, model: object
    ) -> Array:
        """Return a measurement residual with model-specific topology handling."""
