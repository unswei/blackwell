"""Linear Euclidean dynamics and observation models."""

from __future__ import annotations

from typing import NamedTuple

from jax import Array


class LinearDynamics(NamedTuple):
    """Discrete-time linear dynamics with additive process uncertainty."""

    transition: Array
    control: Array
    process_covariance: Array


class LinearObservation(NamedTuple):
    """Linear observation model with additive measurement uncertainty."""

    observation: Array
    measurement_covariance: Array


def propagate(state: Array, control: Array, model: LinearDynamics) -> Array:
    """Propagate a Euclidean state through the linear dynamics."""

    return model.transition @ state + model.control @ control


def transition_jacobian(
    state: Array, control: Array, model: LinearDynamics
) -> Array:
    """Return the state transition Jacobian in Euclidean coordinates."""

    del state, control
    return model.transition


def process_covariance(
    state: Array, control: Array, model: LinearDynamics
) -> Array:
    """Return additive process covariance in the next state coordinates."""

    del state, control
    return model.process_covariance


def observe(state: Array, model: LinearObservation) -> Array:
    """Predict a linear observation from a state."""

    return model.observation @ state


def observation_jacobian(state: Array, model: LinearObservation) -> Array:
    """Return the observation Jacobian in Euclidean coordinates."""

    del state
    return model.observation


def measurement_covariance(state: Array, model: LinearObservation) -> Array:
    """Return additive measurement covariance."""

    del state
    return model.measurement_covariance


def measurement_residual(
    measurement: Array, expected: Array, model: LinearObservation
) -> Array:
    """Return the Euclidean observation residual."""

    del model
    return measurement - expected
