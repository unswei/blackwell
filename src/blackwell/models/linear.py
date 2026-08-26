"""Linear Euclidean dynamics and observation models."""

from __future__ import annotations

from typing import NamedTuple

from jax import Array


class LinearDynamics(NamedTuple):
    """Discrete-time linear dynamics with additive process uncertainty.

    The transition is ``x_next = transition @ x + control @ u``.

    Attributes:
        transition: State transition matrix with shape ``(state_dim, state_dim)``.
        control: Control matrix with shape ``(state_dim, control_dim)``.
        process_covariance: Additive state covariance with shape
            ``(state_dim, state_dim)``.
    """

    transition: Array
    control: Array
    process_covariance: Array


class LinearObservation(NamedTuple):
    """Linear observation model with additive measurement uncertainty.

    The expected measurement is ``observation @ state``.

    Attributes:
        observation: Observation matrix with shape
            ``(measurement_dim, state_dim)``.
        measurement_covariance: Additive covariance with shape
            ``(measurement_dim, measurement_dim)``.
    """

    observation: Array
    measurement_covariance: Array


def propagate(state: Array, control: Array, model: LinearDynamics) -> Array:
    """Propagate a Euclidean state through the linear dynamics.

    Args:
        state: Current state with shape ``(state_dim,)``.
        control: Current input with shape ``(control_dim,)``.
        model: Linear dynamics matrices and process covariance.

    Returns:
        Next deterministic state with shape ``(state_dim,)``.
    """

    return model.transition @ state + model.control @ control


def transition_jacobian(
    state: Array, control: Array, model: LinearDynamics
) -> Array:
    """Return the state transition Jacobian in Euclidean coordinates.

    Args:
        state: Current state; unused because the model is linear.
        control: Current input; unused because the model is linear.
        model: Linear dynamics parameters.

    Returns:
        ``model.transition``.
    """

    del state, control
    return model.transition


def process_covariance(
    state: Array, control: Array, model: LinearDynamics
) -> Array:
    """Return additive process covariance in next-state coordinates.

    Args:
        state: Current state; unused because covariance is constant.
        control: Current input; unused because covariance is constant.
        model: Linear dynamics parameters.

    Returns:
        ``model.process_covariance``.
    """

    del state, control
    return model.process_covariance


def observe(state: Array, model: LinearObservation) -> Array:
    """Predict a linear observation from a state.

    Args:
        state: State with shape ``(state_dim,)``.
        model: Linear observation matrix and covariance.

    Returns:
        Expected measurement with shape ``(measurement_dim,)``.
    """

    return model.observation @ state


def observation_jacobian(state: Array, model: LinearObservation) -> Array:
    """Return the observation Jacobian in Euclidean coordinates.

    Args:
        state: Current state; unused because the model is linear.
        model: Linear observation parameters.

    Returns:
        ``model.observation``.
    """

    del state
    return model.observation


def measurement_covariance(state: Array, model: LinearObservation) -> Array:
    """Return additive measurement covariance.

    Args:
        state: Current state; unused because covariance is constant.
        model: Linear observation parameters.

    Returns:
        ``model.measurement_covariance``.
    """

    del state
    return model.measurement_covariance


def measurement_residual(
    measurement: Array, expected: Array, model: LinearObservation
) -> Array:
    """Return the Euclidean residual ``measurement - expected``.

    Args:
        measurement: Observed measurement array.
        expected: Predicted measurement with matching shape.
        model: Linear observation parameters; unused by subtraction.

    Returns:
        Measurement residual with the input measurement shape.
    """

    del model
    return measurement - expected
