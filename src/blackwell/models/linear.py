"""Linear Euclidean dynamics and observation models."""

from __future__ import annotations

from numbers import Integral
from typing import NamedTuple

import jax
import jax.numpy as jnp
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


class PredictionResponse(NamedTuple):
    """Prepared finite-horizon linear prediction operators.

    This immutable PyTree holds an affine map from one initial state and one
    control sequence to future outputs. Time is axis zero and the initial
    output is excluded. ``state_response`` and ``command_response`` are also
    the exact initial-state and control Jacobians of :func:`predict_sequence`.

    Attributes:
        state_response: Initial-state response, shape ``(H, p, n)``.
        command_response: Control response, shape ``(H, p, H, m)``. The block
            ``[t, :, j, :]`` is zero for ``j > t``.
        offset: Accumulated affine offsets, shape ``(H, p)``.

    Note:
        Storage is quadratic in the horizon: ``O(H**2 * p * m)`` for the
        command response. Rebuild when model matrices, offsets or horizon
        change. No covariance is stored or propagated.
    """

    state_response: Array
    command_response: Array
    offset: Array


def prepare_prediction(
    model: LinearDynamics,
    horizon: int,
    *,
    output_matrix: Array | None = None,
    state_offset: Array | None = None,
    output_offset: Array | None = None,
) -> PredictionResponse:
    """Prepare reusable responses for a time-invariant affine linear model.

    The model is ``x_next = A @ x + B @ u + a`` and
    ``y_next = C @ x_next + d``. ``A`` and ``B`` come from ``model``; its
    process covariance is unused. Zero-mean additive noise does not alter
    these conditional means. Offsets apply only to this prediction utility;
    they do not change :func:`propagate` or filter behaviour.

    Preparation is explicit, with no global cache or internal compilation.
    Pass ``static_argnames="horizon"`` when using ``jax.jit``. To differentiate
    predictions with respect to model parameters, prepare the response inside
    the differentiated computation. Use ``jax.vmap`` for independent models.

    Args:
        model: Linear dynamics with transition ``A`` of shape ``(n, n)`` and
            control matrix ``B`` of shape ``(n, m)``.
        horizon: Positive static integer ``H`` giving the number of controls.
        output_matrix: Projection ``C`` with shape ``(p, n)``; defaults to
            the identity, returning the complete future state.
        state_offset: Constant transition offset ``a``, shape ``(n,)``;
            defaults to zero.
        output_offset: Constant output offset ``d``, shape ``(p,)``;
            defaults to zero.

    Returns:
        Responses for outputs ``y[1:H+1]``. Floating inputs of a common dtype
        retain that dtype; mixed inputs follow JAX type promotion. Integer
        inputs are promoted to floating point. This map applies to Euclidean
        states, not directly to SE(2) or SE(3) pose coordinates.

    Raises:
        ValueError: The horizon is not a positive static integer or array
            shapes are incompatible.
    """

    if isinstance(horizon, bool) or not isinstance(horizon, Integral) or horizon < 1:
        raise ValueError("horizon must be a positive static integer")
    transition = jnp.asarray(model.transition)
    control = jnp.asarray(model.control)
    if transition.ndim != 2 or transition.shape[0] != transition.shape[1]:
        raise ValueError("model.transition must have shape (n, n)")
    n = transition.shape[0]
    if control.ndim != 2 or control.shape[0] != n:
        raise ValueError("model.control must have shape (n, m)")
    arrays = [transition, control]
    if output_matrix is not None:
        output_matrix = jnp.asarray(output_matrix)
        if output_matrix.ndim != 2 or output_matrix.shape[1] != n:
            raise ValueError("output_matrix must have shape (p, n)")
        arrays.append(output_matrix)
    p = n if output_matrix is None else output_matrix.shape[0]
    if state_offset is not None:
        state_offset = jnp.asarray(state_offset)
        _check_shape("state_offset", state_offset, (n,))
        arrays.append(state_offset)
    if output_offset is not None:
        output_offset = jnp.asarray(output_offset)
        _check_shape("output_offset", output_offset, (p,))
        arrays.append(output_offset)

    dtype = jnp.result_type(*arrays, 0.0)
    transition = transition.astype(dtype)
    control = control.astype(dtype)
    projection = (
        jnp.eye(n, dtype=dtype)
        if output_matrix is None
        else output_matrix.astype(dtype)
    )
    a = (
        jnp.zeros(n, dtype=dtype)
        if state_offset is None
        else state_offset.astype(dtype)
    )
    d = (
        jnp.zeros(p, dtype=dtype)
        if output_offset is None
        else output_offset.astype(dtype)
    )

    def step(carry, _):
        projected_power, accumulated_offset = carry
        impulse = projected_power @ control  # C A**t B
        projected_power = projected_power @ transition  # C A**(t + 1)
        accumulated_offset = transition @ accumulated_offset + a
        offset = projection @ accumulated_offset + d
        return (projected_power, accumulated_offset), (projected_power, impulse, offset)

    _, (state_response, impulses, offset) = jax.lax.scan(
        step, (projection, jnp.zeros(n, dtype=dtype)), None, length=horizon
    )
    # Form the causal block Toeplitz operator from C A**(t - j) B.
    lag = jnp.arange(horizon)[:, None] - jnp.arange(horizon)[None, :]
    command_response = jnp.where(
        (lag >= 0)[:, :, None, None], impulses[jnp.maximum(lag, 0)], 0
    ).transpose(0, 2, 1, 3)
    return PredictionResponse(state_response, command_response, offset)


def predict_sequence(
    response: PredictionResponse, initial_state: Array, controls: Array
) -> Array:
    """Apply a prepared affine response to one state and control sequence.

    Args:
        response: Result of :func:`prepare_prediction` for horizon ``H``.
        initial_state: State immediately before ``controls[0]``, shape ``(n,)``.
        controls: Complete future input sequence, shape ``(H, m)``. Units and
            discretisation belong to the dynamics model.

    Returns:
        Future outputs with shape ``(H, p)``, excluding the initial output.
        The first is ``C @ (A @ initial_state + B @ controls[0] + a) + d``.
        No covariance or measurement update is computed. Use ``jax.vmap``
        for candidate sequences or initial states, and ``jax.jit`` explicitly
        for compilation.

    Raises:
        ValueError: Response, state or control shapes are incompatible.
    """

    if response.state_response.ndim != 3 or response.state_response.shape[0] < 1:
        raise ValueError("response.state_response must have shape (H, p, n), H > 0")
    horizon, p, n = response.state_response.shape
    if response.command_response.ndim != 4:
        raise ValueError("response.command_response must have shape (H, p, H, m)")
    m = response.command_response.shape[3]
    _check_shape(
        "response.command_response", response.command_response, (horizon, p, horizon, m)
    )
    _check_shape("response.offset", response.offset, (horizon, p))
    _check_shape("initial_state", initial_state, (n,))
    _check_shape("controls", controls, (horizon, m))
    return (
        jnp.einsum("tpn,n->tp", response.state_response, initial_state)
        + jnp.einsum("tpjm,jm->tp", response.command_response, controls)
        + response.offset
    )


def _check_shape(name: str, array: Array, expected: tuple[int, ...]) -> None:
    if array.shape != expected:
        raise ValueError(f"{name} must have shape {expected}; got {array.shape}")


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
