"""A private linear-Gaussian filtering experiment.

This module exercises the smallest useful JAX interface for a Gaussian belief
and a stochastic state-space model.  It intentionally remains private while
the project's public estimation API is being designed.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array


class GaussianBelief(NamedTuple):
    """Gaussian belief represented in Euclidean state coordinates.

    Attributes:
        mean: State mean with shape ``(state_dim,)``.
        covariance: State covariance with shape ``(state_dim, state_dim)``.
    """

    mean: Array
    covariance: Array


class LinearGaussianModel(NamedTuple):
    """Discrete-time linear state-space model.

    The model is defined by ``x' = transition @ x + control @ u + w`` and
    ``z = observation @ x + v``, where ``w`` and ``v`` have the supplied
    covariances.  A zero-width control matrix supports models without inputs.
    """

    transition: Array
    control: Array
    process_covariance: Array
    observation: Array
    measurement_covariance: Array


def predict(
    belief: GaussianBelief,
    model: LinearGaussianModel,
    control: Array,
) -> GaussianBelief:
    """Apply the Kalman prediction step.

    ``control`` has shape ``(control_dim,)``; it may be empty when the model
    has no control input.  The function is pure and compatible with
    :func:`jax.jit` and :func:`jax.vmap`.
    """

    mean = model.transition @ belief.mean + model.control @ control
    covariance = (
        model.transition @ belief.covariance @ model.transition.T
        + model.process_covariance
    )
    return GaussianBelief(mean=mean, covariance=_symmetrise(covariance))


def update(
    belief: GaussianBelief,
    model: LinearGaussianModel,
    measurement: Array,
) -> GaussianBelief:
    """Apply the Kalman measurement update using the Joseph covariance form."""

    innovation = measurement - model.observation @ belief.mean
    innovation_covariance = (
        model.observation @ belief.covariance @ model.observation.T
        + model.measurement_covariance
    )
    gain = jnp.linalg.solve(
        innovation_covariance,
        model.observation @ belief.covariance,
    ).T

    mean = belief.mean + gain @ innovation
    identity = jnp.eye(belief.mean.shape[0], dtype=belief.covariance.dtype)
    residual = identity - gain @ model.observation
    covariance = (
        residual @ belief.covariance @ residual.T
        + gain @ model.measurement_covariance @ gain.T
    )
    return GaussianBelief(mean=mean, covariance=_symmetrise(covariance))


def step(
    belief: GaussianBelief,
    model: LinearGaussianModel,
    control: Array,
    measurement: Array,
) -> GaussianBelief:
    """Perform one prediction followed by one measurement update."""

    return update(predict(belief, model, control), model, measurement)


def _symmetrise(matrix: Array) -> Array:
    """Limit numerical skew in a covariance matrix."""

    return (matrix + matrix.T) / 2
