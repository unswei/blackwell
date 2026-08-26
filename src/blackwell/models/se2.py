"""SE(2) body-frame motion models."""

from __future__ import annotations

from typing import NamedTuple

from jax import Array

from blackwell.spaces import se2 as space


class BodyMotion(NamedTuple):
    """Body-frame SE(2) motion with additive local process uncertainty."""

    process_covariance: Array


def propagate(state: Array, control: Array, model: BodyMotion) -> Array:
    """Propagate a pose by a body-frame SE(2) tangent control increment."""

    del model
    return space.retract(state, control)


def transition_jacobian(state: Array, control: Array, model: BodyMotion) -> Array:
    """Return the tangent transition Jacobian for right SE(2) retraction."""

    del state, model
    return space.adjoint(space.inverse(space.exp(control)))


def process_covariance(state: Array, control: Array, model: BodyMotion) -> Array:
    """Return process covariance in the post-motion local tangent coordinates."""

    del state, control
    return model.process_covariance
