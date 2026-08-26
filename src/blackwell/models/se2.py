"""SE(2) body-frame motion models."""

from __future__ import annotations

from typing import NamedTuple

from jax import Array

from blackwell.spaces import se2 as space


class BodyMotion(NamedTuple):
    """Body-frame SE(2) motion with additive local process uncertainty.

    Attributes:
        process_covariance: Covariance with shape ``(3, 3)`` in post-motion
            body tangent coordinates ``[forward, lateral, turn]``.
    """

    process_covariance: Array


def propagate(state: Array, control: Array, model: BodyMotion) -> Array:
    """Propagate a pose by a body-frame tangent control increment.

    Args:
        state: SE(2) pose ``[x, y, heading]``.
        control: Body tangent increment ``[forward, lateral, turn]``.
        model: Motion covariance parameters; unused by deterministic propagation.

    Returns:
        Retracted SE(2) pose with its heading wrapped to ``[-pi, pi]``.
    """

    del model
    return space.retract(state, control)


def transition_jacobian(state: Array, control: Array, model: BodyMotion) -> Array:
    """Return the tangent transition Jacobian for right SE(2) retraction.

    Args:
        state: Current pose; the Jacobian is pose-independent under this
            right-invariant convention.
        control: Body tangent control increment.
        model: Motion covariance parameters; unused by the Jacobian.

    Returns:
        A ``(3, 3)`` tangent transition matrix.
    """

    del state, model
    return space.adjoint(space.inverse(space.exp(control)))


def process_covariance(state: Array, control: Array, model: BodyMotion) -> Array:
    """Return process covariance in post-motion tangent coordinates.

    Args:
        state: Current pose; unused because covariance is constant.
        control: Current control; unused because covariance is constant.
        model: Body-motion parameters.

    Returns:
        ``model.process_covariance`` with shape ``(3, 3)``.
    """

    del state, control
    return model.process_covariance
