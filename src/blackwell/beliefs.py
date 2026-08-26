"""Immutable belief containers used by Blackwell filters."""

from __future__ import annotations

from typing import NamedTuple

from jax import Array


class GaussianBelief(NamedTuple):
    """A Gaussian belief with covariance in local tangent coordinates.

    The container deliberately does not retain a state-space object. Pass the
    matching operations separately to keep the belief an uncomplicated JAX
    PyTree.

    Attributes:
        mean: State array in the accompanying state space. For SE(2), this is
            ``[x, y, heading]`` with shape ``(3,)``.
        covariance: Symmetric local covariance with shape
            ``(tangent_dim, tangent_dim)`` at ``mean``. For SE(2), its axes are
            body-frame ``[forward, lateral, turn]``.
    """

    mean: Array
    covariance: Array


class ParticleBelief(NamedTuple):
    """A normalised weighted collection of particles.

    Attributes:
        particles: State samples with shape
            ``(particle_count, *state_shape)``.
        weights: Non-negative, normalised sample weights with shape
            ``(particle_count,)``.

    Note:
        The container does not enforce normalisation at construction. Filter
        operations assume finite weights that sum to one.
    """

    particles: Array
    weights: Array
