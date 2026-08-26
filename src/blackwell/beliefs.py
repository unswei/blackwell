"""Immutable belief containers used by Blackwell filters."""

from __future__ import annotations

from typing import NamedTuple

from jax import Array


class GaussianBelief(NamedTuple):
    """A Gaussian belief with covariance in local tangent coordinates.

    ``mean`` is a state in its accompanying state space. ``covariance`` has
    shape ``(tangent_dim, tangent_dim)`` at that mean.
    """

    mean: Array
    covariance: Array


class ParticleBelief(NamedTuple):
    """A normalised weighted collection of particles.

    ``particles`` has leading shape ``(particle_count,)`` and ``weights`` has
    shape ``(particle_count,)``.
    """

    particles: Array
    weights: Array
