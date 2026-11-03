"""Numerical estimation-error, consistency and finite-candidate decision metrics."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array


class SelectionRegret(NamedTuple):
    """Finite-bank selection regret for one root, with explicit availability.

    All fields are scalar JAX arrays. Numerical placeholders must be read with
    their flags; an undefined regret of zero does not mean an optimal choice.

    Attributes:
        regret: Selected reference cost minus the bank's best feasible cost;
            zero when undefined.
        best_reference_cost: Best observed finite feasible cost, or zero when
            ``reference_available`` is false.
        best_reference_index: First candidate attaining that cost, or ``-1``
            when no reference candidate is available.
        reference_available: At least one occupied candidate has observed
            finite cost and observed acceptable constraints.
        selection_available: The supplied index is in range and occupied.
        selected_cost_available: Its reference cost is observed and finite.
        selected_constraint_observed: Its reference constraint label is known.
        selected_feasible: The selection has an observed acceptable constraint
            label. False with an unobserved label does not establish failure.
        regret_defined: The selection belongs to the feasible reference set.
    """

    regret: Array
    best_reference_cost: Array
    best_reference_index: Array
    reference_available: Array
    selection_available: Array
    selected_cost_available: Array
    selected_constraint_observed: Array
    selected_feasible: Array
    regret_defined: Array


class BinaryDecisionCounts(NamedTuple):
    """Counts and conditional error rates for one root's binary decisions.

    All fields are scalar JAX arrays. Counts are integers. Only occupied slots
    with observed reference labels contribute to the confusion counts or rate
    denominators. Zero is a placeholder for an undefined rate.

    Attributes:
        correct_acceptance: Model and reference both accept.
        false_acceptance: Model accepts and reference rejects.
        false_rejection: Model rejects and reference accepts.
        correct_rejection: Model and reference both reject.
        compared_count: Number of observed, occupied reference labels.
        unobserved_count: Occupied candidates without reference labels.
        excluded_count: All slots excluded from comparison, including padding
            and occupied candidates without reference labels.
        model_accepted_count: Observed candidates accepted by the model.
        reference_accepted_count: Observed candidates accepted by the reference.
        false_acceptance_rate: False accepts divided by model accepts.
        false_acceptance_rate_defined: The model-accepted denominator is nonzero.
        false_rejection_rate: False rejects divided by reference accepts.
        false_rejection_rate_defined: The reference-accepted denominator is nonzero.
    """

    correct_acceptance: Array
    false_acceptance: Array
    false_rejection: Array
    correct_rejection: Array
    compared_count: Array
    unobserved_count: Array
    excluded_count: Array
    model_accepted_count: Array
    reference_accepted_count: Array
    false_acceptance_rate: Array
    false_acceptance_rate_defined: Array
    false_rejection_rate: Array
    false_rejection_rate_defined: Array


def selection_regret(
    reference_costs: Array,
    reference_accepted: Array,
    selected_index: Array,
    *,
    candidate_mask: Array,
    cost_observed: Array,
    constraint_observed: Array,
) -> SelectionRegret:
    """Evaluate the caller's selection against the feasible finite bank.

    This function does not select an action from predicted costs. The caller's
    actual rule supplies ``selected_index``. Infeasible selections remain
    separate outcomes rather than receiving an arbitrary large penalty.

    Args:
        reference_costs: Real costs with shape ``(K,)``, ``K > 0``; lower is
            better. Only observed finite costs enter the reference set.
        reference_accepted: Boolean reference constraint labels, shape ``(K,)``.
        selected_index: Scalar integer; ``-1`` denotes no selection. Every
            negative, out-of-range or padded index is unavailable.
        candidate_mask: Boolean occupied-slot mask, shape ``(K,)``. Keep model
            acceptance separate from this mask when evaluating a selection.
        cost_observed: Boolean availability of reference costs, shape ``(K,)``.
        constraint_observed: Boolean availability of reference constraint
            labels, shape ``(K,)``; independent of cost availability.

    Returns:
        Scalar results in an immutable PyTree. Ties use stable candidate order.
        Undefined costs/regret are zero, the undefined best index is ``-1``,
        and availability/defined flags distinguish them from measured results.
        A bank with all slots masked is supported. Use explicit ``jax.jit``
        and ``jax.vmap`` for compilation and independent roots.

    Raises:
        ValueError: Shapes are incompatible or bank capacity is zero.
        TypeError: Masks are not Boolean, costs are not real numbers, or the
            selection is not a scalar integer.
    """

    costs = jnp.asarray(reference_costs)
    if costs.ndim != 1 or costs.shape[0] < 1:
        raise ValueError("reference_costs must have shape (K,), K > 0")
    if not (
        jnp.issubdtype(costs.dtype, jnp.floating)
        or jnp.issubdtype(costs.dtype, jnp.integer)
    ):
        raise TypeError("reference_costs must contain real numbers")
    shape = costs.shape
    accepted = _decision_mask("reference_accepted", reference_accepted, shape)
    occupied = _decision_mask("candidate_mask", candidate_mask, shape)
    cost_known = _decision_mask("cost_observed", cost_observed, shape)
    constraint_known = _decision_mask("constraint_observed", constraint_observed, shape)
    selected = jnp.asarray(selected_index)
    if selected.shape != ():
        raise ValueError("selected_index must be a scalar")
    if not jnp.issubdtype(selected.dtype, jnp.integer):
        raise TypeError("selected_index must be an integer")

    costs = costs.astype(jnp.result_type(costs, 0.0))
    finite = jnp.isfinite(costs)
    safe_costs = jnp.where(finite, costs, 0)
    eligible = occupied & cost_known & finite & constraint_known & accepted
    reference_available = jnp.any(eligible)
    # Finite sentinels avoid inf/nan reductions even for an empty reference set.
    minimum = jnp.min(jnp.where(eligible, safe_costs, jnp.finfo(costs.dtype).max))
    best_index = jnp.argmax(eligible & (safe_costs == minimum))
    best_cost = jnp.where(reference_available, minimum, 0)
    index = jnp.clip(selected, 0, shape[0] - 1)
    selection_available = (selected >= 0) & (selected < shape[0]) & occupied[index]
    selected_cost_available = selection_available & cost_known[index] & finite[index]
    selected_constraint_observed = selection_available & constraint_known[index]
    selected_feasible = selected_constraint_observed & accepted[index]
    defined = reference_available & selected_cost_available & selected_feasible
    regret = jnp.where(defined, safe_costs[index], 0) - jnp.where(defined, best_cost, 0)
    return SelectionRegret(
        regret, best_cost, jnp.where(reference_available, best_index, -1),
        reference_available, selection_available, selected_cost_available,
        selected_constraint_observed, selected_feasible, defined,
    )


def binary_decision_counts(
    predicted_accepted: Array,
    reference_accepted: Array,
    *,
    candidate_mask: Array,
    constraint_observed: Array,
) -> BinaryDecisionCounts:
    """Compare matching binary criteria while retaining counts and denominators.

    Use the same criterion and horizon for model and reference labels. Call
    separately for physical feasibility and a reserve-margin criterion, or for
    completion-at-horizon labels. Thresholds and their units belong to the
    experiment, not this metric.

    Args:
        predicted_accepted: Boolean model labels with shape ``(K,)``, ``K > 0``.
        reference_accepted: Boolean reference labels with the same shape.
        candidate_mask: Boolean occupied-slot mask with shape ``(K,)``.
        constraint_observed: Boolean reference-label availability, shape
            ``(K,)``. For completion labels, pass completion availability here.

    Returns:
        Scalar counts, conditional error rates and defined flags in an
        immutable PyTree. False acceptance is conditioned on observed model
        accepts; false rejection is conditioned on observed reference accepts.
        A zero denominator gives a zero placeholder with a false defined flag.
        Counts can be summed across roots before deriving pooled rates; do not
        treat candidates within a root as independent statistical trials.

    Raises:
        ValueError: Shapes are incompatible or bank capacity is zero.
        TypeError: Any label or mask is not Boolean.
    """

    predicted = jnp.asarray(predicted_accepted)
    if predicted.ndim != 1 or predicted.shape[0] < 1:
        raise ValueError("predicted_accepted must have shape (K,), K > 0")
    shape = predicted.shape
    predicted = _decision_mask("predicted_accepted", predicted, shape)
    reference = _decision_mask("reference_accepted", reference_accepted, shape)
    occupied = _decision_mask("candidate_mask", candidate_mask, shape)
    known = _decision_mask("constraint_observed", constraint_observed, shape)
    compared = occupied & known
    correct_acceptance = jnp.sum(compared & predicted & reference)
    false_acceptance = jnp.sum(compared & predicted & ~reference)
    false_rejection = jnp.sum(compared & ~predicted & reference)
    correct_rejection = jnp.sum(compared & ~predicted & ~reference)
    model_count = correct_acceptance + false_acceptance
    reference_count = correct_acceptance + false_rejection
    return BinaryDecisionCounts(
        correct_acceptance, false_acceptance, false_rejection, correct_rejection,
        jnp.sum(compared), jnp.sum(occupied & ~known), jnp.sum(~compared),
        model_count, reference_count,
        false_acceptance / jnp.maximum(model_count, 1), model_count > 0,
        false_rejection / jnp.maximum(reference_count, 1), reference_count > 0,
    )


def _decision_mask(name: str, value: Array, shape: tuple[int, ...]) -> Array:
    mask = jnp.asarray(value)
    if mask.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {mask.shape}")
    if mask.dtype != jnp.bool_:
        raise TypeError(f"{name} must be Boolean")
    return mask


def root_mean_square_error(errors: Array, axis: int | tuple[int, ...] = 0) -> Array:
    """Return component-wise root mean square error.

    Args:
        errors: Error array. Remaining axes are preserved after reduction.
        axis: One axis or a tuple of axes over which samples are averaged.

    Returns:
        Square root of the mean squared error over ``axis``.
    """

    return jnp.sqrt(jnp.mean(errors**2, axis=axis))


def position_rmse(errors: Array, axis: int | tuple[int, ...] = 0) -> Array:
    """Return planar position RMSE from errors with position first.

    Args:
        errors: Tangent errors with shape ``(..., tangent_dim)`` and planar
            position in the first two coordinates.
        axis: One axis or a tuple of axes over which samples are averaged.

    Returns:
        Root mean squared Euclidean position norm.
    """

    squared_position_error = jnp.sum(errors[..., :2] ** 2, axis=-1)
    return jnp.sqrt(jnp.mean(squared_position_error, axis=axis))


def normalised_estimation_error_squared(errors: Array, covariances: Array) -> Array:
    """Return tangent-space NEES for matching errors and covariances.

    Args:
        errors: Local errors with shape ``(..., tangent_dim)``.
        covariances: Matching nonsingular covariances with shape
            ``(..., tangent_dim, tangent_dim)``.

    Returns:
        NEES values with shape ``errors.shape[:-1]``.
    """

    covariance_inverse_errors = jnp.linalg.solve(covariances, errors[..., None])[
        ..., 0
    ]
    return jnp.sum(errors * covariance_inverse_errors, axis=-1)


def mean_nees(
    errors: Array,
    covariances: Array,
    axis: int | tuple[int, ...] = 0,
) -> Array:
    """Return NEES averaged over one or more sample axes.

    Args:
        errors: Local errors with shape ``(..., tangent_dim)``.
        covariances: Matching nonsingular local covariances.
        axis: One axis or a tuple of axes over which NEES is averaged.

    Returns:
        Mean normalised estimation error squared over ``axis``.
    """

    return jnp.mean(normalised_estimation_error_squared(errors, covariances), axis=axis)
