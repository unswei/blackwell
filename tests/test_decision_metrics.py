"""Finite-bank decision metrics, missing truth and JAX transformations."""

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from blackwell import metrics


def regret_inputs():
    return dict(
        reference_costs=jnp.array([5.0, 1.0, 3.0, -100.0, jnp.nan, jnp.inf]),
        reference_accepted=jnp.array([True, True, False, True, True, True]),
        selected_index=jnp.array(0),
        candidate_mask=jnp.array([True, True, True, False, True, True]),
        cost_observed=jnp.array([True, True, True, True, False, True]),
        constraint_observed=jnp.array([True, True, True, True, True, True]),
    )


def assert_same_tree(actual, expected):
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        assert_array_equal(a, b)


def test_regret_uses_supplied_selection_and_feasible_observed_bank():
    result = metrics.selection_regret(**regret_inputs())
    assert result.regret == 4
    assert result.best_reference_cost == 1
    assert result.best_reference_index == 1
    assert result.reference_available
    assert result.selection_available
    assert result.selected_cost_available
    assert result.selected_constraint_observed
    assert result.selected_feasible
    assert result.regret_defined
    assert_same_tree(jax.jit(metrics.selection_regret)(**regret_inputs()), result)


@pytest.mark.parametrize("index", [-1, -2, 3, 6, 100])
def test_invalid_or_padded_selection_is_unavailable(index):
    inputs = regret_inputs() | {"selected_index": jnp.array(index)}
    result = jax.jit(metrics.selection_regret)(**inputs)
    assert result.reference_available
    assert not result.selection_available
    assert not result.selected_constraint_observed
    assert not result.selected_cost_available
    assert not result.selected_feasible
    assert not result.regret_defined
    assert result.regret == 0


def test_infeasibility_missing_constraint_and_missing_cost_are_distinct():
    inputs = regret_inputs() | {"selected_index": 2}
    infeasible = metrics.selection_regret(**inputs)
    assert infeasible.selection_available and infeasible.selected_constraint_observed
    assert infeasible.selected_cost_available and not infeasible.selected_feasible
    assert not infeasible.regret_defined
    unknown = metrics.selection_regret(
        **(
            inputs
            | {"constraint_observed": inputs["constraint_observed"].at[2].set(False)}
        )
    )
    assert unknown.selection_available and not unknown.selected_constraint_observed
    assert not unknown.selected_feasible and not unknown.regret_defined
    missing_cost = metrics.selection_regret(**(inputs | {"selected_index": 4}))
    assert missing_cost.selected_feasible and missing_cost.selected_constraint_observed
    assert not missing_cost.selected_cost_available and not missing_cost.regret_defined


@pytest.mark.parametrize(
    "empty_by",
    ["candidate_mask", "cost_observed", "constraint_observed", "reference_accepted"],
)
def test_empty_reference_has_explicit_finite_placeholders(empty_by):
    inputs = regret_inputs()
    inputs[empty_by] = jnp.zeros(6, dtype=bool)
    result = jax.jit(metrics.selection_regret)(**inputs)
    assert not result.reference_available and not result.regret_defined
    assert result.best_reference_index == -1
    assert result.best_reference_cost == 0 and result.regret == 0


def test_stable_ties_and_finite_sentinel_do_not_choose_padding():
    inputs = regret_inputs()
    for cost in (1.0, jnp.finfo(jnp.float32).max):
        result = metrics.selection_regret(
            **(
                inputs
                | {
                    "reference_costs": jnp.full(6, cost),
                    "candidate_mask": jnp.array(
                        [False, True, True, True, False, False]
                    ),
                    "reference_accepted": jnp.ones(6, dtype=bool),
                    "selected_index": 2,
                }
            )
        )
        assert result.best_reference_index == 1
        assert result.regret_defined and result.regret == 0


def test_nonfinite_missing_costs_do_not_enter_compiled_arithmetic():
    inputs = regret_inputs() | {
        "reference_costs": jnp.array(
            [jnp.nan, jnp.inf, -jnp.inf, 0.0, jnp.nan, jnp.inf]
        )
    }
    with jax.debug_nans(True), jax.debug_infs(True):
        result = jax.jit(metrics.selection_regret)(**inputs)
    assert not result.reference_available and not result.selected_cost_available
    assert result.regret == 0 and not result.regret_defined


def binary_inputs():
    return dict(
        predicted_accepted=jnp.array([True, True, False, False, True, True]),
        reference_accepted=jnp.array([True, False, True, False, False, False]),
        candidate_mask=jnp.array([True, True, True, True, True, False]),
        constraint_observed=jnp.array([True, True, True, True, False, True]),
    )


def test_binary_counts_retain_denominators_and_exclusions():
    result = metrics.binary_decision_counts(**binary_inputs())
    assert result.correct_acceptance == 1 and result.false_acceptance == 1
    assert result.false_rejection == 1 and result.correct_rejection == 1
    assert result.compared_count == 4
    assert result.unobserved_count == 1 and result.excluded_count == 2
    assert result.model_accepted_count == 2 and result.reference_accepted_count == 2
    assert result.false_acceptance_rate == 0.5 and result.false_acceptance_rate_defined
    assert result.false_rejection_rate == 0.5 and result.false_rejection_rate_defined
    assert_same_tree(jax.jit(metrics.binary_decision_counts)(**binary_inputs()), result)


def test_binary_rate_conditioning_and_zero_denominators():
    inputs = binary_inputs() | {
        "predicted_accepted": jnp.array([True, False, False, False, False, False]),
    }
    result = metrics.binary_decision_counts(**inputs)
    assert result.false_acceptance_rate == 0
    assert result.false_acceptance_rate_defined
    assert result.false_rejection_rate == 0.5
    result = metrics.binary_decision_counts(
        **(
            inputs
            | {
                "predicted_accepted": jnp.zeros(6, dtype=bool),
                "reference_accepted": jnp.zeros(6, dtype=bool),
            }
        )
    )
    assert not result.false_acceptance_rate_defined
    assert not result.false_rejection_rate_defined
    assert result.false_acceptance_rate == result.false_rejection_rate == 0
    result = metrics.binary_decision_counts(
        **(
            inputs
            | {
                "constraint_observed": jnp.zeros(6, dtype=bool),
            }
        )
    )
    assert result.compared_count == 0 and result.unobserved_count == 5
    assert not result.false_acceptance_rate_defined
    assert not result.false_rejection_rate_defined


@pytest.mark.parametrize(
    "function,inputs",
    [
        (metrics.selection_regret, regret_inputs),
        (metrics.binary_decision_counts, binary_inputs),
    ],
)
def test_vmap_and_immutable_scalar_pytree_results(function, inputs):
    first = inputs()
    second = first | {"candidate_mask": jnp.zeros_like(first["candidate_mask"])}
    batched = jax.tree.map(lambda a, b: jnp.stack((a, b)), first, second)
    result = jax.jit(jax.vmap(lambda values: function(**values)))(batched)
    for i, values in enumerate((first, second)):
        expected = function(**values)
        assert all(leaf.shape == () for leaf in expected)
        assert_same_tree(jax.tree.map(lambda x, i=i: x[i], result), expected)
        with pytest.raises(AttributeError):
            setattr(expected, expected._fields[0], 2)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_regret_preserves_floating_cost_dtype(dtype):
    original = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", dtype == jnp.float64)
    try:
        inputs = regret_inputs()
        inputs["reference_costs"] = inputs["reference_costs"].astype(dtype)
        result = jax.jit(metrics.selection_regret)(**inputs)
        assert (
            result.regret.dtype == dtype and result.best_reference_cost.dtype == dtype
        )
        assert_allclose(result.regret, 4.0, rtol=0, atol=0)
    finally:
        jax.config.update("jax_enable_x64", original)


@pytest.mark.parametrize(
    "name",
    ["reference_accepted", "candidate_mask", "cost_observed", "constraint_observed"],
)
def test_regret_rejects_mask_shapes_and_types(name):
    for function in (metrics.selection_regret, jax.jit(metrics.selection_regret)):
        with pytest.raises(ValueError, match=name):
            function(**(regret_inputs() | {name: jnp.ones((6, 1), dtype=bool)}))
        with pytest.raises(TypeError, match=name):
            function(**(regret_inputs() | {name: jnp.ones(6)}))


@pytest.mark.parametrize(
    "name",
    [
        "predicted_accepted",
        "reference_accepted",
        "candidate_mask",
        "constraint_observed",
    ],
)
def test_binary_rejects_mask_shapes_and_types(name):
    for function in (
        metrics.binary_decision_counts,
        jax.jit(metrics.binary_decision_counts),
    ):
        with pytest.raises(ValueError, match=name):
            function(**(binary_inputs() | {name: jnp.ones((6, 1), dtype=bool)}))
        with pytest.raises(TypeError, match=name):
            function(**(binary_inputs() | {name: jnp.ones(6)}))


@pytest.mark.parametrize("costs", [jnp.ones((6, 1)), jnp.ones(0)])
def test_regret_rejects_invalid_cost_shapes(costs):
    with pytest.raises(ValueError, match="reference_costs"):
        metrics.selection_regret(**(regret_inputs() | {"reference_costs": costs}))


@pytest.mark.parametrize(
    "index,error", [(jnp.array([0]), ValueError), (1.5, TypeError), (True, TypeError)]
)
def test_regret_rejects_non_scalar_integer_selections(index, error):
    with pytest.raises(error, match="selected_index"):
        metrics.selection_regret(**(regret_inputs() | {"selected_index": index}))
