"""Decision counterexamples, completion windows and reproducible paired runs."""

import runpy
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose, assert_array_equal


@pytest.fixture(scope="module")
def example():
    return runpy.run_path(
        str(
            Path(__file__).parents[1] / "examples/decision_relevant_model_evaluation.py"
        )
    )


@pytest.fixture(scope="module")
def report(example):
    return example["run"]()


def assert_tree_close(actual, expected):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        assert_allclose(a, b, rtol=5e-6, atol=5e-6)


def test_lower_rmse_can_give_worse_decisions_and_error_can_leave_actions_unchanged(
    report,
):
    a = report["records"]["ranking_boundary"]["gain_error"]
    b = report["records"]["ranking_boundary"]["lateral_drift"]
    assert a["prediction_rmse"] < b["prediction_rmse"]
    assert a["selected_index"] == 1 and b["selected_index"] == 2
    assert a["regret"].regret > b["regret"].regret
    assert a["execution"]["timeout"] and b["execution"]["success"]
    assert b["execution"]["arrival_time"] == 3.25
    unchanged = report["records"]["unchanged_action"]
    assert unchanged["gain_error"]["selected_index"] == 2
    assert unchanged["lateral_drift"]["selected_index"] == 2
    assert unchanged["gain_error"]["prediction_rmse"] > 0
    assert unchanged["lateral_drift"]["prediction_rmse"] > 0
    pair = report["paired"]["unchanged_action"]
    assert pair["both_success"] and pair["arrival_difference_defined"]
    assert pair["arrival_difference"] == 0


def test_same_state_replay_selected_execution_and_reference_labels(example, report):
    controls, mask = example["candidate_bank"]()
    model = example["dynamics"]()
    rollout_many = jax.jit(
        jax.vmap(example["rollout"], in_axes=(None, None, 1), out_axes=1)
    )
    for root_id, root in example["roots"]().items():
        first = rollout_many(model, root["initial_state"], controls)
        second = rollout_many(model, root["initial_state"], controls)
        assert first.shape == (example["HORIZON"], mask.size, 4)
        assert_array_equal(first, second)
        a, b = report["records"][root_id].values()
        for name in (
            "reference_costs",
            "cost_observed",
            "failed",
            "observation_end",
            "actual_time",
        ):
            assert_array_equal(a["record"][name], b["record"][name])
        for row in (a, b):
            selected = int(row["selected_index"])
            replay = jax.jit(example["rollout"])(
                model, root["initial_state"], controls[:, selected]
            )
            assert_array_equal(
                replay,
                jax.jit(example["rollout"])(
                    model, root["initial_state"], controls[:, selected]
                ),
            )
            assert_allclose(replay, first[:, selected], rtol=3e-6, atol=1e-7)
            if row["regret"].selected_cost_available:
                cost = example["task_cost"](
                    replay[:, None], controls[:, selected : selected + 1], root["goal"]
                )[0]
                assert_allclose(
                    cost, row["record"]["reference_costs"][selected], atol=1e-7
                )


def test_example_eager_jit_and_vmap_over_roots_match(example, report):
    controls, mask = example["candidate_bank"]()
    roots = list(example["roots"]().values())
    model = example["dynamics"](0.21)
    offset = jnp.zeros(4)
    evaluate = example["evaluate_root"]
    stacked = jax.tree.map(lambda *values: jnp.stack(values), *roots)
    batch = jax.jit(jax.vmap(evaluate, in_axes=(0, None, None, None, None)))(
        stacked, controls, mask, model, offset
    )
    for i, (root_id, root) in enumerate(example["roots"]().items()):
        eager = evaluate(root, controls, mask, model, offset)
        assert_tree_close(eager, report["records"][root_id]["gain_error"])
        assert_tree_close(jax.tree.map(lambda x, i=i: x[i], batch), eager)


def test_collision_truth_is_available_without_total_cost_and_censoring_is_separate(
    report,
):
    failure = report["records"]["physical_failure"]["lateral_drift"]
    assert failure["execution"]["physical_failure"]
    assert not failure["execution"]["censored"]
    assert failure["regret"].selected_constraint_observed
    assert not failure["regret"].selected_feasible
    assert not failure["regret"].selected_cost_available
    assert not failure["regret"].regret_defined
    assert failure["execution"]["restricted_time_defined"]
    assert failure["execution"]["restricted_time"] == 6
    short = report["records"]["short_recording"]["gain_error"]
    assert short["execution"]["censored"]
    assert not short["execution"]["outcome_observed"]
    assert not short["execution"]["restricted_time_defined"]
    assert not short["regret"].selected_constraint_observed
    assert short["reserve_counts"].compared_count == 0
    assert short["reserve_counts"].unobserved_count == 4
    pair = report["paired"]["short_recording"]
    assert pair["unobserved_pair"] and not pair["neither"]
    assert not pair["restricted_time_difference_defined"]


def test_an_event_before_censoring_does_not_prove_complete_execution_success(example):
    root = example["roots"]()["unchanged_action"] | {"observation_steps": jnp.array(16)}
    controls, mask = example["candidate_bank"]()
    row = example["evaluate_root"](
        root, controls, mask, example["dynamics"](), jnp.zeros(4)
    )
    selected = int(row["selected_index"])
    assert row["record"]["actual_event"][selected]
    assert row["execution"]["censored"]
    assert not row["execution"]["success"]
    assert not row["execution"]["outcome_observed"]
    assert not row["execution"]["restricted_time_defined"]


def test_completion_distinguishes_windows_timing_censoring_and_failure(example):
    # Exact, early but no longer at the target, delayed, unobservable,
    # observed non-arrival, failed, and padding.
    claimed = jnp.ones(7, dtype=bool)
    predicted_time = jnp.array([2.0, 3.0, 2.0, 5.0, 2.0, 2.0, 2.0])
    confirmed = jnp.array([True, False, False, False, False, False, True])
    event = jnp.array([True, True, True, False, False, False, True])
    actual_time = jnp.array([2.0, 1.0, 3.0, 0.0, 0.0, 0.0, 2.0])
    end = jnp.array([6.0, 6.0, 6.0, 3.0, 6.0, 1.0, 6.0])
    failed = jnp.array([False, False, False, False, False, True, False])
    mask = jnp.array([True, True, True, True, True, True, False])
    args = (claimed, predicted_time, confirmed, event, actual_time, end, failed, mask)
    compare = example["completion_comparison"]
    result = compare(*args)
    assert_array_equal(
        result["observable_claim"], [True, True, True, False, True, False, False]
    )
    assert_array_equal(
        result["confirmed_claim"], [True, False, False, False, False, False, False]
    )
    assert_array_equal(
        result["early"], [False, True, False, False, False, False, False]
    )
    assert_array_equal(
        result["delayed"], [False, False, True, False, False, False, False]
    )
    assert_array_equal(
        result["right_censored"], [False, False, False, True, False, False, False]
    )
    assert_array_equal(
        result["unconfirmed_through_horizon"],
        [False, False, False, False, True, False, False],
    )
    assert result["failed"][5] and not result["right_censored"][5]
    assert_array_equal(result["timing_error"], [0.0, -2.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    assert_tree_close(jax.jit(compare)(*args), result)
    batched = jax.jit(jax.vmap(compare))(*(jnp.stack((x, x)) for x in args))
    assert_tree_close(jax.tree.map(lambda x: x[0], batched), result)


def test_completion_requires_dwell_and_checks_actual_claimed_window(example):
    trajectory = jnp.zeros((8, 1, 4))
    goal = jnp.zeros(2)
    trajectory = trajectory.at[0, 0, 0].set(
        1.0
    )  # One near endpoint alone is insufficient.
    trajectory = trajectory.at[5:, 0, 0].set(1.0)  # Leave after the first event.
    windows = jax.jit(example["completion_windows"])(trajectory, goal)
    assert_array_equal(
        windows[:, 0], [False, False, False, True, True, False, False, False]
    )
    event, time = example["first_event"](windows)
    assert event[0] and time[0] == 1.0
    result = example["completion_comparison"](
        jnp.array([True]),
        jnp.array([2.0]),
        windows[7],
        event,
        time,
        jnp.array([2.0]),
        jnp.array([False]),
        jnp.array([True]),
    )
    assert result["early"][0] and result["missed_window"][0]
    assert not result["confirmed_claim"][0]


def test_filtering_loss_and_empty_survivor_set_are_explicit(example, report):
    clearance = report["records"]["clearance"]["gain_error"]
    assert clearance["filtering"]["defined"]
    assert_allclose(
        clearance["filtering"]["loss"], clearance["regret"].regret, atol=1e-7
    )
    root = example["roots"]()["ranking_boundary"]
    # Make every model forecast violate the reserve while leaving reference truth fixed.
    root = root | {"obstacle": jnp.array([0.0, 0.2]), "radius": jnp.array(0.05)}
    controls, mask = example["candidate_bank"]()
    result = example["evaluate_root"](
        root, controls, mask, example["dynamics"](), jnp.array([0.0, 0.2, 0.0, 0.0])
    )
    assert result["regret"].reference_available
    assert not result["filtering"]["surviving_reference_available"]
    assert not result["filtering"]["defined"]
    assert result["execution"]["no_selection"]


def test_paired_counts_and_difference_signs_use_observed_pairs(example):
    known = jnp.array([True, True, True, True, False])
    defaults = {
        name: jnp.zeros(5, dtype=bool)
        for name in (
            "physical_failure",
            "censored",
            "timeout",
            "no_selection",
        )
    }
    defaults |= dict(outcome_observed=known, restricted_time_defined=known)
    a = defaults | dict(
        success=jnp.array([True, True, False, False, False]),
        arrival_time=jnp.array([1.0, 2.0, 0.0, 0.0, 0.0]),
        restricted_time=jnp.array([1.0, 2.0, 6.0, 6.0, 0.0]),
    )
    b = defaults | dict(
        success=jnp.array([True, False, True, False, False]),
        arrival_time=jnp.array([1.5, 0.0, 3.0, 0.0, 0.0]),
        restricted_time=jnp.array([1.5, 6.0, 3.0, 6.0, 0.0]),
    )
    result = jax.jit(example["paired_outcomes"])(a, b)
    for index, name in enumerate(
        ("both_success", "a_only", "b_only", "neither", "unobserved_pair")
    ):
        assert_array_equal(result[name], jnp.arange(5) == index)
    assert_array_equal(result["arrival_difference"], [0.5, 0.0, 0.0, 0.0, 0.0])
    assert_array_equal(
        result["arrival_difference_defined"], [True, False, False, False, False]
    )
    assert_array_equal(result["restricted_time_difference"], [0.5, 4.0, -3.0, 0.0, 0.0])
    assert_array_equal(result["restricted_time_difference_defined"], known)
