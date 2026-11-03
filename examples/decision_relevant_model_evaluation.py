"""Compare forecast accuracy and decisions on a shared planar candidate bank.

This deterministic, open-loop experiment supplies its own task and selector.
Only selection regret and binary comparisons come from Blackwell's metric core.
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp

from blackwell import metrics
from blackwell.models import linear

DT = 0.25
HORIZON = 24
HOLD_STEPS = 4
DWELL_STEPS = 3
POSITION_TOLERANCE = 0.10
SPEED_TOLERANCE = 0.04
RESERVE = 0.03
TIMING_TOLERANCE = 0.25


def dynamics(gain=0.20):
    """State [px, py, vx, vy]; inputs change velocity before position advances."""
    identity = jnp.eye(2)
    return linear.LinearDynamics(
        jnp.block(
            [[identity, DT * 0.65 * identity], [jnp.zeros((2, 2)), 0.65 * identity]]
        ),
        jnp.concatenate((DT * gain * identity, gain * identity)),
        jnp.zeros((4, 4)),
    )


def candidate_bank():
    """Four held forward inputs followed by neutral, plus one padded slot."""
    amplitudes = jnp.array([0.6, 0.9, 1.2, 1.5, 0.0])
    commands = jnp.stack((amplitudes, jnp.zeros_like(amplitudes)), axis=-1)
    controls = (jnp.arange(HORIZON) < HOLD_STEPS)[:, None, None] * commands[None]
    return controls, jnp.array([True, True, True, True, False])


def roots():
    """Designed cases, not a sampled deployment distribution."""
    base = dict(
        initial_state=jnp.zeros(4),
        goal=jnp.array([0.62, 0.0]),
        obstacle=jnp.array([2.0, 2.0]),
        radius=jnp.array(0.045),
        observation_steps=jnp.array(HORIZON),
    )
    return {
        "ranking_boundary": base,
        "unchanged_action": base | {"goal": jnp.array([0.70, 0.0])},
        "clearance": base | {"obstacle": jnp.array([0.59, 0.0])},
        "physical_failure": base
        | {
            "goal": jnp.array([0.70, 0.0]),
            "obstacle": jnp.array([0.68, 0.0]),
        },
        "short_recording": base | {"observation_steps": jnp.array(10)},
    }


def rollout(model, initial_state, controls):
    def step(state, control):
        state = linear.propagate(state, control, model)
        return state, state

    return jax.lax.scan(step, initial_state, controls)[1]


def completion_windows(trajectory, goal):
    """True at the end of each fully satisfied dwell window, time on axis zero."""
    near = jnp.linalg.norm(trajectory[..., :2] - goal, axis=-1) <= POSITION_TOLERANCE
    rest = jnp.linalg.norm(trajectory[..., 2:], axis=-1) <= SPEED_TOLERANCE

    def step(count, satisfied):
        count = jnp.where(satisfied, count + 1, 0)
        return count, count >= DWELL_STEPS

    return jax.lax.scan(step, jnp.zeros(near.shape[1], dtype=int), near & rest)[1]


def first_event(windows):
    event = jnp.any(windows, axis=0)
    time = jnp.where(event, (jnp.argmax(windows, axis=0) + 1) * DT, 0.0)
    return event, time


def completion_comparison(
    claimed,
    predicted_time,
    window_confirmed,
    actual_event,
    actual_time,
    observation_end,
    failed,
    candidate_mask,
):
    """Example-local event timing and censoring; all times are seconds."""
    claimed = claimed & candidate_mask
    observable = claimed & (predicted_time <= observation_end)
    timing_defined = claimed & actual_event & candidate_mask
    error = jnp.where(timing_defined, actual_time, 0) - jnp.where(
        timing_defined, predicted_time, 0
    )
    return dict(
        claimed=claimed,
        observable_claim=observable,
        confirmed_claim=observable & window_confirmed,
        missed_window=observable & ~window_confirmed,
        unobservable_claim=claimed & ~observable,
        early=timing_defined & (error < -TIMING_TOLERANCE),
        delayed=timing_defined & (error > TIMING_TOLERANCE),
        on_time=timing_defined & (jnp.abs(error) <= TIMING_TOLERANCE),
        timing_error=error,
        timing_defined=timing_defined,
        unconfirmed_through_horizon=claimed
        & ~actual_event
        & ~failed
        & (observation_end >= HORIZON * DT),
        right_censored=claimed
        & ~actual_event
        & ~failed
        & (observation_end < HORIZON * DT),
        failed=candidate_mask & failed,
    )


def task_cost(trajectory, controls, goal):
    return jnp.sum((trajectory[-1, :, :2] - goal) ** 2, axis=-1) + 0.002 * jnp.sum(
        controls**2, axis=(0, 2)
    )


def select(costs, accepted, candidate_mask):
    """The declared caller rule: minimum finite cost among reserve accepts."""
    eligible = accepted & candidate_mask & jnp.isfinite(costs)
    index = jnp.argmin(jnp.where(eligible, costs, jnp.finfo(costs.dtype).max))
    return jnp.where(jnp.any(eligible), index, -1)


def evaluate_root(root, controls, candidate_mask, model, state_offset):
    """Evaluate one model/root, keeping observation masks specific to each metric."""
    reference = jax.vmap(rollout, in_axes=(None, None, 1), out_axes=1)(
        dynamics(), root["initial_state"], controls
    )
    response = linear.prepare_prediction(model, HORIZON, state_offset=state_offset)
    predicted = jax.vmap(linear.predict_sequence, in_axes=(None, None, 1), out_axes=1)(
        response, root["initial_state"], controls
    )
    steps = jnp.arange(1, HORIZON + 1)[:, None]
    clearance = (
        jnp.linalg.norm(reference[..., :2] - root["obstacle"], axis=-1) - root["radius"]
    )
    recording = steps <= root["observation_steps"]
    collisions = (clearance < 0) & recording
    failed = jnp.any(collisions, axis=0)
    failure_step = jnp.argmax(collisions, axis=0) + 1
    end_step = jnp.where(failed, failure_step, root["observation_steps"])
    observed = steps <= end_step
    full = end_step == HORIZON
    physical_observed = failed | full
    physical_accepted = ~failed
    reserve_violation = jnp.any((clearance < RESERVE) & observed, axis=0)
    reserve_observed = reserve_violation | full
    reserve_accepted = ~reserve_violation
    cost_observed = full & ~failed
    reference_costs = task_cost(reference, controls, root["goal"])
    predicted_costs = task_cost(predicted, controls, root["goal"])
    predicted_clearance = (
        jnp.linalg.norm(predicted[..., :2] - root["obstacle"], axis=-1) - root["radius"]
    )
    predicted_physical = jnp.all(predicted_clearance >= 0, axis=0)
    predicted_accepted = jnp.all(predicted_clearance >= RESERVE, axis=0)
    selected = select(predicted_costs, predicted_accepted, candidate_mask)
    regret = metrics.selection_regret(
        reference_costs,
        reserve_accepted,
        selected,
        candidate_mask=candidate_mask,
        cost_observed=cost_observed,
        constraint_observed=reserve_observed,
    )
    survivors = metrics.selection_regret(
        reference_costs,
        reserve_accepted,
        -1,
        candidate_mask=candidate_mask & predicted_accepted,
        cost_observed=cost_observed,
        constraint_observed=reserve_observed,
    )
    filtering_defined = regret.reference_available & survivors.reference_available
    filtering_loss = jnp.where(
        filtering_defined, survivors.best_reference_cost, 0
    ) - jnp.where(filtering_defined, regret.best_reference_cost, 0)

    predicted_windows = completion_windows(predicted, root["goal"])
    # Completion cannot be established at or after a terminating collision.
    alive = observed & (~failed | (steps < failure_step))
    reference_windows = completion_windows(reference, root["goal"]) & alive
    claimed, predicted_time = first_event(predicted_windows)
    actual_event, actual_time = first_event(reference_windows)
    claimed_step = jnp.clip(
        jnp.rint(predicted_time / DT).astype(int) - 1, 0, HORIZON - 1
    )
    # Check the entire actual dwell window ending at the claimed step.
    window_confirmed = jnp.take_along_axis(
        reference_windows, claimed_step[None], axis=0
    )[0]
    completion = completion_comparison(
        claimed,
        predicted_time,
        window_confirmed,
        actual_event,
        actual_time,
        end_step * DT,
        failed,
        candidate_mask,
    )
    completion_observed = actual_event | full | failed
    observed_slots = observed & candidate_mask
    squared_error = jnp.sum((predicted[..., :2] - reference[..., :2]) ** 2, axis=-1)
    rmse = jnp.sqrt(
        jnp.sum(jnp.where(observed_slots, squared_error, 0))
        / jnp.maximum(jnp.sum(observed_slots), 1)
    )
    selected_slot = jnp.clip(selected, 0, candidate_mask.shape[0] - 1)
    success = (
        regret.selection_available
        & actual_event[selected_slot]
        & ~failed[selected_slot]
        & full[selected_slot]
    )
    failure = regret.selection_available & failed[selected_slot]
    # A non-arrival gets the declared timeout only in this separate summary.
    arrival_time = jnp.where(success, actual_time[selected_slot], 0)
    restricted_defined = regret.selection_available & (
        success | failure | full[selected_slot]
    )
    return dict(
        prediction_rmse=rmse,
        selected_index=selected,
        regret=regret,
        physical_counts=metrics.binary_decision_counts(
            predicted_physical,
            physical_accepted,
            candidate_mask=candidate_mask,
            constraint_observed=physical_observed,
        ),
        reserve_counts=metrics.binary_decision_counts(
            predicted_accepted,
            reserve_accepted,
            candidate_mask=candidate_mask,
            constraint_observed=reserve_observed,
        ),
        reserve_vs_physical_counts=metrics.binary_decision_counts(
            predicted_accepted,
            physical_accepted,
            candidate_mask=candidate_mask,
            constraint_observed=physical_observed,
        ),
        completion_counts=metrics.binary_decision_counts(
            claimed,
            actual_event,
            candidate_mask=candidate_mask,
            constraint_observed=completion_observed,
        ),
        filtering=dict(
            loss=filtering_loss,
            defined=filtering_defined,
            surviving_reference_available=survivors.reference_available,
        ),
        completion=completion,
        execution=dict(
            success=success,
            physical_failure=failure,
            no_selection=~regret.selection_available,
            outcome_observed=~regret.selection_available
            | success
            | failure
            | full[selected_slot],
            timeout=regret.selection_available
            & full[selected_slot]
            & ~success
            & ~failure,
            censored=regret.selection_available
            & ~success
            & ~failure
            & ~full[selected_slot],
            arrival_time=arrival_time,
            arrival_observed=success,
            restricted_time=jnp.where(
                restricted_defined, jnp.where(success, arrival_time, HORIZON * DT), 0
            ),
            restricted_time_defined=restricted_defined,
        ),
        record=dict(
            predicted_costs=predicted_costs,
            reference_costs=jnp.where(cost_observed, reference_costs, 0),
            cost_observed=cost_observed,
            physical_observed=physical_observed,
            reserve_observed=reserve_observed,
            completion_observed=completion_observed,
            predicted_accepted=predicted_accepted,
            reference_accepted=reserve_accepted,
            reference_physical_accepted=physical_accepted,
            predicted_physical_accepted=predicted_physical,
            predicted_time=predicted_time,
            actual_event=actual_event,
            actual_time=actual_time,
            observation_end=end_step * DT,
            failed=failed,
            requested_controls=jnp.where(
                regret.selection_available, controls[:, selected_slot], 0
            ),
            executed_controls=jnp.where(
                regret.selection_available & observed[:, selected_slot, None],
                controls[:, selected_slot],
                0,
            ),
            executed_control_observed=regret.selection_available
            & observed[:, selected_slot],
        ),
    )


def paired_outcomes(a, b):
    """Pair root-level executions; B - A is valid only for both observed arrivals."""
    both = a["success"] & b["success"]
    comparable = a["outcome_observed"] & b["outcome_observed"]
    restricted_defined = a["restricted_time_defined"] & b["restricted_time_defined"]
    return dict(
        a_success=a["success"],
        b_success=b["success"],
        both_success=both,
        a_only=comparable & a["success"] & ~b["success"],
        b_only=comparable & b["success"] & ~a["success"],
        neither=comparable & ~a["success"] & ~b["success"],
        comparable=comparable,
        unobserved_pair=~comparable,
        a_physical_failure=a["physical_failure"],
        b_physical_failure=b["physical_failure"],
        a_censored=a["censored"],
        b_censored=b["censored"],
        a_timeout=a["timeout"],
        b_timeout=b["timeout"],
        a_no_selection=a["no_selection"],
        b_no_selection=b["no_selection"],
        arrival_difference=jnp.where(both, b["arrival_time"], 0)
        - jnp.where(both, a["arrival_time"], 0),
        arrival_difference_defined=both,
        restricted_time_difference=jnp.where(
            restricted_defined, b["restricted_time"], 0
        )
        - jnp.where(restricted_defined, a["restricted_time"], 0),
        restricted_time_difference_defined=restricted_defined,
    )


def serialise(value):
    if hasattr(value, "_asdict"):
        return serialise(value._asdict())
    if isinstance(value, dict):
        return {key: serialise(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialise(item) for item in value]
    return value.tolist() if hasattr(value, "tolist") else value


def run():
    controls, mask = candidate_bank()
    models = {
        "gain_error": (dynamics(0.21), jnp.zeros(4)),
        "lateral_drift": (dynamics(), jnp.array([0.0, 0.01, 0.0, 0.0])),
    }
    evaluate = jax.jit(evaluate_root)
    records, pairs = {}, {}
    for root_id, root in roots().items():
        records[root_id] = {
            name: evaluate(root, controls, mask, model, offset)
            for name, (model, offset) in models.items()
        }
        pairs[root_id] = paired_outcomes(
            records[root_id]["gain_error"]["execution"],
            records[root_id]["lateral_drift"]["execution"],
        )
    return dict(
        protocol=dict(
            dt=DT,
            horizon=HORIZON,
            hold_steps=HOLD_STEPS,
            dwell_steps=DWELL_STEPS,
            dwell_window_span=(DWELL_STEPS - 1) * DT,
            position_tolerance=POSITION_TOLERANCE,
            speed_tolerance=SPEED_TOLERANCE,
            reserve=RESERVE,
            timing_tolerance=TIMING_TOLERANCE,
            selection="minimum terminal squared distance + 0.002 input energy "
            "among predicted reserve accepts",
            execution="complete open-loop sequence; no replanning or overrides",
            reference="deterministic supplied dynamics; sampled point clearance",
            candidate_ids=list(range(mask.size)),
            candidate_mask=mask,
            controls=controls,
        ),
        roots=roots(),
        models={
            name: dict(dynamics=model, state_offset=offset)
            for name, (model, offset) in models.items()
        },
        records=records,
        paired=pairs,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json", type=Path, help="Write all masks and per-root records"
    )
    args = parser.parse_args()
    report = serialise(run())
    print("root              model           position RMSE  selection  regret")
    for root_id, models in report["records"].items():
        for name, row in models.items():
            regret = row["regret"]
            value = (
                f"{regret['regret']:.5f}" if regret["regret_defined"] else "undefined"
            )
            print(
                f"{root_id:18s} {name:15s} {row['prediction_rmse']:12.5f} "
                f"{row['selected_index']:10d}  {value}"
            )
    counts = {
        name: sum(row[name] for row in report["paired"].values())
        for name in (
            "both_success",
            "a_only",
            "b_only",
            "neither",
            "unobserved_pair",
            "a_physical_failure",
            "b_physical_failure",
            "a_censored",
            "b_censored",
            "a_timeout",
            "b_timeout",
            "a_no_selection",
            "b_no_selection",
        )
    }
    print("Paired root counts (A=gain_error, B=lateral_drift):", counts)
    differences = [
        row["arrival_difference"]
        for row in report["paired"].values()
        if row["arrival_difference_defined"]
    ]
    print(
        "Both-success arrival differences B - A (seconds):",
        differences,
        "denominator:",
        len(differences),
    )
    for model_name in ("gain_error", "lateral_drift"):
        rows = [row[model_name] for row in report["records"].values()]
        for criterion in (
            "physical_counts",
            "reserve_counts",
            "reserve_vs_physical_counts",
            "completion_counts",
        ):
            names = (
                "false_acceptance",
                "model_accepted_count",
                "false_rejection",
                "reference_accepted_count",
                "compared_count",
                "unobserved_count",
            )
            print(
                model_name,
                criterion,
                {name: sum(row[criterion][name] for row in rows) for name in names},
            )
        names = (
            "claimed",
            "observable_claim",
            "confirmed_claim",
            "early",
            "delayed",
            "unconfirmed_through_horizon",
            "unobservable_claim",
            "right_censored",
            "failed",
        )
        print(
            model_name,
            "completion records:",
            {name: sum(sum(row["completion"][name]) for row in rows) for name in names},
        )
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Saved {args.json}")


if __name__ == "__main__":
    main()
