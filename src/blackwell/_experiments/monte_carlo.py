"""Batched Monte Carlo evaluation for the private SE(2) EKF experiment."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from blackwell import metrics
from blackwell._experiments.se2_range_bearing import (
    RangeBearingLocalisationModel,
    SE2Belief,
    local_coordinates,
    range_bearing,
    retract,
    wrap_angle,
)
from blackwell._experiments.se2_range_bearing import step as ekf_step


class MonteCarloResult(NamedTuple):
    """Truth and filter outputs from independent localisation trials.

    Each field has leading ``(trial_count, step_count)`` dimensions.
    Covariances remain in the tangent coordinates of the corresponding estimate.
    """

    true_poses: Array
    estimates: Array
    covariances: Array


def run_se2_ekf_trials(
    key: Array,
    initial_true_pose: Array,
    initial_belief: SE2Belief,
    model: RangeBearingLocalisationModel,
    controls: Array,
    trial_count: int,
) -> MonteCarloResult:
    """Run independent noisy SE(2) EKF trials in parallel.

    ``controls`` has shape ``(step_count, 3)`` and uses the body-frame tangent
    convention of the localisation experiment. ``trial_count`` must be static
    when this function is passed to :func:`jax.jit`.
    """

    process_key, measurement_key = jax.random.split(key)
    step_count = controls.shape[0]
    process_noise = jax.random.multivariate_normal(
        process_key,
        mean=jnp.zeros(3, dtype=initial_true_pose.dtype),
        cov=model.process_covariance,
        shape=(trial_count, step_count),
        method="svd",
    )
    measurement_noise = jax.random.multivariate_normal(
        measurement_key,
        mean=jnp.zeros(2, dtype=initial_true_pose.dtype),
        cov=model.measurement_covariance,
        shape=(trial_count, step_count, model.landmarks.shape[0]),
        method="svd",
    )

    def run_single_trial(
        trial_process_noise: Array, trial_measurement_noise: Array
    ) -> tuple[Array, Array, Array]:
        def run_step(
            carry: tuple[Array, SE2Belief],
            inputs: tuple[Array, Array, Array],
        ) -> tuple[tuple[Array, SE2Belief], tuple[Array, SE2Belief]]:
            true_pose, belief = carry
            control, step_process_noise, step_measurement_noise = inputs
            true_pose = retract(retract(true_pose, control), step_process_noise)
            measurements = range_bearing(true_pose, model.landmarks)
            measurements = measurements + step_measurement_noise
            measurements = measurements.at[:, 1].set(wrap_angle(measurements[:, 1]))
            belief = ekf_step(belief, model, control, measurements)
            return (true_pose, belief), (true_pose, belief)

        _, (true_poses, beliefs) = jax.lax.scan(
            run_step,
            (initial_true_pose, initial_belief),
            (controls, trial_process_noise, trial_measurement_noise),
        )
        return true_poses, beliefs.pose, beliefs.covariance

    true_poses, estimates, covariances = jax.vmap(run_single_trial)(
        process_noise, measurement_noise
    )
    return MonteCarloResult(
        true_poses=true_poses,
        estimates=estimates,
        covariances=covariances,
    )


def tangent_errors(result: MonteCarloResult) -> Array:
    """Return true pose errors in each estimate's local tangent coordinates."""

    return jax.vmap(jax.vmap(local_coordinates))(
        result.estimates, result.true_poses
    )


def position_rmse(result: MonteCarloResult) -> Array:
    """Return per-step position RMSE over the independent trials."""

    return metrics.position_rmse(tangent_errors(result))


def normalised_estimation_error_squared(result: MonteCarloResult) -> Array:
    """Return the per-trial, per-step tangent-space NEES values."""

    return metrics.normalised_estimation_error_squared(
        tangent_errors(result), result.covariances
    )


def mean_nees(result: MonteCarloResult) -> Array:
    """Return the per-step NEES averaged over all independent trials."""

    return metrics.mean_nees(tangent_errors(result), result.covariances)
