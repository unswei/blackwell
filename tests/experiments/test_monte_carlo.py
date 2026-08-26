"""Tests for batched Monte Carlo SE(2) localisation experiments."""

import jax
import jax.numpy as jnp

from blackwell._experiments.monte_carlo import (
    mean_nees,
    normalised_estimation_error_squared,
    position_rmse,
    run_se2_ekf_trials,
    tangent_errors,
)
from blackwell._experiments.se2_range_bearing import (
    RangeBearingLocalisationModel,
    SE2Belief,
)


def _model() -> RangeBearingLocalisationModel:
    return RangeBearingLocalisationModel(
        landmarks=jnp.array([[5.0, 0.0], [0.0, 5.0]]),
        process_covariance=jnp.zeros((3, 3)),
        measurement_covariance=jnp.diag(jnp.array([0.1, 0.02])),
    )


def test_trials_batch_independent_noisy_measurements_over_shared_truth() -> None:
    result = run_se2_ekf_trials(
        jax.random.key(0),
        initial_true_pose=jnp.zeros(3),
        initial_belief=SE2Belief(jnp.zeros(3), jnp.diag(jnp.array([1.0, 1.0, 0.2]))),
        model=_model(),
        controls=jnp.array([[1.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        trial_count=4,
    )

    expected_truth = jnp.array([[1.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    assert result.true_poses.shape == (4, 2, 3)
    assert result.estimates.shape == (4, 2, 3)
    assert result.covariances.shape == (4, 2, 3, 3)
    assert jnp.allclose(result.true_poses, jnp.broadcast_to(expected_truth, (4, 2, 3)))
    assert not jnp.allclose(result.estimates[0], result.estimates[1])


def test_metrics_are_finite_and_have_one_value_per_step() -> None:
    result = run_se2_ekf_trials(
        jax.random.key(1),
        initial_true_pose=jnp.zeros(3),
        initial_belief=SE2Belief(jnp.zeros(3), jnp.eye(3)),
        model=_model(),
        controls=jnp.array([[0.5, 0.0, 0.1], [0.5, 0.0, -0.1]]),
        trial_count=3,
    )

    assert tangent_errors(result).shape == (3, 2, 3)
    assert position_rmse(result).shape == (2,)
    assert normalised_estimation_error_squared(result).shape == (3, 2)
    assert mean_nees(result).shape == (2,)
    assert jnp.all(jnp.isfinite(position_rmse(result)))
    assert jnp.all(jnp.isfinite(mean_nees(result)))


def test_trial_runner_is_jittable_when_trial_count_is_static() -> None:
    compiled_runner = jax.jit(run_se2_ekf_trials, static_argnames=("trial_count",))

    result = compiled_runner(
        jax.random.key(2),
        jnp.zeros(3),
        SE2Belief(jnp.zeros(3), jnp.eye(3)),
        _model(),
        jnp.array([[0.1, 0.0, 0.0]]),
        trial_count=2,
    )

    assert result.true_poses.shape == (2, 1, 3)
