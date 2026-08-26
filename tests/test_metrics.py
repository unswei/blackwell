"""Tests for public error and consistency metrics."""

import jax
import jax.numpy as jnp

from blackwell import metrics


def test_rmse_and_position_rmse_have_expected_values() -> None:
    errors = jnp.array([[3.0, 4.0, 1.0], [0.0, 0.0, 3.0]])

    assert jnp.allclose(
        metrics.root_mean_square_error(errors),
        jnp.array([jnp.sqrt(4.5), jnp.sqrt(8.0), jnp.sqrt(5.0)]),
    )
    assert jnp.isclose(metrics.position_rmse(errors), jnp.sqrt(12.5))


def test_nees_metrics_are_jittable_and_use_matching_covariances() -> None:
    errors = jnp.array([[3.0, 4.0], [0.0, 2.0]])
    covariances = jnp.array(
        [
            [[1.0, 0.0], [0.0, 4.0]],
            [[4.0, 0.0], [0.0, 1.0]],
        ]
    )

    nees = jax.jit(metrics.normalised_estimation_error_squared)(errors, covariances)

    assert jnp.allclose(nees, jnp.array([13.0, 4.0]))
    assert jnp.isclose(metrics.mean_nees(errors, covariances), 8.5)
