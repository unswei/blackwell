"""Tests for the private linear-Gaussian filtering experiment."""

import jax
import jax.numpy as jnp

from blackwell._experiments.linear_gaussian import (
    GaussianBelief,
    LinearGaussianModel,
    predict,
    step,
    update,
)


def test_predict_applies_dynamics_control_and_process_noise() -> None:
    model = LinearGaussianModel(
        transition=jnp.array([[1.0, 1.0], [0.0, 1.0]]),
        control=jnp.array([[0.5], [1.0]]),
        process_covariance=jnp.array([[0.1, 0.0], [0.0, 0.2]]),
        observation=jnp.array([[1.0, 0.0]]),
        measurement_covariance=jnp.array([[1.0]]),
    )
    belief = GaussianBelief(
        mean=jnp.array([2.0, 3.0]),
        covariance=jnp.array([[2.0, 0.5], [0.5, 1.0]]),
    )

    result = predict(belief, model, jnp.array([4.0]))

    assert jnp.allclose(result.mean, jnp.array([7.0, 7.0]))
    assert jnp.allclose(result.covariance, jnp.array([[4.1, 1.5], [1.5, 1.2]]))


def test_update_matches_closed_form_scalar_measurement() -> None:
    model = LinearGaussianModel(
        transition=jnp.eye(2),
        control=jnp.zeros((2, 0)),
        process_covariance=jnp.zeros((2, 2)),
        observation=jnp.array([[1.0, 0.0]]),
        measurement_covariance=jnp.array([[2.0]]),
    )
    belief = GaussianBelief(
        mean=jnp.array([0.0, 0.0]),
        covariance=jnp.array([[2.0, 0.0], [0.0, 1.0]]),
    )

    result = update(belief, model, jnp.array([4.0]))

    assert jnp.allclose(result.mean, jnp.array([2.0, 0.0]))
    assert jnp.allclose(result.covariance, jnp.array([[1.0, 0.0], [0.0, 1.0]]))


def test_step_is_jittable_and_returns_a_symmetric_covariance() -> None:
    model = LinearGaussianModel(
        transition=jnp.array([[1.0, 1.0], [0.0, 1.0]]),
        control=jnp.array([[0.5], [1.0]]),
        process_covariance=jnp.eye(2) * 0.01,
        observation=jnp.array([[1.0, 0.0]]),
        measurement_covariance=jnp.array([[0.25]]),
    )
    belief = GaussianBelief(jnp.array([0.0, 1.0]), jnp.eye(2))

    result = jax.jit(step)(belief, model, jnp.array([0.0]), jnp.array([1.2]))

    assert result.mean.shape == (2,)
    assert result.covariance.shape == (2, 2)
    assert jnp.allclose(result.covariance, result.covariance.T)
    assert jnp.all(jnp.linalg.eigvalsh(result.covariance) >= -1e-6)
