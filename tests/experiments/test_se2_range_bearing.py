"""Tests for the private SE(2) range-bearing EKF localisation experiment."""

import jax
import jax.numpy as jnp

from blackwell._experiments.se2_range_bearing import (
    RangeBearingLocalisationModel,
    SE2Belief,
    local_coordinates,
    predict,
    range_bearing,
    range_bearing_jacobian,
    retract,
    step,
    update,
)


def test_predict_applies_a_body_frame_se2_control() -> None:
    model = RangeBearingLocalisationModel(
        landmarks=jnp.array([[3.0, 4.0]]),
        process_covariance=jnp.zeros((3, 3)),
        measurement_covariance=jnp.eye(2),
    )
    belief = SE2Belief(jnp.array([1.0, 2.0, jnp.pi / 2]), jnp.eye(3))

    result = predict(belief, model, jnp.array([1.0, 0.0, 0.0]))

    assert jnp.allclose(result.pose, jnp.array([1.0, 3.0, jnp.pi / 2]))
    assert jnp.allclose(
        local_coordinates(belief.pose, result.pose), jnp.array([1.0, 0.0, 0.0])
    )


def test_range_bearing_uses_the_robot_local_frame() -> None:
    measurements = range_bearing(
        jnp.array([0.0, 0.0, jnp.pi / 2]),
        jnp.array([[0.0, 2.0], [2.0, 0.0]]),
    )

    assert jnp.allclose(measurements[0], jnp.array([2.0, 0.0]), atol=1e-6)
    assert jnp.allclose(measurements[1], jnp.array([2.0, -jnp.pi / 2]), atol=1e-6)


def test_range_bearing_jacobian_matches_automatic_differentiation() -> None:
    pose = jnp.array([1.0, -0.5, 0.4])
    landmarks = jnp.array([[4.0, 2.0], [-1.0, 3.0]])

    automatic_jacobian = jax.jacfwd(
        lambda tangent: range_bearing(retract(pose, tangent), landmarks)
    )(jnp.zeros(3))

    assert jnp.allclose(
        range_bearing_jacobian(pose, landmarks), automatic_jacobian, atol=1e-6
    )


def test_update_reduces_measurement_error_and_preserves_covariance_properties() -> None:
    model = RangeBearingLocalisationModel(
        landmarks=jnp.array([[5.0, 0.0], [0.0, 5.0]]),
        process_covariance=jnp.eye(3) * 0.01,
        measurement_covariance=jnp.diag(jnp.array([0.05, 0.01])),
    )
    belief = SE2Belief(jnp.array([0.0, 0.0, 0.0]), jnp.diag(jnp.array([2.0, 2.0, 0.5])))
    measurements = range_bearing(jnp.array([0.8, -0.4, 0.15]), model.landmarks)

    result = update(belief, model, measurements)

    initial_error = jnp.linalg.norm(
        range_bearing(belief.pose, model.landmarks) - measurements
    )
    updated_error = jnp.linalg.norm(
        range_bearing(result.pose, model.landmarks) - measurements
    )
    assert updated_error < initial_error
    assert jnp.allclose(result.covariance, result.covariance.T)
    assert jnp.all(jnp.linalg.eigvalsh(result.covariance) >= -1e-6)


def test_update_wraps_a_bearing_innovation_across_pi() -> None:
    model = RangeBearingLocalisationModel(
        landmarks=jnp.array([[5.0, 0.0]]),
        process_covariance=jnp.zeros((3, 3)),
        measurement_covariance=jnp.diag(jnp.array([0.01, 0.001])),
    )
    belief = SE2Belief(
        jnp.array([0.0, 0.0, -jnp.pi + 0.01]),
        jnp.diag(jnp.array([0.01, 0.01, 1.0])),
    )
    measurements = jnp.array([[5.0, -jnp.pi + 0.03]])

    result = update(belief, model, measurements)
    local_update = local_coordinates(belief.pose, result.pose)

    assert -0.1 < local_update[2] < 0.0


def test_step_is_jittable() -> None:
    model = RangeBearingLocalisationModel(
        landmarks=jnp.array([[4.0, 0.0], [0.0, 4.0]]),
        process_covariance=jnp.eye(3) * 0.1,
        measurement_covariance=jnp.diag(jnp.array([0.2, 0.1])),
    )
    belief = SE2Belief(jnp.array([0.0, 0.0, 0.0]), jnp.eye(3))
    measurements = range_bearing(jnp.array([0.2, 0.0, 0.05]), model.landmarks)

    result = jax.jit(step)(belief, model, jnp.array([0.1, 0.0, 0.02]), measurements)

    assert result.pose.shape == (3,)
    assert result.covariance.shape == (3, 3)
