"""Estimate an SE(2) trajectory from known-landmark range-bearing measurements.

Run from the repository root with:

    uv run python examples/se2_localisation.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from blackwell import metrics, simulation
from blackwell.beliefs import GaussianBelief
from blackwell.filters.ekf import ExtendedKalmanFilter
from blackwell.models import range_bearing
from blackwell.models import se2 as se2_models
from blackwell.spaces import se2


def main() -> None:
    dynamics_model = se2_models.BodyMotion(jnp.diag(jnp.array([0.01, 0.01, 0.0025])))
    observation_model = range_bearing.KnownLandmarksRangeBearing(
        landmarks=jnp.array([[5.0, 0.0], [0.0, 5.0], [-4.0, -3.0]]),
        measurement_covariance=jnp.diag(jnp.array([0.08, 0.02])),
    )
    simulator = simulation.Simulator(se2, se2_models, range_bearing)
    filter_ = ExtendedKalmanFilter(se2, se2_models, range_bearing)
    controls = jnp.tile(jnp.array([[0.4, 0.0, 0.06]]), (20, 1))
    initial_true_pose = jnp.zeros(3)
    initial_belief = GaussianBelief(
        mean=jnp.array([0.3, -0.2, 0.1]),
        covariance=jnp.diag(jnp.array([0.5, 0.5, 0.1])),
    )

    def estimate_trajectory(
        key: jax.Array,
    ) -> tuple[Array, Array, Array, Array]:
        states, measurements = simulator.rollout(
            key,
            initial_true_pose,
            dynamics_model,
            observation_model,
            controls,
        )

        def update_belief(
            belief: GaussianBelief,
            inputs: tuple[Array, Array],
        ) -> tuple[GaussianBelief, GaussianBelief]:
            control, measurement = inputs
            belief = filter_.step(
                belief,
                dynamics_model,
                observation_model,
                control,
                measurement,
            )
            return belief, belief

        _, beliefs = jax.lax.scan(
            update_belief, initial_belief, (controls, measurements)
        )
        errors = jax.vmap(se2.local_coordinates)(beliefs.mean, states)
        return states, beliefs.mean, errors, beliefs.covariance

    states, estimates, errors, covariances = jax.jit(estimate_trajectory)(
        jax.random.key(7)
    )
    print("Final true pose:", states[-1])
    print("Final estimated pose:", estimates[-1])
    print("Trajectory position RMSE:", metrics.position_rmse(errors))
    print("Trajectory mean NEES:", metrics.mean_nees(errors, covariances))


if __name__ == "__main__":
    main()
