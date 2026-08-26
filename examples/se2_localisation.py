"""Estimate an SE(2) trajectory from known-landmark observations."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from blackwell import GaussianBelief, metrics, simulation
from blackwell.filters.ekf import ExtendedKalmanFilter
from blackwell.models import range_bearing
from blackwell.models import se2 as se2_models
from blackwell.spaces import se2


class Result(NamedTuple):
    """Outputs retained for evaluation and optional plotting."""

    true_poses: Array
    estimates: Array
    errors: Array
    covariances: Array
    landmarks: Array


def run(key: Array) -> Result:
    dynamics = se2_models.BodyMotion(
        jnp.diag(jnp.array([0.01, 0.01, 0.0025]))
    )
    observation = range_bearing.KnownLandmarksRangeBearing(
        landmarks=jnp.array([[5.0, 0.0], [0.0, 5.0], [-4.0, -3.0]]),
        measurement_covariance=jnp.diag(jnp.array([0.08, 0.02])),
    )
    simulator = simulation.Simulator(se2, se2_models, range_bearing)
    filter_ = ExtendedKalmanFilter(se2, se2_models, range_bearing)
    controls = jnp.tile(jnp.array([[0.4, 0.0, 0.06]]), (20, 1))
    initial_belief = GaussianBelief(
        mean=jnp.array([0.3, -0.2, 0.1]),
        covariance=jnp.diag(jnp.array([0.5, 0.5, 0.1])),
    )
    true_poses, measurements = simulator.rollout(
        key, jnp.zeros(3), dynamics, observation, controls
    )

    def update(
        belief: GaussianBelief,
        inputs: tuple[Array, Array],
    ) -> tuple[GaussianBelief, GaussianBelief]:
        control, measurement = inputs
        belief = filter_.step(
            belief, dynamics, observation, control, measurement
        )
        return belief, belief

    _, beliefs = jax.lax.scan(update, initial_belief, (controls, measurements))
    errors = jax.vmap(se2.local_coordinates)(beliefs.mean, true_poses)
    return Result(
        true_poses,
        beliefs.mean,
        errors,
        beliefs.covariance,
        observation.landmarks,
    )


def plot(result: Result, output: Path) -> None:
    """Save a trajectory plot; Matplotlib is imported only when requested."""

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Ellipse

    true_poses, estimates, _, covariances, landmarks = jax.device_get(result)
    figure, axes = plt.subplots(figsize=(8, 6), constrained_layout=True)
    axes.plot(true_poses[:, 0], true_poses[:, 1], label="true", linewidth=2.4)
    axes.plot(estimates[:, 0], estimates[:, 1], label="EKF", linewidth=2)
    axes.scatter(landmarks[:, 0], landmarks[:, 1], marker="*", s=130, label="landmarks")

    for index in range(3, len(estimates), 4):
        heading = estimates[index, 2]
        rotation = np.array(
            [[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]]
        )
        position_covariance = rotation @ covariances[index, :2, :2] @ rotation.T
        eigenvalues, eigenvectors = np.linalg.eigh(position_covariance)
        direction = eigenvectors[:, -1]
        angle = np.degrees(np.arctan2(direction[1], direction[0]))
        width, height = 4 * np.sqrt(np.maximum(eigenvalues[::-1], 0.0))
        axes.add_patch(
            Ellipse(
                estimates[index, :2],
                width,
                height,
                angle=angle,
                fill=False,
                alpha=0.45,
                linewidth=1,
            )
        )

    axes.set(xlabel="x", ylabel="y", title="SE(2) range-bearing localisation")
    axes.axis("equal")
    axes.grid(alpha=0.2)
    axes.legend()
    figure.savefig(output, dpi=170)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", type=Path, help="optional output PNG path")
    arguments = parser.parse_args()
    result = jax.jit(run)(jax.random.key(7))

    print("Final true pose:", result.true_poses[-1])
    print("Final estimated pose:", result.estimates[-1])
    print("Trajectory position RMSE:", metrics.position_rmse(result.errors))
    print(
        "Trajectory mean NEES:",
        metrics.mean_nees(result.errors, result.covariances),
    )
    if arguments.plot:
        plot(result, arguments.plot)
        print("Saved plot:", arguments.plot)


if __name__ == "__main__":
    main()
