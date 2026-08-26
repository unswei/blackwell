"""Localise an SE(2) robot with a bootstrap particle filter."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from blackwell import ParticleBelief, metrics, simulation
from blackwell.filters.particle import BootstrapParticleFilter
from blackwell.models import range_bearing
from blackwell.models import se2 as se2_models
from blackwell.spaces import se2


class Result(NamedTuple):
    true_poses: Array
    estimates: Array
    errors: Array
    final_belief: ParticleBelief
    landmarks: Array


def weighted_pose(belief: ParticleBelief) -> Array:
    """Compute a weighted translation and circular heading mean."""

    translation = jnp.sum(belief.weights[:, None] * belief.particles[:, :2], axis=0)
    heading = jnp.arctan2(
        jnp.sum(belief.weights * jnp.sin(belief.particles[:, 2])),
        jnp.sum(belief.weights * jnp.cos(belief.particles[:, 2])),
    )
    return jnp.concatenate((translation, jnp.atleast_1d(heading)))


def run(key: Array, particle_count: int = 500) -> Result:
    simulation_key, initialisation_key, filtering_key = jax.random.split(key, 3)
    dynamics = se2_models.BodyMotion(
        jnp.diag(jnp.array([0.015, 0.015, 0.004]))
    )
    observation = range_bearing.KnownLandmarksRangeBearing(
        landmarks=jnp.array([[5.0, 0.0], [0.0, 5.0], [-4.0, -3.0]]),
        measurement_covariance=jnp.diag(jnp.array([0.1, 0.025])),
    )
    simulator = simulation.Simulator(se2, se2_models, range_bearing)
    filter_ = BootstrapParticleFilter(se2, se2_models, range_bearing)
    controls = jnp.tile(jnp.array([[0.35, 0.0, 0.055]]), (20, 1))
    true_poses, measurements = simulator.rollout(
        simulation_key, jnp.zeros(3), dynamics, observation, controls
    )
    belief = filter_.initialise(
        initialisation_key,
        jnp.array([0.6, -0.4, 0.15]),
        jnp.diag(jnp.array([1.0, 1.0, 0.35])),
        particle_count,
    )
    keys = jax.random.split(filtering_key, controls.shape[0])

    def update(
        belief: ParticleBelief,
        inputs: tuple[Array, Array, Array],
    ) -> tuple[ParticleBelief, Array]:
        step_key, control, measurement = inputs
        prediction_key, resampling_key = jax.random.split(step_key)
        weighted = filter_.step(
            prediction_key, belief, dynamics, observation, control, measurement
        )
        estimate = weighted_pose(weighted)
        belief = jax.lax.cond(
            filter_.effective_sample_size(weighted) < particle_count / 2,
            lambda candidate: filter_.systematic_resample(
                resampling_key, candidate
            ),
            lambda candidate: candidate,
            weighted,
        )
        return belief, estimate

    final_belief, estimates = jax.lax.scan(
        update, belief, (keys, controls, measurements)
    )
    errors = jax.vmap(se2.local_coordinates)(estimates, true_poses)
    return Result(
        true_poses,
        estimates,
        errors,
        final_belief,
        observation.landmarks,
    )


def plot(result: Result, output: Path) -> None:
    import matplotlib.pyplot as plt

    true_poses, estimates, _, belief, landmarks = jax.device_get(result)
    figure, axes = plt.subplots(figsize=(8, 6), constrained_layout=True)
    axes.plot(true_poses[:, 0], true_poses[:, 1], label="true", linewidth=2.4)
    axes.plot(estimates[:, 0], estimates[:, 1], label="particles", linewidth=2)
    axes.scatter(
        belief.particles[:, 0],
        belief.particles[:, 1],
        s=8,
        alpha=0.18,
        label="final particles",
    )
    axes.scatter(landmarks[:, 0], landmarks[:, 1], marker="*", s=130, label="landmarks")
    axes.set(xlabel="x", ylabel="y", title="SE(2) particle localisation")
    axes.axis("equal")
    axes.grid(alpha=0.2)
    axes.legend()
    figure.savefig(output, dpi=170)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", type=Path, help="optional output PNG path")
    arguments = parser.parse_args()
    result = jax.jit(run, static_argnames=("particle_count",))(
        jax.random.key(11), particle_count=500
    )

    print("Final true pose:", result.true_poses[-1])
    print("Final estimated pose:", result.estimates[-1])
    print("Trajectory position RMSE:", metrics.position_rmse(result.errors))
    if arguments.plot:
        plot(result, arguments.plot)
        print("Saved plot:", arguments.plot)


if __name__ == "__main__":
    main()
