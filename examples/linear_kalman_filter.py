"""Track one-dimensional position and velocity with a linear Kalman filter."""

import jax
import jax.numpy as jnp

from blackwell import GaussianBelief
from blackwell.filters.ekf import ExtendedKalmanFilter
from blackwell.models import linear
from blackwell.spaces import euclidean


def run() -> GaussianBelief:
    filter_ = ExtendedKalmanFilter(euclidean, linear, linear)
    dynamics = linear.LinearDynamics(
        transition=jnp.array([[1.0, 1.0], [0.0, 1.0]]),
        control=jnp.array([[0.5], [1.0]]),
        process_covariance=jnp.diag(jnp.array([0.02, 0.04])),
    )
    observation = linear.LinearObservation(
        observation=jnp.array([[1.0, 0.0]]),
        measurement_covariance=jnp.array([[0.25]]),
    )
    initial_belief = GaussianBelief(
        mean=jnp.array([0.0, 0.0]),
        covariance=jnp.diag(jnp.array([1.0, 1.0])),
    )
    accelerations = jnp.array([[0.2], [0.1], [0.0], [-0.1], [0.0]])
    positions = jnp.array([[0.1], [0.8], [1.7], [2.8], [3.9]])

    def update(
        belief: GaussianBelief,
        inputs: tuple[jax.Array, jax.Array],
    ) -> tuple[GaussianBelief, GaussianBelief]:
        control, measurement = inputs
        belief = filter_.step(
            belief, dynamics, observation, control, measurement
        )
        return belief, belief

    final_belief, _ = jax.jit(
        lambda belief: jax.lax.scan(
            update, belief, (accelerations, positions)
        )
    )(initial_belief)
    return final_belief


def main() -> None:
    belief = run()
    print("Final [position, velocity]:", belief.mean)
    print("Final covariance:\n", belief.covariance)


if __name__ == "__main__":
    main()
