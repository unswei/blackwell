"""Correlated SE(3) uncertainty propagation using only Blackwell's public API.

Run with ``uv run python examples/se3_uncertainty.py``. Pose covariances use
right/body tangents [rho, phi]; point covariances use Cartesian coordinates.
Cross-covariances are supplied explicitly, never inferred from two marginals.
"""

import jax
import jax.numpy as jnp
from jax import Array

from blackwell import GaussianBelief, ParticleBelief
from blackwell.spaces import se3


def compose_beliefs(
    left: GaussianBelief, right: GaussianBelief, cross_covariance: Array
) -> GaussianBelief:
    """Compose poses with Cov(delta_left, delta_right), shape (6, 6)."""
    mean = se3.compose(left.mean, right.mean)

    def residual(delta):
        value = se3.compose(
            se3.retract(left.mean, delta[:6]),
            se3.retract(right.mean, delta[6:]),
        )
        return se3.local_coordinates(mean, value)

    jacobian = jax.jacfwd(residual)(jnp.zeros(12, dtype=mean.dtype))
    joint = jnp.block(
        [
            [left.covariance, cross_covariance],
            [cross_covariance.T, right.covariance],
        ]
    )
    covariance = jacobian @ joint @ jacobian.T
    return GaussianBelief(mean, (covariance + covariance.T) / 2)


def invert_belief(belief: GaussianBelief) -> GaussianBelief:
    """Invert a pose, expressing output covariance at the inverse mean."""
    mean = se3.inverse(belief.mean)
    jacobian = jax.jacfwd(
        lambda delta: se3.local_coordinates(
            mean, se3.inverse(se3.retract(belief.mean, delta))
        )
    )(jnp.zeros(6, dtype=mean.dtype))
    covariance = jacobian @ belief.covariance @ jacobian.T
    return GaussianBelief(mean, (covariance + covariance.T) / 2)


def transform_point_beliefs(
    pose: GaussianBelief, point: GaussianBelief, cross_covariance: Array
) -> GaussianBelief:
    """Transform a point with Cov(delta_pose, delta_point), shape (6, 3)."""
    mean = se3.transform_point(pose.mean, point.mean)

    def transform(delta):
        return se3.transform_point(
            se3.retract(pose.mean, delta[:6]), point.mean + delta[6:]
        )

    jacobian = jax.jacfwd(transform)(jnp.zeros(9, dtype=mean.dtype))
    joint = jnp.block(
        [
            [pose.covariance, cross_covariance],
            [cross_covariance.T, point.covariance],
        ]
    )
    covariance = jacobian @ joint @ jacobian.T
    return GaussianBelief(mean, (covariance + covariance.T) / 2)


def point_scenario():
    """Small correlated pose/point errors from a positive-definite joint model."""
    scales = jnp.array([0.015, 0.02, 0.01, 0.006, 0.008, 0.007, 0.02, 0.015, 0.025])
    factor = jnp.eye(9).at[6, 0].set(0.8).at[7, 5].set(0.7).at[8, 4].set(-0.6)
    factor = scales[:, None] * factor
    joint = factor @ factor.T
    pose = GaussianBelief(
        se3.exp(jnp.array([1.0, -0.4, 0.3, 0.4, -0.2, 0.5])), joint[:6, :6]
    )
    point = GaussianBelief(jnp.array([2.0, -1.0, 0.5]), joint[6:, 6:])
    return pose, point, joint


def sample_transformed_points(
    key: Array,
    pose: GaussianBelief,
    point: GaussianBelief,
    joint_covariance: Array,
    sample_count: int,
) -> ParticleBelief:
    """Draw paired errors jointly, preserving pose/point correlation."""
    errors = jax.random.normal(key, (sample_count, 9), dtype=pose.mean.dtype)
    errors = errors @ jnp.linalg.cholesky(joint_covariance).T
    weights = jnp.full(sample_count, 1 / sample_count, dtype=pose.mean.dtype)
    poses = ParticleBelief(
        jax.vmap(se3.retract, in_axes=(None, 0))(pose.mean, errors[:, :6]), weights
    )
    points = point.mean + errors[:, 6:]
    return ParticleBelief(
        jax.vmap(se3.transform_point)(poses.particles, points), poses.weights
    )


def empirical_gaussian(particles: ParticleBelief) -> GaussianBelief:
    """Summarise the Euclidean transformed-point cloud (population covariance)."""
    mean = particles.weights @ particles.particles
    centred = particles.particles - mean
    covariance = centred.T @ (particles.weights[:, None] * centred)
    return GaussianBelief(mean, covariance)


def main() -> None:
    pose, point, joint = point_scenario()
    # A shared latent source correlates both transform inputs.
    left_factor = jnp.diag(jnp.array([0.02, 0.01, 0.03, 0.006, 0.008, 0.01]))
    right_factor = left_factor * 0.5
    left = GaussianBelief(pose.mean, left_factor @ left_factor.T)
    right = GaussianBelief(
        se3.exp(jnp.array([0.2, 0.1, -0.3, -0.1, 0.2, 0.1])),
        right_factor @ right_factor.T + jnp.eye(6) * 1e-5,
    )
    composed = jax.jit(compose_beliefs)(left, right, left_factor @ right_factor.T)
    inverted = jax.jit(invert_belief)(composed)
    print("Composed pose [x, y, z, qx, qy, qz, qw]:", composed.mean)
    print("Composed body covariance diagonal:", jnp.diag(composed.covariance))
    print("Inverse body covariance diagonal:", jnp.diag(inverted.covariance))

    predicted = jax.jit(transform_point_beliefs)(pose, point, joint[:6, 6:])
    independent = transform_point_beliefs(pose, point, jnp.zeros((6, 3)))
    cloud = jax.jit(sample_transformed_points, static_argnames="sample_count")(
        jax.random.key(2026), pose, point, joint, sample_count=50_000
    )
    empirical = empirical_gaussian(cloud)

    def relative_error(covariance):
        return jnp.linalg.norm(covariance - empirical.covariance) / jnp.linalg.norm(
            empirical.covariance
        )

    print("Transformed point:", predicted.mean)
    print("Monte Carlo mean:", empirical.mean)
    print("First-order covariance:\n", predicted.covariance)
    print("Monte Carlo covariance:\n", empirical.covariance)
    print(
        f"Relative covariance error: {float(relative_error(predicted.covariance)):.2%}"
    )
    print(
        "Error if correlation is discarded:",
        f"{float(relative_error(independent.covariance)):.2%}",
    )


if __name__ == "__main__":
    main()
