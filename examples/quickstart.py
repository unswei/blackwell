"""One complete SE(2) extended-Kalman-filter update."""

import jax
import jax.numpy as jnp

from blackwell import GaussianBelief
from blackwell.filters.ekf import ExtendedKalmanFilter
from blackwell.models import range_bearing
from blackwell.models import se2 as se2_models
from blackwell.spaces import se2


def main() -> None:
    filter_ = ExtendedKalmanFilter(se2, se2_models, range_bearing)
    dynamics = se2_models.BodyMotion(
        process_covariance=jnp.diag(jnp.array([0.01, 0.01, 0.0025]))
    )
    observation = range_bearing.KnownLandmarksRangeBearing(
        landmarks=jnp.array([[5.0, 0.0], [0.0, 5.0]]),
        measurement_covariance=jnp.diag(jnp.array([0.08, 0.02])),
    )
    belief = GaussianBelief(
        mean=jnp.array([0.0, 0.0, 0.0]),
        covariance=jnp.diag(jnp.array([0.5, 0.5, 0.1])),
    )
    control = jnp.array([0.3, 0.0, 0.04])
    true_pose = jnp.array([0.35, 0.08, 0.05])
    measurement = range_bearing.observe(true_pose, observation)

    belief = jax.jit(filter_.step)(
        belief, dynamics, observation, control, measurement
    )
    print("Estimated pose:", belief.mean)
    print("Tangent covariance:\n", belief.covariance)


if __name__ == "__main__":
    main()
