# Five-minute localisation

This example performs one SE(2) extended-Kalman-filter step from body-frame
motion and two known-landmark range-bearing observations.

## 1. Configure the model families

`ExtendedKalmanFilter` captures *operation modules*—the geometry and algorithms
that remain static during compilation. Covariance values and landmarks are
separate JAX PyTrees.

```python
import jax
import jax.numpy as jnp

from blackwell import GaussianBelief
from blackwell.filters.ekf import ExtendedKalmanFilter
from blackwell.models import range_bearing
from blackwell.models import se2 as se2_models
from blackwell.spaces import se2

filter_ = ExtendedKalmanFilter(se2, se2_models, range_bearing)

dynamics = se2_models.BodyMotion(
    process_covariance=jnp.diag(jnp.array([0.01, 0.01, 0.0025]))
)
observation = range_bearing.KnownLandmarksRangeBearing(
    landmarks=jnp.array([[5.0, 0.0], [0.0, 5.0]]),
    measurement_covariance=jnp.diag(jnp.array([0.08, 0.02])),
)
```

## 2. Create a belief and measurement

SE(2) states are `[x, y, heading]`. Their covariance is a $3\times3$ matrix in
local body-frame tangent coordinates `[forward, lateral, turn]`.

```python
belief = GaussianBelief(
    mean=jnp.array([0.0, 0.0, 0.0]),
    covariance=jnp.diag(jnp.array([0.5, 0.5, 0.1])),
)
control = jnp.array([0.3, 0.0, 0.04])

true_pose = jnp.array([0.35, 0.08, 0.05])
measurement = range_bearing.observe(true_pose, observation)
```

## 3. Compile and estimate

```python
step = jax.jit(filter_.step)
belief = step(belief, dynamics, observation, control, measurement)

print(belief.mean)
print(belief.covariance)
```

The result remains a `GaussianBelief` PyTree and can be carried through
`jax.lax.scan` for an entire trajectory.

!!! tip "Run the exact example"

    ```console
    uv run python examples/quickstart.py
    ```

## What happened?

1. The dynamics model retracted the control on SE(2).
2. The EKF transported the old covariance through the tangent transition
   Jacobian and added process covariance.
3. The range-bearing model predicted observations and wrapped bearing
   residuals across the $\pm\pi$ branch cut.
4. The EKF applied a Joseph-form covariance update and re-expressed the result
   at the corrected mean.

Next, read [Beliefs and tangent spaces](../concepts/beliefs-and-tangent-spaces.md)
for the geometry, or run the full [SE(2) localisation example](../examples/se2-localisation.md).
