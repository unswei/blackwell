# Extended Kalman filter

Blackwell's `ExtendedKalmanFilter` applies the same tangent-space algorithm to
Euclidean vectors and manifold-valued states. You supply a state space, a
dynamics family and an observation family; the filter supplies prediction and
Joseph-form measurement updates.

## Configure the operations once

For known-landmark planar localisation:

```python
from blackwell.filters.ekf import ExtendedKalmanFilter
from blackwell.models import range_bearing
from blackwell.models import se2 as se2_models
from blackwell.spaces import se2

filter_ = ExtendedKalmanFilter(
    state_space=se2,
    dynamics=se2_models,
    observation=range_bearing,
)
```

The instance stores Python operation modules. Model values such as covariances
and landmarks stay in immutable JAX-compatible containers:

```python
import jax.numpy as jnp

dynamics = se2_models.BodyMotion(
    process_covariance=jnp.diag(jnp.array([0.04, 0.02, 0.01]))
)
observation = range_bearing.KnownLandmarksRangeBearing(
    landmarks=jnp.array([[8.0, 1.0], [1.0, 7.0], [-4.0, 5.0]]),
    measurement_covariance=jnp.diag(jnp.array([0.10, 0.02])),
)
```

## Initialise a local Gaussian

The SE(2) mean is `[x, y, heading]`. Its covariance is expressed in local
body-frame tangent coordinates `[forward, lateral, turn]`:

```python
from blackwell import GaussianBelief

belief = GaussianBelief(
    mean=jnp.array([0.0, 0.0, 0.0]),
    covariance=jnp.diag(jnp.array([0.5, 0.5, 0.2])),
)
```

Blackwell does not silently validate or repair inputs inside compiled kernels.
Use a symmetric, positive-semidefinite covariance with a shape matching the
state space's tangent dimension.

## Predict, update or do both

Use the individual operations when the timing of control and measurements
differs:

```python
predicted = filter_.predict(belief, dynamics, control)
updated = filter_.update(predicted, observation, measurement)
```

When each control has one corresponding measurement, `step` performs the same
sequence:

```python
updated = filter_.step(
    belief,
    dynamics,
    observation,
    control,
    measurement,
)
```

The update:

1. forms a topology-aware innovation using the observation family;
2. computes the tangent-space Kalman gain;
3. retracts the correction onto the state space;
4. uses the Joseph covariance form; and
5. transports covariance to the corrected mean's tangent coordinates.

## Compile a trajectory

Bind the configured filter before compilation, and carry the belief through
`jax.lax.scan`:

```python
import jax

step = jax.jit(filter_.step)

def scan_step(belief, inputs):
    control, measurement = inputs
    next_belief = filter_.step(
        belief, dynamics, observation, control, measurement
    )
    return next_belief, next_belief

final_belief, history = jax.lax.scan(
    scan_step, belief, (controls, measurements)
)
```

The complete [SE(2) EKF localisation example](../examples/se2-localisation.md)
adds simulation, error metrics and a plot.

## Practical checks

- Plot innovations or residual norms. Persistent bias usually indicates a
  model, frame or units mismatch.
- Inspect covariance symmetry and eigenvalues during model development.
- Compare mean NEES with the tangent dimension over repeated trials; a much
  larger value usually indicates overconfidence.
- Keep heading and bearing values in radians.
- Avoid a range-bearing landmark exactly coincident with the robot pose; the
  observation Jacobian is singular there.

See the [`ExtendedKalmanFilter` API](../reference/filters/ekf.md) for signatures
and the [model contract](../concepts/models-and-filters.md) for custom models.
