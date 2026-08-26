# API guide

Blackwell's public API is intentionally small and pre-alpha. All numerical
values are JAX arrays, and beliefs and model parameters are immutable PyTrees.
Random keys are explicit.

## Modules

| Module | Purpose |
| --- | --- |
| `blackwell.beliefs` | `GaussianBelief` and `ParticleBelief` containers. |
| `blackwell.spaces` | Euclidean and right-retraction SE(2) state-space operations. |
| `blackwell.models` | Linear dynamics/observations, SE(2) body motion and known-landmark range-bearing observations. |
| `blackwell.filters.ekf` | Configurable extended Kalman filter. |
| `blackwell.filters.particle` | Configurable bootstrap particle filter with explicit resampling. |
| `blackwell.simulation` | Local additive-Gaussian model simulation. |
| `blackwell.metrics` | RMSE, planar position RMSE and NEES. |

## SE(2) EKF setup

```python
from blackwell.beliefs import GaussianBelief
from blackwell.filters.ekf import ExtendedKalmanFilter
from blackwell.models import range_bearing
from blackwell.models import se2 as se2_models
from blackwell.spaces import se2

filter_ = ExtendedKalmanFilter(se2, se2_models, range_bearing)
dynamics = se2_models.BodyMotion(process_covariance)
observation = range_bearing.KnownLandmarksRangeBearing(
    landmarks, measurement_covariance
)
next_belief = filter_.step(
    belief, dynamics, observation, control, measurement
)
```

The state is `[x, y, heading]`; covariance and controls use local, body-frame
SE(2) tangent coordinates. Compile a configured operation by binding the method:

```python
compiled_step = jax.jit(filter_.step)
```

## Particle filtering

`BootstrapParticleFilter.step` performs prediction and weighting only. Use
`systematic_resample(key, belief)` explicitly when the effective sample size
warrants it. This makes resampling policy visible to the caller.

## Runnable example

The complete SE(2) localisation example simulates noisy known-landmark
observations, estimates the trajectory with the public EKF and reports position
RMSE and NEES:

```console
uv run python examples/se2_localisation.py
```
