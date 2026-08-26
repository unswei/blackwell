<div class="bw-hero" markdown>
  <span class="bw-eyebrow">Pre-alpha · open source · Apache-2.0</span>

  # State estimation that speaks JAX

  Blackwell is a compact probabilistic-robotics library for manifold-aware
  Gaussian and particle inference—pure, immutable and ready for `jit`, `vmap`
  and `scan`.

  <div class="bw-actions">
    [Install Blackwell](getting-started/installation.md){ .md-button .md-button--primary }
    [Run the quick start](getting-started/quickstart.md){ .md-button }
    [Browse the API](reference/index.md){ .md-button }
  </div>
</div>

<div class="bw-facts" markdown>
  <div class="bw-fact"><strong>SE(2)-native</strong>Covariance lives in local tangent coordinates.</div>
  <div class="bw-fact"><strong>Transform-ready</strong>Pure kernels compose with JAX transformations.</div>
  <div class="bw-fact"><strong>Two estimators</strong>Extended Kalman and bootstrap particle filters.</div>
  <div class="bw-fact"><strong>Evaluation built in</strong>Simulation, RMSE and NEES are public APIs.</div>
</div>

## What can I do with it today?

<div class="grid cards" markdown>

-   :material-map-marker-path:{ .lg .middle } **Localise a planar robot**

    ---

    Estimate an SE(2) pose from body-frame motion and known-landmark
    range-bearing observations.

    [:octicons-arrow-right-24: EKF localisation guide](guides/extended-kalman-filter.md)

-   :material-chart-bell-curve-cumulative:{ .lg .middle } **Track linear systems**

    ---

    Use the same EKF kernel as an ordinary Kalman filter with Euclidean state
    and linear dynamics.

    [:octicons-arrow-right-24: Linear example](examples/linear-kalman-filter.md)

-   :material-dots-hexagon:{ .lg .middle } **Represent non-Gaussian beliefs**

    ---

    Run a bootstrap particle filter with log-space weighting and explicit
    systematic resampling.

    [:octicons-arrow-right-24: Particle-filter guide](guides/particle-filter.md)

-   :material-flask-outline:{ .lg .middle } **Test consistency**

    ---

    Generate reproducible trajectories and evaluate errors with RMSE and NEES.

    [:octicons-arrow-right-24: Simulation and evaluation](guides/simulation-and-evaluation.md)

</div>

## A first estimate

Configure an estimator by pairing a state space with dynamics and observation
families. Parameter values remain ordinary JAX PyTrees.

```python
import jax

from blackwell.filters.ekf import ExtendedKalmanFilter
from blackwell.models import range_bearing
from blackwell.models import se2 as se2_models
from blackwell.spaces import se2

filter_ = ExtendedKalmanFilter(se2, se2_models, range_bearing)
compiled_step = jax.jit(filter_.step)
```

The [five-minute localisation](getting-started/quickstart.md) continues from
here with a complete, runnable update.

## Design stance

- **JAX is the numerical backend.** Arrays, random keys and compilation remain
  explicit.
- **Geometry owns geometry.** State spaces implement retraction, local
  coordinates and covariance transport; filters stay generic.
- **Models own uncertainty.** Dynamics and observations describe noise and
  Jacobians; filters perform inference.
- **Shape changes are explicit.** Particle count, landmark count and trajectory
  length are static under JIT.

!!! info "Project maturity"

    Blackwell is useful today but remains pre-alpha. The supported surface is
    deliberately small, and API changes are still possible before version 1.0.
    See the [current scope](getting-started/choose-an-estimator.md) before choosing
    it for a long-lived deployment.
