# Simulation and evaluation

Blackwell exposes simulation and metrics alongside inference so an estimator
can be checked on reproducible trajectories before it meets real sensor data.

## Simulate from the same models

`Simulator` uses the state-space, dynamics and observation operations used by
the filters:

```python
import jax
import jax.numpy as jnp

from blackwell.models import range_bearing
from blackwell.models import se2 as se2_models
from blackwell.simulation import Simulator
from blackwell.spaces import se2

simulator = Simulator(se2, se2_models, range_bearing)
controls = jnp.tile(jnp.array([0.12, 0.0, 0.025]), (80, 1))
states, measurements = simulator.rollout(
    jax.random.key(1),
    initial_state=jnp.array([0.0, 0.0, 0.0]),
    dynamics_model=dynamics,
    observation_model=observation,
    controls=controls,
)
```

Each step propagates the deterministic model, samples local process noise,
retracts it onto the state space and then samples measurement noise. Passing the
same key and inputs reproduces the same rollout.

!!! note "State-dependent process covariance"

    `Simulator.step` queries process covariance at the pre-transition state,
    matching the current filter model contract. Observation covariance is
    queried at the simulated next state.

## Compute errors in the right coordinates

Subtracting SE(2) pose arrays gives the wrong heading behaviour and ignores the
body frame. Use the state space:

```python
errors = jax.vmap(se2.local_coordinates)(estimated_states, states)
```

The resulting errors share coordinates with the filter covariance.

## Report accuracy and consistency

```python
from blackwell.metrics import mean_nees, position_rmse

rmse = position_rmse(errors)
nees = mean_nees(errors, estimated_covariances)
```

Position RMSE answers “how far away was the estimate?” NEES answers “was that
error plausible under the reported covariance?” For a calibrated Gaussian
estimator, the expected NEES is approximately the tangent dimension when
averaged over enough independent trials.

One trajectory is useful for debugging, not statistical validation. For a
meaningful consistency check:

1. split independent keys for many trials;
2. use `jax.vmap` to run them in parallel;
3. compute NEES at each time and across trials; and
4. compare against chi-squared confidence bounds appropriate to the sample
   count and tangent dimension.

## Move from simulation to sensors

Keep the estimator code unchanged and replace only the source of controls and
measurements. Confirm:

- timestamps and control integration intervals agree;
- all headings and bearings are radians;
- landmarks and poses share a world frame;
- process covariance describes local post-motion uncertainty; and
- measurement ordering matches landmark ordering.

The [SE(2) example](../examples/se2-localisation.md) shows simulation, filtering
and evaluation end to end. The [metrics reference](../reference/metrics.md)
documents array shapes and return values.
