# Public interface design

The four private experiments establish the public API direction. This document
records the contract to implement next; it does not make the private experiment
modules supported API.

## Design decisions

- Numerical states, tangent vectors, measurements, controls and covariances are
  JAX arrays. Public data containers are immutable `NamedTuple` PyTrees.
- A **state space** owns `retract`, `local_coordinates` and covariance transport.
  Filters do not special-case SE(2), angle wrapping or future manifolds.
- A **dynamics model** owns state propagation, the tangent transition Jacobian
  and process covariance. An **observation model** owns predicted measurements,
  their tangent Jacobian, measurement covariance and residual operation.
- A **filter** only performs inference from those operations. It must not own
  landmarks, noise parameters or a particular state representation.
- Random keys are the first argument to every stochastic operation. Kernels do
  not create or retain keys internally.
- Functions operate on one belief by default. Callers compose `jax.vmap` for
  independent batches and `jax.lax.scan` for trajectories. Particle counts,
  landmark counts and other array shapes are static under JIT.

## Core data containers

```python
class GaussianBelief(NamedTuple):
    mean: Array                 # State in the model's state space.
    covariance: Array           # (tangent_dim, tangent_dim), at `mean`.

class ParticleBelief(NamedTuple):
    particles: Array            # (particle_count, *state_shape)
    weights: Array              # (particle_count,), normalised.
```

For Euclidean spaces, `mean` and its tangent have the same coordinates. For a
manifold, the covariance is always expressed in the tangent coordinates at the
belief mean. `GaussianBelief` deliberately does not retain a reference to a
state space; passing that separately keeps it an uncomplicated JAX PyTree.

## State-space contract

The first supported state space will be `blackwell.spaces.se2`. Its state is
`[x, y, heading]` and its tangent is `[forward, lateral, turn]`, both with shape
`(3,)`. It uses a right retraction:

```python
retract(pose, tangent) -> pose
local_coordinates(reference, pose) -> tangent
transport(reference, target, covariance) -> covariance
```

`retract(pose, tangent)` is `pose compose Exp(tangent)`. Consequently, SE(2)
covariances are body-frame, local tangent covariances. Angle-bearing measurement
models must define residuals with principal-angle wrapping. Euclidean support
will provide the same three operations as ordinary addition, subtraction and an
identity covariance transport.

## Model contract

Models are immutable PyTree data with pure module-level functions. This avoids
putting Python callables inside a model PyTree, which would complicate JIT and
batching. A model family provides the following operations:

```python
propagate(state, control, dynamics) -> state
transition_jacobian(state, control, dynamics) -> Array
process_covariance(state, control, dynamics) -> Array

observe(state, observation) -> Array
observation_jacobian(state, observation) -> Array
measurement_covariance(state, observation) -> Array
measurement_residual(measurement, expected, observation) -> Array
```

The first concrete families will be `LinearDynamics` / `LinearObservation` and
SE(2) body-motion / known-landmark range-bearing models. A range-bearing model
may hold landmarks and a per-landmark covariance, but it must not embed EKF or
particle-filter logic.

## Filter contract

The supported kernels live under `blackwell.filters`. A filter instance captures
the state-space and the static operation modules; its methods receive the
dynamic model parameter values:

```python
filter = ekf.ExtendedKalmanFilter(
    state_space=se2,
    dynamics=se2_motion,
    observation=range_bearing,
)
filter.predict(belief, dynamics_model, control) -> GaussianBelief
filter.update(belief, observation_model, measurement) -> GaussianBelief
filter.step(belief, dynamics_model, observation_model, control, measurement)

particles = particle.BootstrapParticleFilter(
    state_space=se2,
    dynamics=se2_motion,
    observation=range_bearing,
)
particles.predict(key, belief, dynamics_model, control) -> ParticleBelief
particles.update(belief, observation_model, measurement) -> ParticleBelief
particles.systematic_resample(key, belief) -> ParticleBelief
```

This keeps Python callables outside the dynamic JAX PyTree arguments. To compile
an operation, bind the method before calling `jax.jit`, for example
`compiled_step = jax.jit(filter.step)`.

EKF prediction transports the old tangent covariance through the model's
transition Jacobian. EKF updates retract the correction, use the Joseph
covariance form, and transport covariance into the corrected tangent coordinate
system. Particle filtering applies process noise through the dynamics model,
combines log likelihoods stably, and treats resampling as an explicit operation
rather than an unavoidable side effect of every update.

## Public package layout

```text
blackwell/
  beliefs.py
  spaces/
    euclidean.py
    se2.py
  models/
    linear.py
    se2.py
    range_bearing.py
  filters/
    ekf.py
    particle.py
  metrics.py
  simulation.py
  _experiments/                 # retained as non-supported reference code
```

`blackwell.__init__` will export only the principal beliefs and selected
state-space constructors. Algorithms remain explicit imports, for example
`from blackwell.filters import ekf`, so that the top-level namespace stays
small and future filters do not force breaking export changes.

## Migration from the experiments

| Private experiment | Public destination | Required change |
| --- | --- | --- |
| `linear_gaussian` | `models.linear`, `filters.ekf` | Split matrices into dynamics and observation models. |
| `se2_range_bearing` | `spaces.se2`, `models.se2`, `models.range_bearing` | Move group operations out of filter-specific code and separate model operations. |
| `particle_localisation` | `filters.particle` | Generalise over the state-space and model contract; make resampling optional. |
| `monte_carlo` | `simulation`, `metrics` | Generalise trial generation and keep RMSE/NEES independent of the EKF runner. |

The experiment modules remain until the public implementations match their
numerical regression tests. They will then become compact examples or be
removed in a separate compatibility-reviewed change.

## Implementation acceptance criteria

- Public filter kernels work under `jax.jit` and `jax.vmap` for their documented
  shapes, and preserve immutable PyTree inputs.
- The SE(2) Jacobians agree with automatic differentiation away from known
  geometric singularities; bearing residuals cross the angle branch cut safely.
- Covariance outputs are symmetric and the Joseph update remains
  positive-semidefinite within numerical tolerance.
- Particle weighting is normalised in log space; zero process covariance and
  deterministic resampling cases remain supported.
- Simulation and metrics operate on batches without Python loops over trials.
- At least one public, runnable SE(2) localisation example demonstrates the
  supported API end to end.
