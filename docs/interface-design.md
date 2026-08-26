# Public interface design

Blackwell separates state geometry, stochastic models and inference. This page
records the current public contract and the reasoning behind it.

## Data and transformations

- Numerical states, tangents, measurements, controls and covariances are JAX
  arrays.
- Public data containers are immutable `NamedTuple` PyTrees.
- Filter and simulator instances capture static Python operation modules.
- Dynamic model values stay in method arguments as JAX PyTrees.
- Random keys are explicit inputs to stochastic operations.
- Kernels operate on one belief; callers compose `vmap` and `scan`.

## State spaces own geometry

A state space implements:

```python
retract(state, tangent) -> state
local_coordinates(reference, state) -> tangent
transport(reference, target, covariance) -> covariance
```

Euclidean operations reduce to addition, subtraction and identity transport.
For SE(2), Blackwell uses a right/body-frame retraction. The state is
`[x, y, heading]`, while its tangent is `[forward, lateral, turn]`.

Filters therefore do not special-case angle wrapping or a particular manifold.
Future state spaces can satisfy the same small contract.

## Models own uncertainty

A dynamics family provides deterministic propagation, its tangent transition
Jacobian and local process covariance. An observation family provides expected
measurements, the observation Jacobian, measurement covariance and a residual
operation.

```python
propagate(state, control, dynamics_model) -> state
transition_jacobian(state, control, dynamics_model) -> Array
process_covariance(state, control, dynamics_model) -> Array

observe(state, observation_model) -> Array
observation_jacobian(state, observation_model) -> Array
measurement_covariance(state, observation_model) -> Array
measurement_residual(measurement, expected, observation_model) -> Array
```

The residual belongs to the observation family because measurement topology is
model-specific. Range-bearing models wrap bearing differences; linear models
subtract directly.

## Filters own inference

The EKF receives a local Gaussian belief, uses model Jacobians, retracts the
correction and transports the covariance to the corrected tangent frame. The
particle filter samples local process noise, evaluates model-specific
likelihoods and exposes resampling as a separate policy decision.

Neither filter owns landmarks, model covariances or a particular state
representation. This makes the same inference kernel usable for built-in and
compatible custom model families.

## Supported public surface

```text
blackwell/
  beliefs.py
  spaces/{euclidean,se2}.py
  models/{linear,se2,range_bearing}.py
  filters/{ekf,particle}.py
  simulation.py
  metrics.py
```

`blackwell._experiments` and names beginning with an underscore are private.
The generated [API reference](reference/index.md) is the authoritative list of
supported modules.

## Compatibility posture

Blackwell is pre-alpha. Shapes, geometry conventions and explicit key handling
are intentional design commitments, but names and extension contracts may
still change before version 1.0. Public changes should include tests, updated
source docstrings, a migration note when appropriate and a strict docs build.
