# Models and filters

Blackwell keeps stochastic modelling separate from inference.

## State-space operations

A state space owns three operations:

- `retract(state, tangent)` applies a local displacement;
- `local_coordinates(reference, state)` computes a local displacement; and
- `transport(reference, target, covariance)` changes a covariance's tangent
  coordinates.

## Dynamics families

A dynamics family supplies pure functions for deterministic propagation, the
tangent transition Jacobian and process covariance. Its parameter container is
an immutable PyTree—for example `LinearDynamics` or `BodyMotion`.

## Observation families

An observation family predicts measurements, computes local-tangent Jacobians,
returns measurement covariance and defines a topology-aware residual. This last
operation is why range-bearing observations can wrap bearings correctly while
linear observations simply subtract.

## Configured inference

Filters capture the *operation modules* as static Python configuration:

```python
filter_ = ExtendedKalmanFilter(se2, se2_models, range_bearing)
```

The data that changes—belief, covariance parameters, landmarks, control and
measurements—stays in method arguments as JAX PyTrees. A bound method can then
be compiled without placing Python callables inside a dynamic PyTree:

```python
compiled_step = jax.jit(filter_.step)
```

This same pattern configures `BootstrapParticleFilter` and `Simulator`.

## Custom model families

Custom families are modules or objects that implement the same function names
and signatures as the built-in families. The required operations are defined by
the internal structural protocols and demonstrated in the
[linear model reference](../reference/models/linear.md). Keep functions pure,
shapes static and all numerical parameters in the model PyTree.
