# JAX execution model

Blackwell functions are designed to compose with JAX rather than hide it.

## Random keys are explicit

Stochastic operations accept a key as their first argument. Split keys at the
call site so replay and batching remain deterministic:

```python
prediction_key, resampling_key = jax.random.split(key)
belief = particle_filter.step(
    prediction_key, belief, dynamics, observation, control, measurement
)
belief = particle_filter.systematic_resample(resampling_key, belief)
```

## Bind before JIT

Filter and simulator instances contain static Python operation modules. Compile
a bound method:

```python
compiled_step = jax.jit(filter_.step)
```

Model containers, beliefs and arrays remain dynamic arguments and do not trigger
recompilation when only their values change.

## Static shapes

JAX specialises compiled programs to shapes. Keep these fixed inside compiled
loops:

- state and tangent dimensions;
- particle count;
- landmark count;
- measurement shape; and
- trajectory length passed to a single `scan`.

Changing one of these shapes normally triggers a new compilation.

## Batch and scan

Core operations handle one belief. Use:

- `jax.vmap` for independent beliefs or Monte Carlo trials; and
- `jax.lax.scan` for time-ordered state estimation.

The runnable examples demonstrate both patterns. See
[Simulation and evaluation](../guides/simulation-and-evaluation.md) for a
complete trajectory workflow.

## Precision

JAX commonly defaults to 32-bit floats. Enable 64-bit mode *before creating
arrays* when your application needs it:

```python
from jax import config

config.update("jax_enable_x64", True)
```

Use one floating dtype consistently across states and covariance matrices.
