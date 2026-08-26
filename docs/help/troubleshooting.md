# Troubleshooting

## Installation cannot find a release

Blackwell is not yet published on PyPI. Install the current Git revision:

```console
python -m pip install "blackwell @ git+https://github.com/unswei/blackwell.git"
```

For an editable checkout, use `python -m pip install -e .` or `uv sync`.

## JAX reports an accelerator or driver error

Blackwell does not pin accelerator-specific JAX packages. Install the JAX build
matching your operating system, accelerator and driver using the
[official JAX installation guide](https://docs.jax.dev/en/latest/installation.html).
Confirm JAX independently:

```console
python -c "import jax; print(jax.devices())"
```

## JIT rejects a filter argument

Compile a configured bound method:

```python
compiled_step = jax.jit(filter_.step)
```

Do not pass the filter itself as a dynamic argument. Its state-space and model
operation modules are static Python objects; beliefs and model parameter
containers are the dynamic PyTrees.

## JAX recompiles repeatedly

Check whether particle count, landmark count, measurement shape, dtype or
trajectory length is changing. JAX specialises programs to shapes and dtypes.
Value changes alone should not cause recompilation.

## Bearings jump near pi

Use `range_bearing.measurement_residual` or another topology-aware observation
family. Subtracting raw bearings gives a discontinuity at the principal-angle
cut. Use radians throughout.

## A range-bearing Jacobian contains NaN or infinity

The range-bearing observation is singular when a landmark is exactly at the
robot position. Remove that observation or model the sensor behaviour at zero
range explicitly.

## Particle weights become NaN

Check that:

- all prior weights are non-negative, finite and normalised;
- measurement covariance is positive definite;
- state and measurement arrays contain no `NaN`; and
- measurement rows use the same landmark order as the model.

## Covariance is not positive semidefinite

Blackwell symmetrises EKF covariance output and uses the Joseph update form, but
it cannot correct an invalid model covariance or severe numerical scaling.
Verify process, measurement and initial covariances; use consistent units; and
consider enabling JAX 64-bit mode for ill-conditioned problems.

## Results differ slightly across machines

Small floating-point differences are normal across JAX backends and devices.
Use the same dtype, JAX version and backend for strict reproducibility. Random
results are reproducible only when keys and inputs are also identical.

If the problem persists, open a
[GitHub issue](https://github.com/unswei/blackwell/issues) with a minimal script,
Python/JAX versions, `jax.devices()` output and the complete traceback.
