# Platforms and JAX

Blackwell supports Python 3.11 and newer. Continuous integration exercises
Linux/Python 3.11 and macOS/Python 3.13; other JAX-supported combinations may
work but are not part of the current validation matrix.

## CPU

The default dependency installs standard JAX. CPU is sufficient for the
examples and often fastest to set up for low-dimensional estimators.

Confirm the active backend:

```console
python -c "import jax; print(jax.default_backend()); print(jax.devices())"
```

## GPU and TPU

Blackwell deliberately does not pin accelerator-specific JAX wheels because
the correct package depends on accelerator, driver and operating system.
Install the suitable build from the
[official JAX installation guide](https://docs.jax.dev/en/latest/installation.html),
then install Blackwell in the same environment.

Accelerators provide the clearest benefit for large particle sets, batched
Monte Carlo evaluation and large model batches. For one small EKF, compilation
and transfer overhead can outweigh faster device arithmetic.

## Apple silicon

The standard CPU backend is the supported path on macOS. The project does not
currently test experimental Metal acceleration.

## Embedded NVIDIA platforms

Jetson/JetPack validation is not yet part of the release matrix. JAX wheel,
CUDA, cuDNN and Python compatibility must be matched to the device image. Treat
deployment there as an integration exercise and record the full software stack
with performance results.

## Precision

JAX defaults to 32-bit floating point in many environments. Enable 64-bit mode
before creating arrays when conditioning or long-horizon consistency demands
it:

```python
from jax import config

config.update("jax_enable_x64", True)
```

Use one dtype consistently across state, controls, measurements and covariance
parameters. A dtype change causes JIT recompilation.
