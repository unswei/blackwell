# Platforms

Blackwell is CPU-first and accelerator-capable. The numerical core depends on
JAX only; GPU-enabled JAX remains an explicit platform choice made by the user.

For installation instructions and supported JAX CPU and accelerator platforms,
refer to the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

The v0.1 validation target will include Linux CPU continuous integration, macOS
ARM development, Linux NVIDIA GPU testing and one documented Jetson Orin smoke
test. Exact Jetson software versions will be recorded when validated.
