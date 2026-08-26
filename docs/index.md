# Blackwell

Blackwell is a JAX-native library for probabilistic robotics. Its deliberately
compact pre-alpha API provides manifold-aware uncertainty, Gaussian and particle
beliefs, nonlinear filtering, simulation and uncertainty evaluation.

The public interface was derived from four private experiments. It supports
Euclidean and SE(2) state spaces, linear and range-bearing models, generic EKF
and bootstrap particle-filter kernels, batched evaluation metrics and runnable
localisation examples.

Run the end-to-end demonstration from the repository root:

```console
uv run python examples/se2_localisation.py
```
