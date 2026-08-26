# Blackwell

Blackwell is a small, JAX-native probabilistic robotics library. Its pre-alpha
public API provides manifold-aware Gaussian beliefs, weighted particles,
Euclidean and SE(2) state spaces, an extended Kalman filter, a bootstrap
particle filter, simulation helpers and uncertainty metrics.

The API emerged from four private interface experiments and is deliberately
compact. It is ready for hands-on evaluation, but remains pre-1.0: expect the
package to evolve as additional state spaces and model families are introduced.

## Design commitments

- JAX is the sole numerical backend.
- The mathematical core is functional and immutable.
- State uncertainty lives in tangent coordinates, including for robotic
  manifolds such as SE(2) and SE(3).
- Models describe uncertainty; filters perform inference.
- Random keys, JIT compilation and covariance regularisation remain explicit.
- Evaluation and consistency metrics are core features rather than add-ons.

## Development

Blackwell uses [uv](https://docs.astral.sh/uv/) for its reproducible development
environment.

```console
uv sync --all-extras
uv run pytest
uv run ruff check .
uv run python -m build
uv run python examples/se2_localisation.py
```

GPU-enabled JAX is deliberately not pinned by this project. Install the JAX
extra appropriate to the target platform, following the
[official JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

## Repository boundaries

This repository contains the Blackwell software, its concise documentation and
runnable examples.

## Licence

Blackwell is licensed under the [Apache License 2.0](LICENSE).
