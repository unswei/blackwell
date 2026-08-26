# Blackwell

Blackwell is a small, JAX-native probabilistic robotics library. It will provide
manifold-aware Gaussian beliefs, weighted particles, common state spaces, an
extended Kalman filter, a bootstrap particle filter, simulation helpers and
uncertainty metrics.

The project is currently in its interface-spike stage. The package scaffold is
installable, but the public estimation API is not implemented yet. The first
experiments will settle the state, model and PyTree interfaces before those APIs
are published.

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
```

GPU-enabled JAX is deliberately not pinned by this project. Install the JAX
extra appropriate to the target platform, following the
[official JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

## Repository boundaries

This repository contains the Blackwell software, its concise documentation and
runnable examples. The Blackwell book, extended notebooks and teaching
materials live in a separate repository and are intentionally not tracked here.

## Licence

Blackwell is licensed under the [Apache License 2.0](LICENSE).
