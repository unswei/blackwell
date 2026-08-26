# Examples

Every example is a standalone script under `examples/`, runs without plotting
dependencies and uses only Blackwell's public API.

| Example | What it demonstrates | Run |
| --- | --- | --- |
| [Quick start](../getting-started/quickstart.md) | One compiled SE(2) EKF step | `uv run python examples/quickstart.py` |
| [Linear Kalman filter](linear-kalman-filter.md) | Euclidean model, `jax.lax.scan` | `uv run python examples/linear_kalman_filter.py` |
| [SE(2) EKF localisation](se2-localisation.md) | Simulation, manifold EKF, RMSE and NEES | `uv run python examples/se2_localisation.py` |
| [SE(2) particle localisation](particle-localisation.md) | Weighted particles, ESS and resampling | `uv run python examples/particle_localisation.py` |

Clone the repository and install all extras to reproduce figures:

```console
git clone https://github.com/unswei/blackwell.git
cd blackwell
uv sync --all-extras
```

The examples fix their random seeds. Identical software and hardware should
therefore reproduce their trajectories, although small floating-point
differences can occur across JAX backends.
