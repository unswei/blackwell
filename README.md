# Blackwell

[![Test](https://github.com/unswei/blackwell/actions/workflows/test.yml/badge.svg)](https://github.com/unswei/blackwell/actions/workflows/test.yml)
[![Documentation](https://github.com/unswei/blackwell/actions/workflows/docs.yml/badge.svg)](https://github.com/unswei/blackwell/actions/workflows/docs.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB)](https://www.python.org/)
[![Apache-2.0](https://img.shields.io/badge/licence-Apache--2.0-blue)](LICENSE)

Blackwell is a compact, JAX-native probabilistic-robotics library. It provides
manifold-aware Gaussian beliefs, weighted particles, Euclidean and SE(2) state
spaces, an extended Kalman filter, a bootstrap particle filter, reproducible
simulation and uncertainty metrics.

It is useful today for planar localisation, linear state estimation and
estimator research, while remaining deliberately pre-alpha: expect API changes
before version 1.0.

**[Documentation](https://unswei.github.io/blackwell/)** ·
**[Five-minute localisation](https://unswei.github.io/blackwell/getting-started/quickstart/)** ·
**[Examples](https://unswei.github.io/blackwell/examples/)** ·
**[API reference](https://unswei.github.io/blackwell/api/)**

The public manual is part of the EICRL lab website, giving Blackwell the same
navigation, accessibility and visual language as the lab's other resources.
This repository retains its source documentation for local validation.

## Install

Blackwell requires Python 3.11 or newer and is not yet on PyPI. Install the
current Git revision:

```console
python -m pip install "blackwell @ git+https://github.com/unswei/blackwell.git"
```

Accelerator-specific JAX packages are intentionally not pinned. See the
[installation guide](https://unswei.github.io/blackwell/getting-started/installation/)
for GPU/TPU and development setups.

## A first SE(2) estimate

```python
import jax
import jax.numpy as jnp

from blackwell import GaussianBelief
from blackwell.filters.ekf import ExtendedKalmanFilter
from blackwell.models import range_bearing
from blackwell.models import se2 as se2_models
from blackwell.spaces import se2

filter_ = ExtendedKalmanFilter(se2, se2_models, range_bearing)
dynamics = se2_models.BodyMotion(
    process_covariance=jnp.diag(jnp.array([0.01, 0.01, 0.0025]))
)
observation = range_bearing.KnownLandmarksRangeBearing(
    landmarks=jnp.array([[5.0, 0.0], [0.0, 5.0]]),
    measurement_covariance=jnp.diag(jnp.array([0.08, 0.02])),
)
belief = GaussianBelief(
    mean=jnp.array([0.0, 0.0, 0.0]),
    covariance=jnp.diag(jnp.array([0.5, 0.5, 0.1])),
)

belief = jax.jit(filter_.step)(
    belief,
    dynamics,
    observation,
    jnp.array([0.3, 0.0, 0.04]),
    range_bearing.observe(jnp.array([0.35, 0.08, 0.05]), observation),
)
print(belief.mean)
```

SE(2) states are `[x, y, heading]`; controls and covariance use local
body-frame tangent coordinates `[forward, lateral, turn]`.

## What is included?

- Euclidean and right-retraction SE(2) state spaces
- Linear dynamics and observations
- SE(2) body-motion and known-landmark range-bearing models
- Manifold-aware extended Kalman filtering
- Bootstrap particle filtering with explicit ESS and systematic resampling
- Reproducible trajectory simulation
- RMSE, planar position RMSE and NEES metrics
- Runnable linear, SE(2) EKF and particle-localisation examples

Blackwell does not yet include SE(3), smoothing, SLAM state augmentation,
data association, sensor drivers or production persistence. See
[Choose an estimator](https://unswei.github.io/blackwell/getting-started/choose-an-estimator/)
for the current fit and limits.

## Design commitments

- JAX is the sole numerical backend.
- Mathematical kernels are pure and beliefs are immutable PyTrees.
- Manifold uncertainty lives in local tangent coordinates.
- Geometry, stochastic models and inference remain separate.
- Random keys, JIT compilation and resampling policy stay explicit.
- Simulation and consistency metrics are part of the public API.

## Run the examples

```console
git clone https://github.com/unswei/blackwell.git
cd blackwell
uv sync --all-extras
uv run python examples/quickstart.py
uv run python examples/linear_kalman_filter.py
uv run python examples/se2_localisation.py --plot localisation.png
uv run python examples/particle_localisation.py --plot particles.png
```

## Contribute

Set up the reproducible development environment and run all local checks:

```console
uv sync --all-extras
uv run ruff check .
uv run pytest
uv run python -m build
uv run mkdocs build --strict
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for workflow, test and documentation
expectations.

## Licence

Blackwell is licensed under the [Apache License 2.0](LICENSE).
