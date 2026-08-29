# Development

## Local environment

Blackwell uses `uv.lock` for a reproducible development environment:

```console
git clone https://github.com/unswei/blackwell.git
cd blackwell
uv sync --all-extras
```

Run the same core checks used in continuous integration:

```console
uv run ruff check .
uv run pytest
uv run python -m build
uv run mkdocs build --strict
```

See the repository's [contribution guide](https://github.com/unswei/blackwell/blob/main/CONTRIBUTING.md)
for branch, testing and documentation expectations.

## Repository layout

```text
src/blackwell/
  beliefs.py           immutable Gaussian and particle containers
  spaces/              Euclidean and SE(2) geometry
  models/              dynamics and observation families
  filters/             EKF and bootstrap particle inference
  simulation.py        reproducible model rollouts
  metrics.py           error and consistency metrics
  _experiments/        private regression/reference implementations
examples/              runnable public-API workflows
tests/                 numerical and transformation tests
docs/                  task guides, concepts and reference page sources
```

## Documentation locally

```console
uv run mkdocs serve
```

Open `http://127.0.0.1:8000`. Source and navigation changes reload
automatically. Always run the strict build before submitting a change; it
rejects broken internal links, invalid snippets and documentation warnings.

The API reference is generated with mkdocstrings from public source docstrings.
Put exact arguments, returns, shapes and edge cases in the source; put task
context and complete workflows in the guides.

## Private experiments

The implementation began with four private experiments: linear Gaussian
filtering, SE(2) range-bearing EKF localisation, particle localisation and
batched Monte Carlo runs. They remain under `blackwell._experiments` as
regression references, but are not supported public API. New user code should
use the modules documented in the [API reference](reference/index.md).

## Release state

The current public release is `0.0.1`. Releases use semantic versioning, with
API stability expected only from version 1.0 onwards. Each release is built as
a wheel and source distribution, verified in a clean environment, published to
PyPI through GitHub Actions Trusted Publishing, and documented in the
[changelog](https://github.com/unswei/blackwell/blob/main/CHANGELOG.md).
