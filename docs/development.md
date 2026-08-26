# Development

Create the reproducible development environment and run the current checks:

```console
uv sync --all-extras
uv run pytest
uv run ruff check .
uv run python -m build
```

The implementation will start with four private experiments: linear Gaussian
filtering, SE(2) range-bearing EKF localisation, particle localisation using
the same models, and batched Monte Carlo runs. They will resolve the model and
PyTree details before public inference APIs are added.
