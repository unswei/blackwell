# Development

Create the reproducible development environment and run the current checks:

```console
uv sync --all-extras
uv run pytest
uv run ruff check .
uv run python -m build
```

The implementation began with four private experiments: linear Gaussian
filtering, SE(2) range-bearing EKF localisation, particle localisation using the
same models, and batched Monte Carlo runs. Their shared interfaces now underpin
the supported pre-alpha API; the experiments remain as regression references.

The completed linear-Gaussian experiment lives in
`blackwell._experiments.linear_gaussian`. It keeps the model and Gaussian belief
as JAX PyTrees, and its pure prediction and update functions are covered by a
JIT compilation test. As a private namespace, it remains outside the supported
public API.

The completed SE(2) range-bearing localisation experiment lives in
`blackwell._experiments.se2_range_bearing`. Its Gaussian covariance is expressed
in the nominal pose's local tangent coordinates; it uses explicit SE(2) group
operations and tests bearing innovations across the angle branch cut.

The completed bootstrap particle-localisation experiment lives in
`blackwell._experiments.particle_localisation`. It reuses the same SE(2) motion
and range-bearing model, keeps random keys explicit, and provides systematic
resampling with an effective-sample-size diagnostic.

The completed batched Monte Carlo experiment lives in
`blackwell._experiments.monte_carlo`. It runs independent noisy SE(2) EKF trials
through JAX vectorisation and reports tangent errors, position RMSE, and NEES.
