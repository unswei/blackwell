# Changelog

All notable changes to Blackwell are documented here. Blackwell follows
[Semantic Versioning](https://semver.org/), with no API compatibility guarantee
before version 1.0.

## [0.0.2] - 2026-09-20

### Added

- SE(3) composition, inversion, point action, exponential/logarithm, right
  retraction, local coordinates, adjoint and covariance transport.
- Documented `xyzw` Hamilton quaternion encoding and translation-first body
  tangents, including principal-logarithm limits and equivalent quaternion signs.
- Generic autodifferentiation examples for transform composition/inversion and
  uncertain points, retaining cross-covariance for correlated inputs.
- Float32/float64 geometry and Jacobian checks, a 50,000-sample Monte Carlo
  comparison and integration tests for existing Gaussian and particle filters.
- A roadmap comparison separating this SE(3) addition from the full v0.1 plan.

Existing public operations and belief containers remain compatible. SE(3)
means have seven stored coordinates and six covariance dimensions; use
`blackwell.spaces.se3` and the documented tangent convention.

## [0.0.1] - 2026-08-29

First public release.

### Added

- Immutable Gaussian and particle belief containers for JAX transformations.
- Euclidean and SE(2) state spaces with tangent-coordinate uncertainty.
- Linear, SE(2) body-motion and landmark range-bearing models.
- Extended Kalman and bootstrap particle filters.
- Reproducible simulation, RMSE and NEES evaluation utilities.
- Runnable localisation examples and a complete online manual.

[0.0.1]: https://github.com/unswei/blackwell/releases/tag/v0.0.1

[0.0.2]: https://github.com/unswei/blackwell/releases/tag/v0.0.2
