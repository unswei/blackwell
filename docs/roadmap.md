# Roadmap and the SE(3) release

## Relationship to the plan

The 26 August 2026 *Blackwell v0.1: Minimal Useful Library Specification*
defines a broader milestone than the first published version, 0.0.1. The
next release, **0.0.2**, adds SE(3) geometry and uncertainty propagation as a
focused step towards that milestone. It does not declare the full v0.1
acceptance criteria complete.

The planning sources live alongside the software in the project's `meta/`
directory: `blackwell_minimal_useful_library_spec.md`,
`blackwell_book_outline_revised.md` and `blackwell_book_writing_guide.md`.
This comparison was made on 20 September 2026.

| Planned commitment | SE(3) implementation |
| --- | --- |
| Explicit state representation (§6.3) | `[x, y, z, qx, qy, qz, qw]`, preserving the specified `xyzw` ordering |
| Manifold uncertainty (§3.4, §11.2) | Six-dimensional right/body covariance, retraction, local coordinates and chart transport |
| Functional JAX core (§3.3, §3.6) | Array operations, immutable beliefs, explicit batching and random keys |
| Small public surface (§3.1–3.2) | One `spaces.se3` module; application examples use generic autodifferentiation |
| Separation of geometry, models and inference (§3.5) | Existing Gaussian and particle contracts and filter kernels are unchanged |
| Numerical correctness (§18) | Matrix-exponential, finite-difference, analytical-Jacobian, covariance and correlated Monte Carlo checks |
| Book chapter 7 and geometry writing conventions | Documented frame direction, Hamilton quaternion convention, tangent order, right perturbations and branch limits |

The original draft sketches classes such as `SE3()` and `Gaussian(..., space)`.
The published 0.0.1 API instead uses static operation modules and immutable
`GaussianBelief`/`ParticleBelief` containers with the space passed separately.
This addition follows that established API; it does not introduce a competing
class hierarchy or rename existing symbols. The interface differences already
present between 0.0.1 and the planning draft need a separate v0.1 API review.

## Remaining v0.1 work

The wider plan still calls for work beyond this SE(3) release:

- Standalone SO(2)/SO(3), product spaces and general model wrappers.
- Missing-observation handling, richer filter diagnostics and sequence helpers.
- Likelihood, NIS, confidence coverage and plotting support beyond existing
  RMSE/NEES and example plots.
- Broader float32/float64 validation, performance benchmarks and recorded
  NVIDIA GPU and Jetson Orin smoke tests.
- A review of the final public API and all twelve v0.1 acceptance criteria.

Chapter 19's inertial-estimation workflows require velocity/bias states and
IMU models beyond a single rigid transform. SE(3) is a foundation for that
work, not a completed inertial-estimation subsystem. Smoothing, factor graphs,
ROS integration and planning remain outside this release.

## 0.0.2 release checks

Local validation on 20 September 2026 used macOS ARM CPU and Python 3.12.13:

- 91 tests passed with the development lock's JAX 0.11.1.
- All 91 tests and both release examples passed against the built wheel in a
  clean environment using the declared minimum JAX 0.6.0.
- SE(3) geometry and tangent Jacobians were checked in float32 and float64.
- The seeded 50,000-point Monte Carlo comparison differed from the linearised
  covariance by 0.69%; dropping cross-correlation raised the error to about 29%.
- Lint, strict documentation build, source/wheel build and strict distribution
  metadata checks passed.

Run from the software checkout:

```console
uv sync --locked --all-extras
uv run ruff check .
uv run pytest
uv run python examples/quickstart.py
uv run python examples/se3_uncertainty.py
uv run python -m build
uv run mkdocs build --strict
```

Before tagging, review the changes, confirm the Linux/macOS CI checks, set the
release date in the changelog and `CITATION.cff`, and verify the built wheel in
a clean environment. The release workflow tests the installed wheel and both
examples before PyPI publication. A pushed `v0.0.2` tag triggers publication;
building locally does not publish anything.

The local CPU checks do not establish GPU or Jetson validation. The existing
[platform policy](platforms.md) continues to apply. The broader platform gates
in the original v0.1 plan remain open until their results are recorded.
