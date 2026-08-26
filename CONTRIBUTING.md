# Contributing to Blackwell

Blackwell welcomes focused bug reports, documentation improvements and small,
well-tested estimation features. The public API is pre-alpha, but changes
should still be deliberate and documented.

## Set up a checkout

Install [uv](https://docs.astral.sh/uv/), then:

```console
git clone https://github.com/unswei/blackwell.git
cd blackwell
uv sync --all-extras
```

## Make a change

1. Create a short, descriptive branch without a tool-specific prefix.
2. Add or update tests for numerical and transformation behaviour.
3. Update guides and API docstrings when public behaviour changes.
4. Keep model parameters in JAX PyTrees and Python operation modules static.
5. Preserve explicit random-key handling and immutable inputs.

## Run the checks

```console
uv run ruff check .
uv run pytest
uv run python examples/quickstart.py
uv run python -m build
uv run mkdocs build --strict
```

The continuous-integration matrix runs Linux/Python 3.11 and macOS/Python 3.13.
Source documentation is built strictly on pull requests. The public manual is
published from the EICRL lab website repository at
<https://unswei.github.io/blackwell/>.

## Test numerical work

Depending on the change, include checks for:

- array shapes and immutable PyTree behaviour;
- `jax.jit`, `jax.vmap` or `jax.lax.scan` compatibility;
- analytical Jacobians against automatic differentiation;
- angle residuals across the principal branch cut;
- covariance symmetry and positive-semidefiniteness;
- normalised particle weights and deterministic seeded cases; and
- degenerate or zero-noise inputs where they are supported.

Use tolerances that reflect dtype and conditioning. Do not rely on exact
floating-point equality unless the operation is mathematically exact under the
chosen representation.

## Write documentation

- Start from a user task and include a runnable command.
- Explain state, tangent, control and measurement shapes.
- Use Australian/British English.
- Add public API details to source docstrings; generated reference pages read
  those docstrings directly.
- Build with `uv run mkdocs build --strict` before opening a pull request.

## Report a bug

Open a [GitHub issue](https://github.com/unswei/blackwell/issues) with a minimal
reproduction, expected and actual behaviour, Python and JAX versions, device
output from `jax.devices()` and the complete traceback.
