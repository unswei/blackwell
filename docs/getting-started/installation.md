# Installation

Blackwell requires Python 3.11 or newer. Install the latest release from PyPI.

=== "pip"

    ```console
    python -m pip install blackwell
    ```

=== "uv"

    ```console
    uv add blackwell
    ```

Verify the installation:

```console
python -c "import blackwell; print(blackwell.__version__)"
```

The first published release reports version `0.0.1`. Blackwell remains
pre-alpha: the supported surface is deliberately small, and API changes are
possible before version 1.0.

## JAX platform choice

The core dependency installs the standard JAX build. CPU execution is enough
for the examples and many small localisation problems. GPU/TPU JAX packages are
deliberately not pinned because the correct wheel depends on your accelerator,
driver and platform.

Follow the [official JAX installation guide](https://docs.jax.dev/en/latest/installation.html)
when you need accelerator support.

!!! warning "Install JAX once"

    If your environment already has an accelerator-specific JAX installation,
    install Blackwell into that same environment. Avoid replacing it afterwards
    with an incompatible CPU-only JAX wheel.

## Optional plotting support

Examples run without plotting. To save their Matplotlib figures, clone the
repository and install the plotting extra:

```console
git clone https://github.com/unswei/blackwell.git
cd blackwell
python -m pip install -e ".[plot]"
python examples/se2_localisation.py --plot localisation.png
```

## Development checkout

Blackwell uses [uv](https://docs.astral.sh/uv/) for its reproducible environment:

```console
git clone https://github.com/unswei/blackwell.git
cd blackwell
uv sync --all-extras
uv run pytest
```

Continue with the [five-minute localisation](quickstart.md), or see
[Platforms and JAX](../platforms.md) for supported Python versions and backend
notes.

## Development version

To test the current `main` branch before the next release, install directly
from GitHub:

```console
python -m pip install "blackwell @ git+https://github.com/unswei/blackwell.git@main"
```

Development versions may contain unreleased API changes. Prefer the PyPI
release for reproducible work.
