# Installation

Blackwell requires Python 3.10 or newer. Install the latest release from PyPI.

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

The current release is `0.0.3`, which adds Python 3.10 support to `0.0.2`'s SE(3) API;
the earlier `0.0.1` release provides Euclidean and SE(2) geometry. Blackwell remains
pre-alpha: the supported surface is deliberately small, and API changes are
possible before version 1.0.

## Python and JAX versions

Blackwell requires `jax>=0.6`. JAX declares which Python versions each release
supports, so installers select a compatible version automatically:

- Python 3.10 selects JAX 0.6.2.
- Python 3.11 and newer can select newer compatible JAX releases.

The installer uses the interpreter running the command; having a newer Python
installed does not change an existing Python 3.10 environment. To choose 3.11:

```console
python3.11 -m pip install --upgrade blackwell jax
```

Existing environments may retain an already installed compatible JAX version.
Use `--upgrade` when you want to update it. For development, the universal
`uv.lock` records separate resolutions for each Python version:

```console
uv sync --locked --all-extras --python 3.10
# Or select a newer interpreter and its corresponding locked JAX version:
uv sync --locked --all-extras --python 3.11
```

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
