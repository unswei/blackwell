"""Package-level smoke tests."""

import blackwell


def test_package_exposes_a_version() -> None:
    assert blackwell.__version__ == "0.0.1"
