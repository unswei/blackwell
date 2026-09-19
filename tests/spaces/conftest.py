"""Precision contexts that also work with the minimum supported JAX version."""

import jax
import jax.numpy as jnp
import pytest


@pytest.fixture(params=[jnp.float32, jnp.float64], ids=["float32", "float64"])
def precision(request):
    original = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", request.param == jnp.float64)
    try:
        yield request.param, 3e-6 if request.param == jnp.float32 else 2e-10
    finally:
        jax.config.update("jax_enable_x64", original)
