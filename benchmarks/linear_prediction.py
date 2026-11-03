"""Measure preparation, compilation and warmed batched prediction/gradients.

Run from a checkout with ``uv run python benchmarks/linear_prediction.py``.
All reported execution times synchronise with the device. This compares the
same affine outputs and objective on the same inputs; compilation and warmed
execution are measured separately. Results describe this workload and device.
"""

import argparse
import json
import math
import platform
import statistics
import time

import jax
import jax.numpy as jnp
from numpy.testing import assert_allclose

from blackwell.models import linear


def scan_predict(model, initial_state, controls, projection, a, d):
    """Reference recurrence, compiled and batched identically to prediction."""

    def step(state, control):
        state = linear.propagate(state, control, model) + a
        return state, projection @ state + d

    return jax.lax.scan(step, initial_state, controls)[1]


def compile_and_measure(function, args, repeats):
    start = time.perf_counter()
    compiled = jax.jit(function).lower(*args).compile()
    compile_seconds = time.perf_counter() - start
    start = time.perf_counter()
    result = jax.block_until_ready(compiled(*args))
    first_seconds = time.perf_counter() - start
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(compiled(*args))
        samples.append(time.perf_counter() - start)
    memory = compiled.memory_analysis()
    # XLA estimates, not allocator/process peak memory. Aliased buffers count once.
    memory_fields = (
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "temp_size_in_bytes",
        "alias_size_in_bytes",
    )
    memory_estimate = (
        {name: getattr(memory, name) for name in memory_fields}
        if memory is not None
        else None
    )
    return result, {
        "compile_seconds": compile_seconds,
        "first_execution_seconds": first_seconds,
        "warmed_median_seconds": statistics.median(samples),
        "xla_memory_bytes": memory_estimate,
    }


def break_even(preparation, cached, recurrent, *, include_compilation):
    saving = recurrent["warmed_median_seconds"] - cached["warmed_median_seconds"]
    if saving <= 0:
        return None
    setup = preparation["warmed_median_seconds"]
    if include_compilation:
        setup += preparation["compile_seconds"]
        setup += cached["compile_seconds"] - recurrent["compile_seconds"]
    return max(1, math.ceil(setup / saving))


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=positive_int, default=32)
    parser.add_argument("--state-dim", type=positive_int, default=32)
    parser.add_argument("--control-dim", type=positive_int, default=3)
    parser.add_argument("--output-dim", type=positive_int, default=3)
    parser.add_argument("--batch-size", type=positive_int, default=256)
    parser.add_argument("--repeats", type=positive_int, default=50)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", args.dtype == "float64")
    dtype = getattr(jnp, args.dtype)
    n, m, p, horizon = args.state_dim, args.control_dim, args.output_dim, args.horizon
    keys = jax.random.split(jax.random.key(2026), 7)
    model = linear.LinearDynamics(
        0.9 * jnp.eye(n, dtype=dtype)
        + 0.01 * jax.random.normal(keys[0], (n, n), dtype) / math.sqrt(n),
        0.1 * jax.random.normal(keys[1], (n, m), dtype),
        jnp.zeros((n, n), dtype=dtype),
    )
    projection = jax.random.normal(keys[2], (p, n), dtype) / math.sqrt(n)
    a = 0.01 * jax.random.normal(keys[3], (n,), dtype)
    d = 0.01 * jax.random.normal(keys[4], (p,), dtype)
    state = jax.random.normal(keys[5], (n,), dtype)
    controls = jax.random.normal(keys[6], (args.batch_size, horizon, m), dtype)
    jax.block_until_ready((model, projection, a, d, state, controls))

    def prepare(model, projection, a, d):
        return linear.prepare_prediction(
            model, horizon, output_matrix=projection, state_offset=a, output_offset=d
        )

    response, preparation = compile_and_measure(
        prepare, (model, projection, a, d), args.repeats
    )
    dense_args = (response, state, controls)
    scan_args = (model, state, controls, projection, a, d)
    dense_many = jax.vmap(linear.predict_sequence, in_axes=(None, None, 0))
    scan_many = jax.vmap(scan_predict, in_axes=(None, None, 0, None, None, None))
    dense_outputs, dense_prediction = compile_and_measure(
        dense_many, dense_args, args.repeats
    )
    scan_outputs, scan_prediction = compile_and_measure(
        scan_many, scan_args, args.repeats
    )

    def dense_objective(response, state, controls):
        outputs = linear.predict_sequence(response, state, controls)
        return jnp.sum(outputs**2) + 0.1 * jnp.sum(controls**2)

    def scan_objective(model, state, controls, projection, a, d):
        outputs = scan_predict(model, state, controls, projection, a, d)
        return jnp.sum(outputs**2) + 0.1 * jnp.sum(controls**2)

    dense_gradients, dense_gradient = compile_and_measure(
        jax.vmap(jax.grad(dense_objective, argnums=2), in_axes=(None, None, 0)),
        dense_args,
        args.repeats,
    )
    scan_gradients, scan_gradient = compile_and_measure(
        jax.vmap(
            jax.grad(scan_objective, argnums=2),
            in_axes=(None, None, 0, None, None, None),
        ),
        scan_args,
        args.repeats,
    )
    tol = 3e-5 if dtype == jnp.float32 else 2e-12
    assert_allclose(dense_outputs, scan_outputs, rtol=tol, atol=tol)
    assert_allclose(dense_gradients, scan_gradients, rtol=tol, atol=tol)
    report = {
        "environment": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "platform": platform.platform(),
            "devices": [str(device) for device in jax.devices()],
        },
        "workload": vars(args),
        "response_bytes": sum(leaf.nbytes for leaf in response),
        "command_response_bytes": response.command_response.nbytes,
        "preparation": preparation,
        "dense_prediction": dense_prediction,
        "scan_prediction": scan_prediction,
        "dense_gradient": dense_gradient,
        "scan_gradient": scan_gradient,
        "break_even_batches": {
            name: {
                "preparation_only": break_even(
                    preparation, dense, scan, include_compilation=False
                ),
                "including_compilation": break_even(
                    preparation, dense, scan, include_compilation=True
                ),
            }
            for name, dense, scan in (
                ("prediction", dense_prediction, scan_prediction),
                ("gradient", dense_gradient, scan_gradient),
            )
        },
        "peak_device_memory_bytes": None,
        "notes": [
            "Each evaluation is a batch; trajectory counts are batches * batch_size.",
            "null break-even means no saving in warmed execution for this workload.",
            "Compilation includes tracing/lowering; execution is measured separately.",
            "Memory is response storage and XLA buffer estimates, not peak allocation.",
            "Break-even uses one warmed preparation; compilation-inclusive counts add "
            "compile-time differences.",
        ],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
