# Reuse linear predictions

Use a prepared response when comparing many future input sequences against
the same linear dynamics and horizon. Preparation builds the response once;
prediction then uses matrix products for each initial state and input
sequence. This supports batched experiments and evaluation of learned linear
models. The matrices need only define a compatible linear system; they do not
need to be Jacobians of another model.

The operations live in `blackwell.models.linear` alongside the existing model
operations. They return conditional mean outputs as arrays. Process covariance
is unused, and neither a belief nor predictive covariance is returned. For
Gaussian uncertainty propagation and measurement updates, use the
[extended Kalman filter](extended-kalman-filter.md).

## Compare candidate inputs

From a development checkout, run:

```console
uv run python examples/linear_prediction.py
```

The example predicts position from a position/velocity state, compares three
input sequences, and differentiates a caller-defined scalar objective:

```python
--8<-- "examples/linear_prediction.py"
```

`prepare_prediction` accepts the existing `LinearDynamics` PyTree and a positive
static integer horizon. Its optional keyword arguments add affine offsets and
an output projection for this calculation. They do not alter the underlying
model or its filtering behaviour.

## Shapes and timing

The prediction model is discrete-time and time-invariant:

\[
x_{t+1} = A x_t + B u_t + a, \qquad y_{t+1} = C x_{t+1} + d.
\]

`A` is `model.transition`; `B` is `model.control`. `C` defaults to the identity,
and `a` and `d` default to zero. Sampling period, discretisation and input
units belong to the supplied model.

| Input | Shape |
| --- | --- |
| State transition `A` | `(n, n)` |
| Control matrix `B` | `(n, m)` |
| `output_matrix=C` | `(p, n)` |
| `state_offset=a` | `(n,)` |
| `output_offset=d` | `(p,)` |
| `initial_state` | `(n,)` |
| `controls` | `(H, m)` |
| Returned outputs | `(H, p)` |

Output row zero is `C @ (A @ initial_state + B @ controls[0] + a) + d`.
The initial output is excluded, matching `Simulator.rollout`. There is one
output for each input, including when `H=1`. Shape checks at both public
functions also run while tracing under JIT.

Use one floating dtype consistently. Float32 and float64 inputs retain their
dtype when used consistently; mixed inputs follow JAX type promotion. Enable
[64-bit mode](../concepts/jax-execution.md#precision) before creating float64
arrays.

## Inspect the responses and sensitivities

The returned `PredictionResponse` is an immutable `NamedTuple` PyTree:

| Field | Shape | Meaning |
| --- | --- | --- |
| `state_response` | `(H, p, n)` | Initial-state response `F` |
| `command_response` | `(H, p, H, m)` | Input-sequence response `G` |
| `offset` | `(H, p)` | Accumulated affine offset `b` |

Prediction contracts these operators with the initial state and inputs:

\[
Y_{t,i} = \sum_k F_{t,i,k} x_{0,k}
        + \sum_{j,r} G_{t,i,j,r} U_{j,r} + b_{t,i}.
\]

At zero-based indices, `F[t] = C A^(t+1)` and
`G[t, :, j, :] = C A^(t-j) B` for `j <= t`, with zero blocks for `j > t`.
The affine offset accumulates the state offset at every step and adds the
output offset once per output. It does not feed `d` back into the dynamics.
Preparation uses a short `lax.scan` to build the powers and affine offsets,
then assembles the causal block response.

The operators can be used directly without differentiation. They are also
the exact Jacobians: `response.command_response[t, i, j, r]` equals
`d outputs[t, i] / d controls[j, r]`, and `state_response` is the Jacobian
with respect to `initial_state`. `jax.jacfwd` and `jax.jacrev` through
`predict_sequence` give the same arrays. A scalar objective can also depend
directly on inputs, as the input penalty in the example demonstrates.

## Compile, batch and refresh explicitly

Preparation has no global cache, implicit device movement or internal JIT
wrapper. To compile preparation explicitly:

```python
prepare = jax.jit(linear.prepare_prediction, static_argnames="horizon")
response = prepare(model, horizon=32)
```

Prediction handles one initial state and one sequence. Use `jax.vmap` to batch
candidate sequences as in the example. For different initial states, use
`in_axes=(None, 0, None)`; for paired initial states and sequences, use
`in_axes=(None, 0, 0)`. Models can be stacked as PyTrees and passed through
`jax.vmap` over preparation and prediction.

Rebuild the response when `A`, `B`, `C`, either offset or `H` changes.
Changing only the initial state or candidate inputs reuses the response.
To obtain derivatives with respect to model matrices or offsets, call
`prepare_prediction` **inside** the differentiated function. An already
prepared response is fixed with respect to the model that produced it.

This utility is for open-loop Euclidean prediction. It does not apply an
affine map directly to SE(2)/SE(3) pose coordinates. An application can predict
Euclidean body-motion channels and integrate poses separately. Expand held
commands into per-step inputs before prediction.

## Choose between preparation and scanning

The dense command response stores `H² p m` values, including causal zeros.
Total response storage is `H p n + H² p m + H p` values. A long horizon or
large output/input dimensions can make this more expensive than scanning.
Use a recurrence when the model changes often or too few evaluations share
the response to amortise preparation. There is no universal CPU/GPU speed-up.

The reproducible benchmark compares a compiled `vmap` of prediction with a
compiled, batched `lax.scan`, using identical models, offsets and inputs. It
also compares gradients of the same scalar objective, including its direct
input penalty:

```console
uv run python benchmarks/linear_prediction.py --horizon 32 --batch-size 256
uv run python benchmarks/linear_prediction.py --horizon 128 --batch-size 256
uv run python benchmarks/linear_prediction.py --dtype float64
```

The JSON report records the environment and separates preparation, tracing
and compilation, first execution, and warmed median execution. All execution
measurements synchronise with the device. It reports response bytes and XLA's
argument/output/temporary buffer estimates; peak device allocation is marked
unavailable rather than inferred from these estimates.

Break-even counts are numbers of **batches**, with `batch_size` trajectories
per batch. The preparation-only count is
`ceil(preparation_time / (scan_time - prediction_time))`; the second estimate
also includes preparation compilation and the difference in prediction
compilation times. Each path's first execution is reported separately and
excluded from these estimates. A `null` count means warmed prediction or
gradient evaluation did not save time for that workload. Rerun on your target
device and shapes before choosing an implementation.

### Example CPU measurements

Measured on 3 November 2026 on an Apple M5 Pro CPU, macOS 26.7.2 arm64,
Python 3.12.13 and JAX 0.11.1. Both runs used float32, `n=32`, `m=p=3`,
256 candidates and 30 warmed repeats. Times are milliseconds; compilation
includes tracing and lowering.

| Operation | H=32 compile | H=32 warmed | H=128 compile | H=128 warmed |
| --- | ---: | ---: | ---: | ---: |
| Preparation | 32.826 | 0.081 | 34.341 | 0.276 |
| Prepared prediction | 15.787 | 0.042 | 17.191 | 0.180 |
| Batched scan prediction | 32.650 | 0.404 | 31.862 | 1.549 |
| Prepared objective gradient | 19.321 | 0.058 | 20.472 | 0.299 |
| Batched scan objective gradient | 42.569 | 0.677 | 43.244 | 2.580 |

Preparation alone amortised within one batch for both workloads and both
operations. Including compilation gave estimates of 45 and 15 batches for
prediction at `H=32` and `H=128`, and 16 and 6 batches for gradients. These
counts use unrounded measurements and exclude first-execution costs.

Total response storage was 49,536 and 640,512 bytes respectively; the command
responses accounted for 36,864 and 589,824 bytes. XLA estimated temporary
buffers of 98,688 and 394,752 bytes for prepared prediction, compared with
295,108 and 884,932 bytes for scan prediction. Response storage remains live
alongside execution buffers, so temporary storage alone is not the total
memory cost. Peak device allocation was not measured. GPU performance was
not measured; these results describe two CPU workloads.

See the [linear API reference](../reference/models/linear.md) for complete
signatures and field documentation.
