# Linear Kalman filter

The generic EKF becomes the ordinary discrete Kalman filter when paired with
Euclidean geometry and linear models. This example tracks one-dimensional
position and velocity from noisy position measurements.

```console
uv run python examples/linear_kalman_filter.py
```

Expected output is similar to:

```text
Final [position, velocity]: [3.7256737  0.91042304]
Final covariance:
 [[0.16104065 0.06601974]
  [0.06601974 0.10097054]]
```

The transition, control and observation matrices are ordinary JAX arrays. The
configured filter is carried through `jax.lax.scan`, so the entire time series
is one compiled computation.

??? example "Complete script"

    ```python
    --8<-- "examples/linear_kalman_filter.py"
    ```

Use this example as the shortest starting point for a custom Euclidean model.
Then read the [EKF guide](../guides/extended-kalman-filter.md) and
[linear-model API](../reference/models/linear.md).
