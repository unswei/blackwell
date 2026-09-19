# Choose an estimator

Blackwell currently provides an extended Kalman filter (EKF) and a bootstrap
particle filter. Both use the same state spaces and model families.

| Question | Extended Kalman filter | Particle filter |
| --- | --- | --- |
| Belief shape | One local Gaussian | Weighted state samples |
| Best fit | Approximately unimodal uncertainty | Skewed, multimodal or strongly nonlinear uncertainty |
| Compute | Matrix operations in tangent dimension | Scales with particle count and measurement cost |
| Random key | Not needed for inference | Required for propagation and resampling |
| Differentiable/JIT | Yes, subject to model operations | Yes; resampling is discrete |
| Current support | Linear and SE(2) range-bearing | SE(2) range-bearing and compatible custom models |

SE(3) geometry and correlated transform/point uncertainty propagation are
available from 0.0.2. Both filters accept `se3` with compatible custom dynamics
and observation families. See the [SE(3) example](../examples/se3-uncertainty.md).

## Start with the EKF when

- your posterior should remain close to one mode;
- the initial estimate is reasonably good;
- you need compact beliefs or fast repeated updates; and
- linearisation around the mean is meaningful.

## Start with particles when

- the initial pose has several plausible modes;
- geometry or observations produce non-Gaussian posteriors;
- recovery from a poor initial estimate matters more than compactness; or
- you want to inspect the posterior shape directly.

!!! note "What Blackwell does not provide yet"

    There is no built-in IMU model, unscented Kalman filter, SLAM state augmentation,
    data-association layer, smoothing, or production sensor integration yet.
    Blackwell is currently strongest as a small JAX-native estimation core and
    research/teaching scaffold.

Continue with the [EKF guide](../guides/extended-kalman-filter.md) or
[particle-filter guide](../guides/particle-filter.md).
