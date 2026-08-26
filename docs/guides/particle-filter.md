# Particle filter

`BootstrapParticleFilter` represents uncertainty with normalised weighted
samples. It shares Blackwell's state-space and model families with the EKF, but
does not require the posterior to stay Gaussian.

## Configure and initialise

```python
import jax
import jax.numpy as jnp

from blackwell.filters.particle import BootstrapParticleFilter
from blackwell.models import range_bearing
from blackwell.models import se2 as se2_models
from blackwell.spaces import se2

filter_ = BootstrapParticleFilter(se2, se2_models, range_bearing)

key = jax.random.key(7)
key, initialisation_key = jax.random.split(key)
belief = filter_.initialise(
    initialisation_key,
    mean=jnp.array([0.0, 0.0, 0.0]),
    covariance=jnp.diag(jnp.array([0.8, 0.8, 0.3])),
    particle_count=1_000,
)
```

Particles are sampled in the local tangent space and retracted onto the state
space. `particle_count` controls both runtime and posterior resolution; it must
remain static inside compiled code.

## Predict and weight

`step` propagates every particle with sampled process noise and updates weights
from the observation likelihood:

```python
key, prediction_key = jax.random.split(key)
belief = filter_.step(
    prediction_key,
    belief,
    dynamics,
    observation,
    control,
    measurement,
)
```

Likelihoods are accumulated and normalised in log space. The method does
**not** resample, leaving the resampling policy visible to your application.

## Resample only when needed

The effective sample size (ESS) is

$$
N_\mathrm{eff} = \frac{1}{\sum_i w_i^2}.
$$

A common policy resamples when ESS falls below half the particle count:

```python
ess = filter_.effective_sample_size(belief)
key, resampling_key = jax.random.split(key)
resampled = filter_.systematic_resample(resampling_key, belief)
belief = jax.lax.cond(
    ess < 0.5 * belief.weights.shape[0],
    lambda _: resampled,
    lambda _: belief,
    operand=None,
)
```

After systematic resampling, weights are uniform. Resampling at every step can
remove diversity unnecessarily; never resampling eventually leaves most
particles with negligible weight.

## Summarise an SE(2) belief

An arithmetic mean is unsuitable for a wrapped heading. For a compact display
estimate, use weighted translation and a circular heading mean:

```python
xy = jnp.sum(belief.weights[:, None] * belief.particles[:, :2], axis=0)
heading = jnp.arctan2(
    jnp.sum(belief.weights * jnp.sin(belief.particles[:, 2])),
    jnp.sum(belief.weights * jnp.cos(belief.particles[:, 2])),
)
pose_estimate = jnp.concatenate((xy, heading[None]))
```

This summary can hide multimodality. Inspect or cluster particles when distinct
hypotheses matter.

## Debugging degeneracy

- If weights become `NaN`, check measurement covariance is positive definite
  and no prior weight is exactly negative.
- If one particle dominates immediately, compare measurement units and increase
  realistic sensor covariance rather than adding arbitrary numerical noise.
- If the posterior impoverishes after repeated resampling, increase the
  particle count, lower the threshold or improve the proposal/model.
- Split every random key exactly once. Reusing a key repeats the same noise.

Run the [particle-localisation example](../examples/particle-localisation.md),
then consult the [particle-filter API](../reference/filters/particle.md).
