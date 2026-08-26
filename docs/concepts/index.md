# Concepts

Blackwell separates geometry, uncertainty models and inference so each part can
be transformed and tested independently.

```text
state space ── retract / local coordinates / covariance transport
     │
     ├── dynamics model ── propagate / transition Jacobian / process noise
     ├── observation model ── observe / residual / Jacobian / measurement noise
     │
     └── filter ── prediction / update / resampling
```

Read these pages in order if manifold-aware estimation is new to you:

1. [Beliefs and tangent spaces](beliefs-and-tangent-spaces.md)
2. [Models and filters](models-and-filters.md)
3. [JAX execution model](jax-execution.md)
