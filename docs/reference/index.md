# API reference

This reference is generated from Blackwell's public source docstrings. Start
with a [task-led guide](../guides/extended-kalman-filter.md) if you are learning
the library; use these pages for exact signatures, fields, shapes and return
values.

| Area | Public modules |
| --- | --- |
| Beliefs | [`blackwell.beliefs`](beliefs.md) |
| State spaces | [`euclidean`](spaces/euclidean.md), [`se2`](spaces/se2.md) |
| Models | [`linear`](models/linear.md), [`se2`](models/se2.md), [`range_bearing`](models/range-bearing.md) |
| Filters | [`ekf`](filters/ekf.md), [`particle`](filters/particle.md) |
| Experiment support | [`simulation`](simulation.md), [`metrics`](metrics.md) |

## Import conventions

Principal belief containers are available at package level:

```python
from blackwell import GaussianBelief, ParticleBelief
```

Algorithms and operation families use explicit imports:

```python
from blackwell.filters.ekf import ExtendedKalmanFilter
from blackwell.models import range_bearing
from blackwell.spaces import se2
```

Only modules documented in this reference are supported public API. Names
under `blackwell._experiments` and `blackwell.filters._protocols` are private.
