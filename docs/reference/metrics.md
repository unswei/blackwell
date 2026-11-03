# Metrics

Error and consistency metrics describe estimation quality. For evaluating a
model's actual selections, `selection_regret` compares against a feasible finite
bank, and `binary_decision_counts` retains observed labels, denominators and
undefined-rate flags. See [Evaluate models through their decisions](../guides/decision-relevant-evaluation.md)
for missing-outcome handling, a planar example and paired execution reporting.

::: blackwell.metrics
