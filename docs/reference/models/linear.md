# Linear models

Linear dynamics and observations support the filters and simulator. For
repeated conditional mean predictions, `prepare_prediction` builds an
immutable `PredictionResponse` and `predict_sequence` applies it to a state
and input sequence. See [Reuse linear predictions](../../guides/linear-prediction.md)
for batching, exact sensitivities, offsets and storage/performance trade-offs.

::: blackwell.models.linear
