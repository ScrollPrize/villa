# EMA99 tracer reference experiments

Commit current code. Freeze current EMA99 and remeasure off and previous best weighted history before experiments. Try EMA, gated EMA, EMA plus fixed seed, and best-match bounded history. Preserve training and manager admission. Record restarts on same difficult held-out fiber.

## Next experiment requested after this chain

Replace fixed seed with a bounded per-beam fiber-appearance model: multiple
representative anchor embeddings that cover many observed points while retaining
distinct appearances. Investigate coverage-weighted medoids/diversity rather
than only outlier selection. Design selection/update/scoring and drift protection
after completing the current sweep; do not mix into its controlled comparison.
