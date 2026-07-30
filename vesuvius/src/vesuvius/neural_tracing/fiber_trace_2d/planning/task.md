# Native Fiber Trace Lookahead And Pipeline Optimization

Continue optimizing the precomputed native C++ fiber tracer from the committed
21.155s wall / 619.366s CPU baseline while retaining no more than 8 restarts on
the approved 87-segment whole-fiber workload.

Test the following options independently and retain only measured improvements:

1. Measure and implement exact lazy lookahead expansion using the nonnegative
   candidate-loss lower bound.
2. Fuse pinned-corner sampling, prediction/normal decoding, and candidate
   scoring to avoid large intermediate arrays and repeated coordinate metadata.
3. Only if exact pruning is insufficient, test deterministic intermediate beam
   caps as an explicitly quality-changing fallback.

Preserve deterministic candidate/tie order. Record unsuccessful experiments,
and do not retain any result above 8 restarts.
