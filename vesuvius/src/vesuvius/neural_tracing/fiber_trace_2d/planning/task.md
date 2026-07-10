# Task: Fix Loader Threading Bottlenecks

The current CP-level loader threading overlaps work, but the profile still
shows poor warm-path wall time:

```text
batch patches total wall work tf coord desc cache source line coord_aug load
1     64     94.76 94.73 374.59 3.95 71.58 68.74 1.61 ...
2     64      2.69  2.66   6.38 2.40  1.48  0.01 0.91 ...
3     64      2.30  2.29   5.98 2.61  1.32  0.00 0.81 ...
4     64      2.55  2.53   5.57 2.20  1.30  0.01 0.82 ...
```

Required behavior:

- Remove avoidable lock contention from deterministic random-order descriptor
  lookup.
- Avoid constructing and shutting down a `ThreadPoolExecutor` every training
  batch.
- Keep deterministic sample order, skip handling, and existing batch outputs.
- Keep wall-time vs worker-time profiling so the threading factor remains
  visible.
- Add tests for the lock/persistent-executor behavior where practical.
