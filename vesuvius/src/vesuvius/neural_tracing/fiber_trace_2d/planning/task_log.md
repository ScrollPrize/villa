# Task log: native anisotropic Fiber3D augmentation

- 2026-09-07: Full CPU pipeline benchmark (16 loaders, 32 augmentation
  workers, 256 measured after 32 warmup) reached 24.02 samples/s production and
  22.49 samples/s end to end. Loading averaged 230ms/sample; value augmentation
  averaged 1.103s/sample and imposed 1.705s mean inter-stage queue wait. The
  artifact is `sample_pipeline_20260907_143832_106216/result.json`. Augmentation,
  most likely the 50%-probability oriented blur branch, is now the bottleneck;
  substage profiling remains required before optimization.
- 2026-09-07: Planned a measured native replacement which retains three
  sequential oriented passes but precomputes constant trilinear stencils and
  never creates full-volume coordinate grids. The native call is single-threaded
  and process-parallel; all work remains CPU-only.
- 2026-09-07: Independent review identified position-dependent rounding in the
  Torch normalized-coordinate round trip and required full-size/high-gradient
  comparisons, exact basis/border semantics, narrow fallback, input-contract
  validation, and service-time-based per-core reporting. The plan incorporates
  these findings.
- 2026-09-07: Timing-only Torch baseline confirmed anisotropic blur was used by
  48.4% of samples and cost about 1.82s per enabled sample. Initial native
  full-128-cube measurements took 0.34s (4.7-5.2x faster). A random normalized
  fixture differed by max 6.18e-5 and mean 1.64e-6; the acceptance threshold was
  explicitly revised to scale-relative bounds before default enablement.
- 2026-09-07: Native matched-topology run improved production throughput from
  22.65 to 58.81 samples/s and full augmentation from 1.165s to 0.494s mean.
  Enabled oriented blur fell to about 339ms (5.4x); full augmentation capacity
  is 2.02 samples/s per worker. Eleven focused native/reference and backend
  tests pass.
- 2026-09-07: A 64-loader/128-augmentation-worker run reached 76.80 samples/s;
  all workers participated and augmentation retained 1.44 samples/s per worker,
  but coordinator batch stacking (66.4ms/batch) and 2.10s result-queue wait
  limited utilization to 78.5 cores. Coordinator delivery is a separate next
  bottleneck, not part of this augmentation replacement.
- 2026-09-07: Release binding build and focused augmentation/pipeline test
  selections passed (11 tests in the final selection; an earlier 12-test
  selection also passed).
- 2026-09-07: Committed that native augmentation increment as `19af65a07`.
  The exact 128-worker 32-load/96-augment baseline produced 78.66 samples/s,
  81.54 ordered samples/s and used 76.5 aggregate worker cores
  (`sample_pipeline_20260907_152553_942459`).
- 2026-09-07: Reusing the native directional kernel for isotropic blur reached
  89.61 produced / 85.75 ordered samples/s and reduced that operation from
  173.1 to 86.7ms/sample (`sample_pipeline_20260907_152959_126973`).
- 2026-09-07: Fusing valid-only normalization, contrast, brightness and gamma
  reached 123.52 produced / 131.01 ordered samples/s; augmentation fell to
  322.6ms/sample (`sample_pipeline_20260907_153756_733494`).
- 2026-09-07: A 56-load/72-augment rebalance regressed to 109.31 produced /
  108.40 ordered samples/s with reorder high-water 188
  (`sample_pipeline_20260907_153849_887277`).
- 2026-09-07: A benchmark-only fused 128-worker probe reached 106.45 produced /
  107.94 ordered samples/s and showed 1.02s/sample output receive lag
  (`sample_pipeline_20260907_154220_816263`), confirming tensor transport as a
  limiter. A following run aborted because `_LoadFactory` accidentally lost its
  return during this probe; it was restored before further measurement.
- 2026-09-07: Axis-specialized blur reduced isotropic blur to 50.2ms/sample.
  The matched run reached 109.52 produced / 99.73 ordered samples/s amid high
  transport/host variability (`sample_pipeline_20260907_154721_325067`).
- 2026-09-07: Seventeen focused native/reference/backend tests pass after the
  fused transform and axis-aligned blur work. A full `test_fiber_trace_3d.py`
  run from the previous increment progressed beyond its initial
  test groups without a reported failure but then stopped producing output and
  was interrupted; it is not counted as a passing full-suite validation.
