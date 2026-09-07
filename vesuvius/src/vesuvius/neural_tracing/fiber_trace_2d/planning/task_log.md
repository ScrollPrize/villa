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
  passed. A full `test_fiber_trace_3d.py` run progressed beyond its initial
  test groups without a reported failure but then stopped producing output and
  was interrupted; it is not counted as a passing full-suite validation.
