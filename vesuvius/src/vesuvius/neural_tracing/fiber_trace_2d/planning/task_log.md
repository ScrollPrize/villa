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
## 2026-09-08 shared-slot transport continuation

- Added bounded shared tensor slots to remove large tensor payloads from the
  multiprocessing queues. Successive 32/96-worker measurements improved from
  112.15 to 123.04 produced samples/s after small metadata was encoded as NumPy.
- Zero-copy contiguous batch views reduced result queue lag to 0.565 ms/sample
  and batch assembly to 0.044 ms/batch (`sample_pipeline_20260908_084414_712495`).
- A 58/70 split with only 128 credits was head-of-line constrained at 125.69
  samples/s (`sample_pipeline_20260908_084509_780222`). Increasing the bounded
  credit/slot window to 256 used 2.5 GiB of shared memory and reached 146.67
  produced samples/s, 165.92 ordered-delivery samples/s, and 129.46 cold-pipeline
  samples/s while consuming 109.9 aggregate CPU cores
  (`sample_pipeline_20260908_084631_245741`).
- The latest profile leaves queue transfer and stacking negligible. Mean CPU
  service is 347.34 ms for loading and 394.81 ms for value augmentation; the
  largest individual cost is native oblique anisotropic blur at 257.54 ms per
  overall sample (enabled for 48.6% of samples), followed by volume sampling at
  242.32 ms and native coordinate generation at 82.44 ms.
- One combined sandboxed benchmark invocation failed because `/ephemeral` was
  read-only in that execution context; the approved standalone benchmark
  launcher remained valid and produced the measurements above.
- Added an exact interior fast path to the native oblique blur stencil, retaining
  the original clamped boundary path and accumulation order. Ten focused native
  blur/reference tests passed. The same 58/70, 256-slot benchmark then reached
  174.16 produced, 184.90 ordered-delivery, and 148.44 cold-pipeline samples/s
  (`sample_pipeline_20260908_085247_017059`). Mean anisotropic blur time fell
  from 257.54 to 94.38 ms/overall sample. The resulting 319.29 ms load versus
  209.26 ms augmentation service times identified the static worker split as
  the next bottleneck.
- Rebalancing the unchanged 128-worker pipeline from 58/70 to 77/51 reached
  195.76 produced, 217.92 ordered-delivery, and 161.52 cold-pipeline samples/s
  (`sample_pipeline_20260908_085425_903157`). Aggregate worker use rose to
  120.5 logical cores. Contended service times were 365.08 ms loading and
  243.99 ms augmentation, leaving roughly a 13% service-cost reduction to the
  221 samples/s target.
- Reordered the oblique-stencil interior loop into tap-major contiguous row
  updates while retaining per-voxel tap order and the exact clamped boundary
  path. Ten focused native blur/reference tests still pass. On 77/51 workers,
  anisotropic blur fell to 37.39 ms/overall sample and augmentation to 164.77
  ms/sample, but the now-imbalanced loading stage (355.93 ms/sample) filled the
  256-entry reorder window; production was 178.80 samples/s despite an ordered
  delivery span of 243.81 samples/s
  (`sample_pipeline_20260908_085748_882853`).
- Rebalanced again to 87 load / 41 augmentation workers and enlarged the
  bounded shared slot/credit window to 512 (5 GiB). The 1,024-sample run reached
  the requested sustained production target at 227.45 samples/s, with 252.68
  ordered-delivery samples/s, 181.40 cold-pipeline samples/s, 124.9 aggregate
  logical cores, and reorder high-water 90
  (`sample_pipeline_20260908_085901_798884`). A longer run is required to
  separate steady-state throughput from startup/drain overhead and confirm the
  target robustly.
- The 4,096-sample confirmation with the same 87/41 split and a 512-slot
  bounded window sustained 229.75 production samples/s and 223.95 samples/s
  from process start through final drain; ordered-completion span was 234.45
  samples/s. Load service was 377.67/366.49/532.46 ms mean/median/p95 and
  augmentation was 172.37/170.38/288.04 ms. Aggregate worker CPU use was
  127.45 logical cores and reorder high-water was 88/512. Artifact:
  `sample_pipeline_20260908_090019_097829`. This meets the requested 221
  loaded-plus-augmented samples/s benchmark target without GPUs.
