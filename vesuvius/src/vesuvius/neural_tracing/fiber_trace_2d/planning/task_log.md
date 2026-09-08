# Shared Fiber3D sample-worker pool task log

- 2026-09-08: Started focused shared-pool task from clean commit `2e1714967`.
  Baseline production throughput is 229.75 samples/s and full start/drain
  throughput is 223.95 samples/s for 4096 measured + 128 warmup samples with
  87 loading and 41 augmentation processes (128 total). No GPU work authorized.
- 2026-09-08: Independent review required explicit admission invariants,
  dependency-aware request/dispatch scheduling, slot lifetime validation, and
  stronger failure/determinism tests. Plan updated before implementation.
- 2026-09-08: A parent request/dispatch prototype was rejected after real S1a
  runs reached only 201.67-213.84 samples/s. Relaying compact sample descriptors
  through one Python dispatcher and per-task assignment round trips reduced
  utilization and disrupted stage locality. A direct-data-queue variation was
  also slower (183.97-208.39 samples/s). These designs were removed.
- 2026-09-08: Final implementation uses autonomous worker-side claims, direct
  bounded data queues, lazy per-stage initialization, soft 87/41 affinity, and
  direct augmentation rather than blocking when the raw queue is full. The
  initial admission window is 16, grows by two per completed load, and remains
  capped by 512 generation-owned slots.
- 2026-09-08: Contemporaneous unchanged fixed-pool control produced 232.20
  samples/s (artifact `sample_pipeline_20260908_100640_434407`). Three final
  shared-pool trials produced 224.68, 227.44, and 224.56 samples/s (mean 225.56,
  median 224.68, approximate p95 227.16); full start/drain rates were 220.58,
  223.03, and 219.87 samples/s (mean 221.16). Artifacts end in `102016_709985`,
  `102112_977641`, and `102157_405229`. All runs were warm S1a, Release GCC 13
  VC bindings, 4096 measured + 128 warmup, 128 processes/native threads, no GPU.
- 2026-09-08: Focused validation passed 34 tests with 233 deselected. OS peak
  RSS was not captured; the fixed shared tensor allocation remains 5 GiB and
  the observed worker population was 128 plus launcher/forkserver processes.
- 2026-09-08: Exact-source confirmation after making worker control pipes
  one-way reached 227.18 production samples/s and 222.04 full start/drain
  samples/s (`sample_pipeline_20260908_102823_682491`).
