# Training measurement log

2026-09-07:
- main checkout was clean.
- Reference saved LR is 0.01, current mutable sd1 config LR is 0.001.
- Reference already enabled BF16; AMP is not an untapped optimization.
- Existing benchmark uses eval/no_grad and rejects DDP.
- Existing training compute timer includes optimizer/scalar collective; target
  timer without profiling measures enqueue time rather than complete GPU work.
- GPUs 2-7 occupied; 0-1 free. Eight-GPU baseline pending availability; no
  existing jobs stopped and no competing training launched.
- Reference events lack test_total_seconds. Train/test timestamp gaps are
  approximate test intervals, not isolated device timings.
- Workflow names spec.md; actual normative file is specs.md. Relevant 3D
  training sections read. No inference behavior changes.
- Independent reviewer confirmed plan/spec compatibility. Added metric-component,
  historical-startup, provenance and throughput-versus-convergence clarifications.
- Added analyze_training.py and three synthetic regression tests; all passed.
  Ran on the actual reference event file and saved summary/config in docs/.
- Best historical loss 1.017569 at 13.97h, last 1.153580 at 15.39h;
  best within first hour 1.886512. No observed threshold crossing.
- Sampled compute median 502ms versus p95 10.03s; paired test gaps sum 3.10h.
  These identify investigation targets, not a proven bottleneck attribution.
- Deferred explicitly: live all-rank profiler instrumentation and current 8-GPU
  measurement await GPU availability. No optimization or convergence claim made.

Two-GPU continuation (same task, user-authorized):
- Using only GPUs 0,1 and /home/hendrik/.venv_las; all experiment output under
  /ephemeral/hendrik/fiber3d_train2. Existing input/cache paths reused.
- Added opt-in normal-loop per-rank wall timers and bounded CPU/CUDA traces;
  default training does not create artifacts. Evaluation/vis/snapshot costs are
  included in iteration totals but not yet individually scoped by this tracer.
- Added explicit compile modes off/default/reduce-overhead and optional model
  initialization seed. Raw models remain the checkpoint/evaluation source.
- Independent review caught cleanup on profiler export error and launcher
  batch/resume validation; fixed both. Summary excludes steps <=30, trace and
  test steps; importantly it retains loader stalls.
- Unit tests: 5 profiler/summary/compile tests pass; 13 existing DDP/snapshot/
  precision tests pass. Existing suite needs PYTHONPATH=/home/hendrik/villa/vesuvius
  to import its tests.zarr_utils helper; initial collection without that failed.
- Eager 120-step run completed successfully in 264.8s. Rank0 steady iteration
  mean 1595ms, median 483ms, p95 8554ms; compute median 449ms. Rank0 loader
  wait mean 1114ms; rank1 waits inside forward/DDP instead. Peak tensor memory
  about 13.83GiB/rank. Five-step CPU/CUDA traces exported successfully.
- Default compile first training compute took 44.69s; Dynamo warns its default
  recompilation limit of8 was reached for a reused conv frame. This is partial
  compilation with library eager fallback, not evidence of full-graph compile.
  Our code does not suppress compiler exceptions. No default compile adoption.
- All three 120-step experiments completed: eager264.8s, default295.0s,
  reduce-overhead284.6s. Rank0 steady medians483.3/531.2/536.0ms;
  mean iteration times1595.3/1560.1/1596.8ms including loader stalls.
- Both ranks exported traces in all modes. Raw eager kernel sums: BN1197.89ms,
  NCCL31.10ms, other872.77ms over five steps. Do not sum parent operator
  device totals together with kernel rows (that double-counts).
- Initial test loss identical. First train loss differs by0.000181675 under
  compilation; step100 test differences recorded in docs, not treated as a
  convergence ranking. Historical and eager model714 keys/shapes match;
  eager/compiled snapshots preserve714 raw keys without wrapper prefixes.
- Final verification:21 focused tests pass (193 unrelated deselected), including
  the13 existing DDP/snapshot/precision tests. No default compile activation.
- Complete report, commands and artifacts: docs/training_profile_2gpu.md and
  docs/training_profile_2gpu_20260907.json. Remaining worker-internal/stage
  profiling, unprofiled repeats, eight-GPU and one-hour convergence runs are
  explicitly follow-ups, not claimed complete.

Loader-worker follow-up:
- Reuse existing load_batch(profile=True) timers in normal two-GPU training;
  add worker identity/receive timestamps to separate service from queue delay.
- Same eager S1a 2x8 setup, .venv_las, logs under fiber3d_train2. No tuning yet.
- Profiling metadata/value-preservation tests passed (19 focused tests). Launched
  120 eager steps with --trace-steps 0 --profile-loader on free GPUs 0,1.
- Completed (251.4s): augmentation ~64% of worker service; 8 workers/rank
  overlap, with computed-ahead batches blocked by ordered delivery. Test losses
  match eager baseline. Details: docs/training_loader_profile_2gpu.md.
- No tuning yet; augmentation substage profiling and BN optimization remain next.
- Throughput budget: historical 22.21 samples/s global including tests and
  27.85 between tests; current loader ~9.6 global on 2 GPUs versus a measured
  ready-batch ceiling of 33.9--36.3. A controlled repeated-batch run is pending.
- Started bounded two-process-stage, per-sample queue benchmark; training path
  remains unchanged until queue behavior and throughput are measured.
- Independent review required one global delivery-credit bound, non-blocking
  completion draining and separate failure control; incorporated before coding.
- Implemented the shared two-stage process pipeline and exact split loader;
  focused pipeline/profile tests pass.
- Warm S1a benchmark: 22.67 samples/s with 16 load + 32 augmentation workers;
  queue residence was small and stage kernels remain the limiting work.
- Added separate VC3D read/decode admission controls and a stable approved
  benchmark launcher; read=1/decode=1 was throughput-neutral across 3 repeats.
- Source inspection found `sample_coords` independently dispatches large patches
  to up to 8 native render-sampler workers. Planning a per-call serial control
  and separate coordinate-generation, codec-decode, and interpolation metrics;
  the previous aggregate timing did not justify attributing extra CPU to probes.
- Added per-call coordinate-sampler limits and per-cache codec plus blocking-
  sampler stage counters. Native/Python focused tests pass; editable VC binding
  rebuilt with GCC 13 after the machine-default GCC 11 failed on existing C++23.
- Controlled 1/1/1 fresh run: 16.00 samples/s, raw CPU/wall 927.6/963.3ms;
  coord generation 10.40MP/s, interpolation 5.83MP/s, resolution 682 chunks/s.
  Uncompressed codec calls are near-no-op and not an I/O bottleneck.
- User authorized exact dependency-discovery and pinned-sampling locality
  optimization with 20 samples/s per-stage targets. Planned CPU-only profiling,
  shared footprint logic, previous-footprint reuse, and pinned direct gathering.
- Independent review required matched instrumented reference/optimized modes,
  worker-local counters, explicit coordinate/edge contracts, independent scalar
  reference coverage, and separate target reporting; incorporated before code.
- Review also identified existing reusable corner-visitor locality machinery
  and differing visitor/scalar edge and missing policies. Plan now requires a
  mechanical shared-primitives extraction before optimization, with explicit
  caller policies and separate pre-change oracles.
- Matched one-worker CPU profiles: dependency discovery improved 240.32ms to
  45.01ms (4.16 to 22.22 samples/s); pinned interpolation improved 377.04ms to
  53.26ms (2.65 to 18.78 samples/s). Combined capacity is 10.18 samples/s.
- Added previous-footprint reuse, shared compact dependency resolution,
  single-pinned-chunk gathering and a worker-local pointer cache. Exact
  reference/optimized C++ tests and focused Python plumbing tests pass.
- Added a raw-only mode to the same bounded pipeline and measured 128 persistent
  loaders with no augmentation: 93.76 samples/s end-to-end over 2,048 S1a
  samples, all workers used, 116.1 mean CPU cores. Coordinate-map generation is
  now the dominant scaling bottleneck.
