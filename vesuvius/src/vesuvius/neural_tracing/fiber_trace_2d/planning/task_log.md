# Native Fiber Trace VC3D Corner-Batch Sampling Task Log

## Baseline

- Representative warm-cache result before this task:
  `trace_wall_s=85.958`, `trace_cpu_s=2453.293`, `restarts=5`, `segments=87`.
- Profile: prediction sampling 29.314s, normal sampling 28.232s, candidate
  scoring 7.170s, frontier construction 7.061s, and pruning 6.509s.

## Findings

- VC3D already provides blocking requested-level coordinate sampling that
  deduplicates dependencies, uses the process-wide threaded chunk reader, pins
  resolved chunks, and samples without repeated global-cache access.
- The representative persisted products are six separate scalar uint8 3D Zarr
  volumes with 64-cubed chunks, so one cache per physical volume maps directly
  to the current manifests.
- The existing public coordinate sampler returns interpolated uint8 values.
  This task instead requires its nearest-neighbor mode over eight explicit
  integer corners so compact `nx/ny` values can be decoded and interpolated as
  one orientation tensor.

## Deviations

- The workflow requests an independent agent plan review. No subagent was
  spawned because higher-level policy requires explicit user authorization for
  delegation. A direct consistency review will be performed instead.

## Validation

- Focused native validation after mixed-grid batching:
  - `test_chunked_plane_sampler_fallback`: 15 passed;
  - `test_lasagna_normal_sampler`: 11 passed;
  - `test_fiber_trace3d`: 25 passed.
- `git diff --check` passes.

## Performance Iterations

- Generic VC coordinate sampling invoked independently for all six channels
  regressed substantially and was stopped at a projected runtime near 3m.
- Grouped three-channel corner sampling with serial pinned traversal took
  158.031s wall / 1821.036s CPU and retained 5 restarts.
- Parallel grouped traversal reduced that to 99.099s wall / 1891.315s CPU and
  retained 5 restarts.
- Float native trace math plus the initial grouped direct path took 103.200s
  wall / 2003.060s CPU but regressed to 8 restarts.
- Direct dependency-index corner lookup took 90.882s wall / 1881.751s CPU with
  8 restarts.
- Retaining the current chunk by coordinate bounds and reading same-chunk
  corners through pinned raw byte pointers took 84.039s wall / 1909.650s CPU
  with 8 restarts. Profiled prediction and normal sampling were 25.933s and
  25.996s respectively.
- Prediction arrays use 64-cubed chunks while Lasagna normal arrays use
  32-cubed chunks. The combined six-volume path now constructs voxel cubes and
  fractions once, reuses one dependency layout per distinct chunk shape, and
  keeps a separate VC3D cache per physical volume. With one unrelated CPU core
  and the GPU occupied, this path took 76.724s wall / 1527.689s CPU with 8
  restarts. The combined prediction/normal batch accounted for 47.281s;
  candidate scoring was 9.945s, frontier construction 7.139s, and pruning
  6.471s. The run processed 105,810,462 candidates across 4,170 generations.
- The combined-path benchmark exposed an eight-worker cap inherited from the
  interactive render pool. The corner batch now has a separate bounded pool,
  uses up to hardware-concurrency minus two workers, and traverses all physical
  volumes sharing a chunk layout together so each point's dependency metadata
  is loaded once per layout. This reduced the same contended run to 70.867s
  wall / 1446.621s CPU with 8 restarts; the combined stage fell from 47.281s
  to 42.881s.
- The next revision retains per-candidate option storage across alternating
  small/large lookahead generations instead of destroying and recreating the
  inner vectors, decodes compact uint8 `nx/ny` corners through a float lookup
  table, and reports corner-read, prediction-decode, and normal-decode times
  separately. It measured 66.384s wall / 1388.770s CPU with 8 restarts. The
  combined 38.900s split into 22.013s corner batching, 8.530s prediction
  decode, and 8.355s normal decode.
- The next revision reduces per-point corner dependency metadata from eight
  64-bit dependency indices plus offsets to one common dependency, eight
  32-bit offsets, and a separate boundary-only dependency table. The float
  compact-normal lookup now stores tensor components, and the combined path
  consumes shared raw corner/fraction/valid arrays without duplicating those
  fields into six channel sample arrays. It measured 57.789s wall / 1364.742s
  CPU with 8 restarts. Corner batching fell from 22.013s to 12.684s; prediction
  and normal decode remained 9.135s and 7.794s.
- The next revision retains the raw corner output buffers across lookahead
  generations, computes interpolation weights once while jointly decoding
  prediction and normal outputs in one parallel pass, and materializes the
  final lookahead frontier in deterministic task slots in parallel rather than
  compacting it serially. It measured 44.070s wall / 1067.310s CPU with 8
  restarts. Combined corner/decode time fell to 19.812s and frontier time fell
  from 8.130s to 3.277s.
- The next revision retains and parallel-fills candidate task buffers, uses
  static scheduling for uniform in-memory candidate scoring, and replaces up
  to eight full frontier pruning scans with one deterministic min-heap plus
  spatial acceptance checks. The heap key preserves loss, depth, and original
  generation order. The combined experiment regressed to 74.329s wall /
  1826.523s CPU with 8 restarts under the one-core competing workload. Static
  scoring rose from 6.134s to 20.374s and parallel task construction rose from
  4.613s to 8.261s, while heap pruning improved from 7.482s to 1.777s.
- Static scoring and parallel task construction were removed. The next control
  retains only reusable task storage and exact heap pruning. It measured
  37.743s wall / 944.706s CPU with 8 restarts. Pruning remained at 1.585s,
  task construction fell to 2.489s, and dynamic scoring returned to 6.438s.
- The next revision specializes the common within-one-chunk corner case: each
  point stores one dependency, one base byte offset, and clamped-axis bits;
  eight dependency/offset pairs are stored only for actual chunk-boundary
  points. Corner offsets are reconstructed once per layout point and reused
  across its three physical channel volumes. Corner batching improved from
  13.422s to 8.721s, but contended decode/scoring/frontier variation produced
  a 38.117s wall / 1075.379s CPU run with 8 restarts.
- The next control changes only the hot uniform decode, score, and frontier
  OpenMP loops to `dynamic,64`. This avoids static ownership of the externally
  occupied core while reducing scheduler traffic relative to `dynamic,8`.
  Focused fiber tests pass; representative measurement is pending approval.

## Open Acceptance Issue

- The user-requested all-float trace conversion changes the representative
  restart result from 5 to 8 in the measured intermediate implementations.
  Performance acceptance cannot hide this quality regression. After the
  combined-path benchmark, cumulative beam-score precision is the first
  targeted control to evaluate if the regression remains.
