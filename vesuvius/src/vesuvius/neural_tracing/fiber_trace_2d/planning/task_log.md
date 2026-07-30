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
  fields into six channel sample arrays. All 51 focused tests pass;
  representative measurement is pending approval.

## Open Acceptance Issue

- The user-requested all-float trace conversion changes the representative
  restart result from 5 to 8 in the measured intermediate implementations.
  Performance acceptance cannot hide this quality regression. After the
  combined-path benchmark, cumulative beam-score precision is the first
  targeted control to evaluate if the regression remains.
