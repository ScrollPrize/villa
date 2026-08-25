# Task log: continuous deterministic crop tracing and zero-copy graph access

## Baseline

- Build: existing GCC optimized `volume-cartographer/build`.
- Dataset: Paris4 combined `fiberlets.zarr`, crop base XYZ
  `[10240,22016,6144)` to `[11264,23040,7168)`, 500 attempts.
- Result: 500 accepted, 27,715 covered anchors, 622 computed candidates,
  122 discarded candidates.
- Timing: 86.80 s wall, 1,313.17 s user, 32.27 s system; graph preparation
  12.07 s wall / 60.56 s CPU and tracing 73.37 s wall / 1,283.52 s CPU.

## Findings

- Fixed batches impose a barrier after at most one candidate per worker.
- Ordered ticket finalization can preserve strongest-first coverage while a
  bounded speculative window keeps workers active beyond a slow candidate.
- A shared payload lease costs one atomic increment/decrement per view. It is
  suitable at query granularity but not per element or persistent search state.
- The immutable graph can return unleased stable views. Direct cached route
  geometry is reconstructed from compact lattice data, so its view must own one
  derived buffer unless that geometry receives a separate memoization layer.
- Crop lookahead currently creates and clips a complete route vector for every
  considered edge even though it needs only exit/fraction information.
- Independent review required dense ticket semantics, explicit limit/error
  behavior, fixed backpressure, callbacks outside scheduler locks, outgoing
  arcs rather than ID-only views, and explicit owner lifetime tests. The
  implementation plan was updated before coding.

## Implementation

- Replaced immutable replay maps with sorted contiguous anchor, physical-edge,
  transition, and flat directed-adjacency arrays.
- Added directional views for full outgoing arcs, route points, segment
  lengths, and cost densities. Immutable views borrow stable arrays. The
  compatibility fallback owns one complete derived vector/profile through one
  shared owner, which also covers compact cache-derived results without
  retaining element-level references.
- Crop lookahead now scans route views directly and constructs clipped route
  points only for the selected committed edge.
- Replaced synchronized worker batches with dense tickets, a continuous pool,
  and ordered coordinator commits. The measured speculation window is
  `workers + max(1, workers / 8)`; a `2 * workers` trial performed too much
  invalidated work.
- Limits and exceptions are resolved at the ordered commit frontier. Work past
  the equivalent serial stop is joined but ignored.

## Performance

| Implementation | Wall | User | System | Computed | Discarded |
| --- | ---: | ---: | ---: | ---: | ---: |
| Synchronized batches | 86.80 s | 1,313.17 s | 32.27 s | 622 | 122 |
| Continuous, `2 * workers` window | 107.17 s | 1,722.32 s | 327.35 s | 860 | 360 |
| Continuous, 12.5% queue headroom | 79.66 s | 1,381.65 s | 33.44 s | 724 | 224 |

- The accepted result is unchanged: 500 lines and 27,715 covered anchors.
- Every complete/directional/anchor OBJ in `/tmp/fiber-crop-parallel-smallq`
  is byte-identical to the baseline in `/tmp/fiber-crop-debug`.
- The selected implementation reduces total wall time by 7.14 s (8.2%) and
  trace wall time from 73.37 s to 67.39 s (8.1%). CPU time increases by 5.3%
  because continuous refill intentionally computes 102 additional candidates.
- `perf` is not installed in this environment. Existing stage/task timing
  identifies candidate traversal as the sustained hotspot; graph preparation
  remains about 12 s and is unchanged for this crop.

## Deviations

- The plan-review wording proposed pinning decoded cache chunks for views.
  Current stored/cached route geometry is necessarily reconstructed from its
  compact lattice representation. The implementation instead gives the view
  one owned derived buffer, which is valid after LRU eviction and does not
  delay eviction. This preserves the requested one-owner cost and is safer for
  the cache memory budget.

## Validation

- GCC Release build: `vc_fiber_trace_chunk`, `test_fiberlet_crop_trace`,
  `test_fiberlet_storage`, and `test_fiberlet_paths` all compile with `-j 32`.
- `test_fiberlet_crop_trace`: 11 cases pass, including skewed completion,
  canonical failures, post-limit failure suppression, view ownership, and
  serial/parallel equality. Twenty consecutive GCC Release runs pass.
- `test_fiberlet_storage`: 37 cases pass; materialized forward/reverse view
  parity is covered.
- Clang Debug builds and passes all 11 crop, 37 storage, and 87 Fiberlet path
  cases. Its only build warning is an existing ignored `nodiscard` result in
  `FiberReplay.cpp:1659`.
- The full GCC Release `test_fiberlet_paths` executable has 295 bitwise
  local-metric failures at line 414 in untouched metric code; the same suite
  passes under the configured Clang Debug build.
- `git diff --check` passes after removing the task-file trailing blank line.
