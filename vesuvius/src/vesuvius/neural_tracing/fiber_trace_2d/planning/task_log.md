# Task log: lock-free parallel Fiberlet crop tracing

## Findings

- The original crop tracer ran every seed serially. The first parallel attempt
  merely moved seed loops onto workers while leaving each graph query on the
  shared blocking `ChunkCache`.
- `incidentEdges()` synchronously prefetched every anchor's reach cube;
  `getChunkBlocking()` and `prefetchChunks()` serialized through one cache-state
  mutex/condition variable. Chunk completion used `notify_all`, producing a
  thundering herd. Combined datasets also decoded anchor, prefix, and route
  tuples for separate facet requests.
- The rejected implementation's system CPU rose from about 0.7 s at one trace
  worker to 491 s at 32 workers. Wall time regressed from 59.7 s to 71.4 s.
- `materializeChunkRouteGraph()` was the correct shared bulk-loading boundary.
  It already selected crop anchors, incident physical Fiberlets, external
  endpoints, canonical arcs, and stored transitions; it lacked route geometry
  and a reusable immutable replay adapter.

## Implementation

- Added a reusable `FiberletImmutableReplayGraphSource` keyed directly by
  `FiberletStorageKey`. The prior eager `FiberletGraph` replay overload now uses
  this shared implementation.
- Extended the existing chunk-route materializer to batch-prefetch anchor,
  prefix, and route chunks; reconstruct each retained route once; preserve
  decoded stored costs and profiles; and build immutable adjacency and joins.
- Kept canonical starting seeds separate from additional geometric-inside
  endpoints discovered while closing traversal adjacency. This was required to
  preserve established seed semantics while making routes complete.
- The stored blocking adapter no longer advertises concurrent trace queries.
  The crop CLI explicitly bulk-materializes first, then runs all seed workers
  against the lock-free graph.
- Removed `--trace-threads` and the four-worker workaround. `--threads` defaults
  to host CPUs and controls both bulk preparation and seed tracing.
- Candidate exceptions are captured by canonical batch slot, all work is
  drained, and the first canonical failure is rethrown. Coverage and acceptance
  remain serial and deterministic.

## Validation

- Release build: the normal `volume-cartographer/build` tree uses
  `-O3 -DNDEBUG`.
- Focused tests passed: `test_fiberlet_crop_trace`, `test_fiberlet_storage`,
  `test_lasagna_normal_sampler`, and `test_open_data_manifest`.
- The stored-dataset test compares materialized adjacency, arcs, profiles, and
  route points against the direct stored adapter, then queries the immutable
  graph concurrently.
- Real workload: Paris4 combined Fiberlet dataset, Lasagna normals, half-open
  base XYZ box `10240 22016 6144 11264 23040 7168`, 16 attempts, Release,
  warm local files.
- Original direct serial baseline: median 59.73 s wall across three runs.
- Corrected immutable one-thread run: graph 24.74 s, trace 17.57 s, total
  43.71 s.
- Corrected host-thread runs with 2 GiB cache:

  | run | graph s | trace s | wall s | user s | sys s | peak RSS KiB |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | 1 | 13.86 | 3.07 | 18.23 | 75.35 | 31.80 | 13,495,492 |
  | 2 | 13.89 | 3.07 | 18.26 | 66.27 | 25.27 | 13,583,336 |
  | 3 | 13.85 | 3.03 | 18.23 | 66.38 | 32.71 | 13,491,504 |

- Median wall time is 18.23 s: 3.28x faster than the original serial median.
  The immutable trace phase alone is 5.80x faster than its one-thread run.
- Every current one-thread/host-thread run reported 35,491 candidates, 16
  accepted bidirectional lines, 2,305 covered anchors, and no no-edge or
  one-sided results. Every OBJ is byte-identical with SHA-256
  `9c4606ec9f92bc4087864b6358e0735b544125f2e51ce4e1302e910b03c8b90b`.

## Limitations

- Bulk materialization trades cache contention for crop-local memory. Peak RSS
  on this 1024-base-voxel crop is about 12.9 GiB and scales with the complete
  incident route geometry. A compact immutable index/geometry representation
  can reduce that separately without changing tracing numerics.

## Follow-up correction

- Bulk materialization originally returned every in-bounds anchor as a crop
  seed but inserted only Fiberlet endpoints into the immutable graph. Reaching
  a degree-zero seed therefore raised `fiberlet replay anchor is absent`.
- Materialization now inserts all returned crop seeds, including degree-zero
  anchors. Their adjacency is empty, so crop tracing records `no_edge` and
  continues normally.
- The stored-dataset regression now includes an orphan anchor and verifies its
  materialized adjacency is empty. `test_fiberlet_storage` (36 cases) and
  `test_fiberlet_crop_trace` (6 cases) pass.
