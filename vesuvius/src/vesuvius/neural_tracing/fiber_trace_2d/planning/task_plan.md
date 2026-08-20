# Plan: corridor-filter on-demand fiberlet preprocessing

## 1. Corridor selection and cache identity

1. Reuse the existing exact fiber replay tube cell selector. It enumerates only
   the bounded coarse-cell neighborhoods of reference segments and performs an
   exact segment-to-cell-AABB test; do not add a brute-force cell/segment scan.
2. Compute the selected cells once for the active reference interval and radius,
   preserve canonical Z/Y/X order, and bind the canonical clipped reference
   geometry, exact radius, selector version, and cell selection into both
   generated dataset fingerprints.
3. Pass the immutable selection and the existing R-tree-backed replay-tube
   containment query into the on-demand preprocessor. Validate that
   cells are ordered, unique, in the prediction grid, and consistent with both
   cache grids.

## 2. Filtered chunk generation

4. Partition the selected cells by anchor owner chunk once during preprocessor
   initialization. Anchor generation must look up only its selected cells; it
   must not enumerate all 4,096 cells and then perform geometry checks. The
   existing extraction pipeline must still construct neighboring NMS context as
   suppressors while publishing only selected cells owned by the requested
   chunk, including across owner-chunk boundaries.
5. Apply the shared exact-tube query to fitted-anchor retention and every
   fiberlet DP point, matching eager tube extraction. Keep empty valid chunks
   representable. Fiberlet generation continues to consume cached anchor
   dependency chunks and therefore excludes unselected endpoint cells.
6. Retain stable cell, anchor, candidate, and serialized ordering so the cached
   result agrees with eager extraction over the same selected cell set.

## 3. Ahead-of-traversal scheduling

7. Derive both owner partitions and the complete ordered reference chunk
   schedule from the single immutable selected-cell population. Submit that
   schedule through the existing `prefetchScheduled()` entry point. After the
   app initializes progress bookkeeping and before graph evaluation starts,
   call `prefetchScheduled(schedule, 0, schedule.size(), false)`. Do not
   wait for the complete schedule: cache workers should prepare upcoming anchor
   dependencies and fiberlet chunks while traversal consumes earlier chunks.
8. Keep ordinary blocking graph lookups as the correctness fallback and promote
   an already queued background request to foreground priority when demanded.
   Scheduling must not create a second queue or cache outside `ChunkCache` and
   its LRU.
9. Report global schedule counts and completed chunk counts through the existing
   replay progress records so foreground graph progress and background
   preprocessing can be distinguished.

## 4. Validation and measurement

10. Add focused tests showing a touched chunk generates only selected cells,
    displaced out-of-corridor anchors and paths are absent, geometry/radius
    changes reject stale persisted datasets, schedule/partition coverage is
    exact, scheduled prefetch materializes data without a foreground request,
    and a later blocking request promotes queued work with one worker. Cover
    radius-zero selection, shared cell boundaries, partial volume cells, clipped
    endpoints, repeated reference points, non-unit prediction scale,
    cross-chunk NMS, and a retained-endpoint path whose interior leaves the tube.
11. Compare eager and cached replay fixtures for deterministic geometry, costs,
    failures, and ordering. Run the affected cache, storage, path, and replay
    tests and build `vc_fiberlets` with `-j32`.
12. Run same-checkout eager and before/after on-demand canonical
    5,000-base-voxel Paris4 replays with identical build, arguments, threads,
    and cold generated-cache/OS-cache policy. Profile hotspots, repeat at least
    three times, and report mean, median, min/max or p95 wall/CPU time, selected
    versus avoided cells, generated chunks, effective core use, and any
    remaining gap.

## Spec update

Change the on-demand preprocessing contract to make generated anchor/fiberlet
datasets corridor-specific processing caches. Specify bounded exact cell
selection, geometry/radius-dependent dataset identity, shared indexed tube
containment for fitted anchors and DP points, per-owner-chunk selected-cell
lookup, foreground priority promotion, and nonblocking submission of the
ordered reference schedule through the existing chunk-cache scheduler.

## Documentation update

Update `volume-cartographer/docs/fiberlets.md` and the fiberlet storage
documentation with corridor-specific cache identity, filtered anchor ownership,
and the relationship between scheduled background preprocessing and blocking
cache lookup.

## Changelog update

Record removal of full-chunk anchor overprocessing and activation of the
previously unused replay schedule, including the measured 5,000-voxel result.
