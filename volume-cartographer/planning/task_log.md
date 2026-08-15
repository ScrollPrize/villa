# Task log

## Findings

- Fetch parallelism already updates only `ChunkRequestScheduler` admission.
  Existing queued and running work and all source states remain intact.
- `Volume::setCacheBudget()` invalidates source state, cancels scheduler groups,
  clears decoded data, and then reacquires the service-retained state with its
  old immutable capacity.
- Every source currently has a redundant `decodedByteCapacity_`; the shared
  `DecodedChunkCacheBudget` already accounts all sources and evicts the global
  oldest decoded entry.
- Persistent disk limits are manager-owned and do not invalidate RAM sources.
  Persistent write-format options remain source-construction policy and are not
  runtime RAM/scheduler settings.

## Deviations

- None.

## Implementation

- Made the aggregate decoded budget capacity atomic and mutable in place.
- Added service-level decoded-capacity configuration using the existing global
  LRU enforcement callbacks.
- Removed the copied source-local decoded-byte ceiling and its independent
  eviction condition.
- Changed `Volume::setCacheBudget()` to configure its attached service without
  invalidating or resetting the source handle.
- Added regressions for cross-source LRU reduction, preserved in-flight and
  queued requests, stable source IDs, stable Volume handles, and warm-data
  retention on capacity increase.

## Validation

- `cmake --build volume-cartographer/build --target test_chunk_cache test_volume_local -j 8`
- `volume-cartographer/build/bin/test_chunk_cache` (80 passed)
- `volume-cartographer/build/bin/test_volume_local` (16 passed)
- `cmake --build volume-cartographer/build --target test_volume_extras -j 8`
- `volume-cartographer/build/bin/test_volume_extras` (12 passed)
- `cmake --build volume-cartographer/build --target VC3D -j 8`
- `git diff --check`
