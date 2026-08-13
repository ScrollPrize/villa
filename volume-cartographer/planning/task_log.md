# Task log

## Audit findings

- `beginViewRequest()` has no production caller. Its six calls are confined to
  `test_chunk_cache.cpp` and test behavior from the retired implicit frame
  epoch/exclusive-cache model.
- Context-free `tryGetChunk()` and `prefetchChunks()` are live background APIs.
  They are used by Python bindings, CLI tools, static volume/slicing paths, and
  blocking plane/corner samplers and will remain.
- `schedulerGroup_` and `schedulerEpoch_` are live: `invalidate()` advances the
  epoch and cancels pending tasks from the old cache generation.
- `ViewerManager::ChunkCachePool`, `chunkCacheFor()`, and
  `refreshChunkSource()` only preserve the removed private decoded-cache policy.
  Current routing always returns `Volume::sharedChunkCache()`.
- `SurfaceCache::viewGeneration` is written but never read.
- `FrameChunkFootprint.hpp` has no includes or callers and only supported
  private pool sizing.

## Deviations

- None.

## Implementation

- Removed `beginViewRequest()` and its service/facade epoch allocators. Calls
  without a `ChunkRequestContext` now enter the existing background lane
  directly.
- Removed the unused scalar entry priority while retaining scheduler group
  epochs used by source invalidation.
- Replaced VC3D's no-op cache pool selector with direct
  `Volume::sharedChunkCache()` access for plane and surface-tile source reads.
- Removed the unused viewer cache refresh hook, write-only surface view
  generation, and unreferenced private-pool footprint helper.
- Deleted implicit-epoch-only tests and retained capacity, explicit active-view
  priority, per-view cancellation, and source invalidation coverage.

## Validation

- Built `VC3D`, `test_chunk_cache`, `test_chunk_cache_persist`, all chunked
  plane sampler targets, generated annotation views, and Lasagna line-view
  surfaces with the existing CMake build.
- Focused CTest selection passed 7/7 in 3.83 seconds.
- `test_chunk_cache` passed all 57 cases in its normal run. Repeated whole-suite
  stress can exceed a five-second per-run cap while constructing the unchanged
  64-worker adaptive-concurrency test; case-boundary diagnostics localized the
  delay there rather than in explicit view ownership, and all temporary
  diagnostics were removed.
- Removed-symbol search and `git diff --check` passed.
