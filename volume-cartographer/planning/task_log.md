# Task log

## Planning findings

- `ChunkCacheService::openSource()` currently applies
  `ChunkCacheOptions.maxConcurrentReads` and `adaptiveConcurrentReads` to the
  entire service before registering or reacquiring a source.
- `ChunkCacheOptions` mixes service-owned decoded budget/concurrency with
  source-owned metadata and persistent-cache policy.
- `Volume::setIOThreads()` is a volume-scoped API that changes its attached
  shared service, so the apparent ownership and actual effect disagree.
- Scheduler reconfiguration currently creates/selects another scheduler,
  cancels pending old-epoch tasks, and requeues all demanded `InFlight`
  entries. Already-running requests continue, causing duplicate fetches whose
  first result is discarded by fetch-serial checks.
- The failing `test_chunk_cache` migration case explicitly expects this
  duplicate and underflows `BlockingFetcher`'s one-shot latch on the second
  call. The planned fix changes the production invariant and replaces the test;
  it will not make the latch accept duplicate same-key work.

## Implementation

- Split source-local `ChunkCacheOptions` from service-owned decoded-budget and
  fetch-concurrency options.
- Replaced `openSource()` with source-only `acquireSource()` and removed
  `Volume::setIOThreads()` plus its Python binding.
- Replaced scheduler migration with synchronized in-place admission updates.
  Increasing admission wakes existing workers; decreasing it lets running work
  drain without admitting another task early.
- Configured the normal VC3D service explicitly for adaptive operation and kept
  prefill, redownload, batch, local-volume, and standalone caches isolated.
- Replaced the duplicate-fetch migration test with exact-once increase and
  non-cancelling decrease coverage, plus fixed/adaptive transition and source
  acquisition ownership checks.

## Deviations

- The known Valgrind trace-role attribution failure remains deferred to its
  separate task as requested.

## Validation

- `cmake --build build/ci-fast-core --target vc_core --parallel 4`
- `cmake --build build/ci-fast-core --target vc_test_core --parallel 4`
- `build/ci-fast-core/bin/test_chunk_cache`: 72 cases passed in three
  consecutive runs.
- `test_volume_local`: 15 cases passed.
- `test_volume_extras`: 12 cases passed.
- `test_chunk_cache_persist`: 17 cases passed.
- `test_zarr_chunk_fetcher`: 16 cases passed.
- `cmake --build build --target VC3D --parallel 4` passed.
- The complete 131-test `vc-core` shard passed, including both synthetic render
  fixtures and the live remote-volume tests.
- `git diff --check` passed.
