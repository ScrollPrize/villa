# Task log

## Findings

- All cache scheduling infrastructure is already allocated by
  `ChunkCacheService::Impl`, but source states retain weak pointers to a fetch
  scheduler selected from source construction options.
- Shared source reacquisition validates source-local metadata and refreshes
  fetchers, but neither compares nor updates fetch concurrency.
- `Volume::setIOThreads()` invalidates the shared source and drops its handle;
  reacquisition therefore loses decoded data yet still uses the old scheduler.
- VC3D hides the old I/O-thread UI, but the public C++ and Python Volume APIs
  still document and expose runtime configuration.
- Existing `Volume::createChunkCache()` intentionally isolates prefill and
  redownload work. Its isolation should be represented by a separate service,
  not a scheduler attached directly to a source.
- Every explicit `ChunkCacheService` construction in the repository uses
  `shared_ptr`, so a service factory can safely return source handles while
  standalone constructors retain a service internally.

## Constraints

- Fetch configuration is global within one service and last writer wins.
- No source-local probe, fetch, or decode scheduler may remain.
- Decoded chunks and source IDs survive scheduler reconfiguration.
- Stale running work cannot publish after migration.
- Batch isolation uses a separate service.
- Rendering and sampling numerics remain unchanged.

## Implementation

- Added `ChunkCacheService::openSource()` as the shared-source factory. The
  service now validates metadata, interns identities, allocates/reuses source
  state, registers decoded-budget participation, refreshes fetchers, and then
  returns a source-bound handle.
- Removed the public service-taking `ChunkCache` constructor. Standalone
  constructors create a one-source service and open their source through the
  same factory; the handle itself only retains `{service, source state}`.
- Made source-read concurrency one service-global configuration. The latest
  valid source open or explicit configuration call wins and migrates every
  registered source under the shared scheduler-selection gate.
- Scheduler migration increments source epochs, cancels pending old-stage work,
  retains decoded and demand state, and deterministically requeues unresolved
  demanded entries. Running old work cannot publish because its fetch serial is
  stale. Transfer samples remain attached to the scheduler that admitted them.
- `Volume` now lazily creates and retains a service. `setIOThreads()` updates
  that service without invalidating decoded data. Explicit batch cache creation
  creates a separate service while retaining the shared decoded-byte budget.
- Materialized Zarr level metadata before moving fetchers into `openSource()`;
  this avoids argument-evaluation order producing an empty level list.
- Archived the paused render-replay planning records under
  `planning/stash/render_valgrind_role_attribution/`.

## Deviations

- None. The source handle remains distinct from the service because
  `IChunkedArray` is source-bound, while all scheduler and registry ownership is
  service-owned as planned.

## Validation

- `cmake --build volume-cartographer/build --parallel 4`
  - Passed, including the VC3D executable and all configured targets.
- `volume-cartographer/build/bin/test_chunk_cache`
  - 69 test cases passed.
- `volume-cartographer/build/bin/test_chunk_cache_persist`
  - 17 test cases passed.
- `volume-cartographer/build/bin/test_zarr_chunk_fetcher`
  - 16 test cases passed.
- `volume-cartographer/build/bin/test_volume_local`
  - 15 test cases passed.
- `volume-cartographer/build/bin/test_volume_extras`
  - 12 test cases passed.
- `volume-cartographer/build/bin/test_volume_pyramid`
  - 10 test cases passed.
- `ctest --test-dir volume-cartographer/build --output-on-failure -R '^test_render_synthetic_fixture$'`
  - Passed.
