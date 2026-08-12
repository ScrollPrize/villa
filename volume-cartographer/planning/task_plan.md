# Task plan

## Scope and invariants

- Unify only regular decoded volume chunks. Keep `SurfaceCache`, surface image
  tiles, surface geometry tiles, and unrelated application caches independent.
- Preserve chunk bytes, interpolation, pyramid-level selection, fill handling,
  persistent-cache paths/formats, and rendering output.
- Retain decoded chunks across current-volume changes until the global LRU
  evicts them or that specific source is explicitly invalidated.
- Do not implement cancellation tokens or volume-switch cancellation. Already
  queued/running work may drain and may populate the cache if still valid.
- Keep standalone/core users of `ChunkCache` supported; the unified service is
  injected by VC3D rather than imposed as process-global mutable state on every
  command-line tool and test.

## Baseline and discovery

1. Record the current build type, revision, host-load conditions, and repeated
   `bench_chunked_plane_sampler` results before changing the cache hot path.
   Because its current `DataChunkedArray` bypasses `ChunkCache`, first add a
   behavior-neutral scenario backed by a warmed real `ChunkCache` and a
   synthetic fetcher. Record that scenario on the old implementation before
   introducing the shared backend so the changed key/map path is measured.
2. Inventory every VC3D regular-cache construction/acquisition path:
   `Volume::sharedChunkCache`, `Volume::createChunkCache`, main plane viewers,
   overlay viewers, Spiral private plane pools, and the raw-volume inputs used
   to fill surface tiles.
3. Classify existing `ChunkCache::Options` as either service-wide (decoded RAM
   capacity, worker concurrency), source-specific (levels, dtype, fill value,
   fetchers, persistent directory, all-fill detection), or obsolete once
   storage is unified. Fail loudly when two registrations of one source provide
   incompatible source metadata instead of silently creating divergent views.

## Core cache generalization

1. Introduce a reusable shared regular-cache backend in `vc::render` and keep
   `ChunkCache` as the source-bound `IChunkedArray` facade. Move the decoded
   entry map, global LRU/accounting, metadata-entry bound, in-flight fetch
   deduplication, and shared request scheduler behind that backend.
2. Add a strongly typed `VolumeSourceId` backed by a fixed-width integer and an
   internal source-qualified key containing only:

   ```text
   VolumeSourceId + level + chunk_z + chunk_y + chunk_x
   ```

   The facade stores its numeric source ID, so `tryGetChunk`, cached/blocking
   lookup, prefetch, LRU promotion, and ready-result publication add only an
   integer copy/hash combine. Keep stable identity strings entirely out of
   render-time keys, entry objects, and per-pixel/per-chunk path handling.
3. Add a cold-path source registry that interns a canonical source identity to
   one monotonic, non-reused numeric ID for the backend lifetime. Build the
   canonical identity once from the actual data source:
   - canonical local volume/group path plus selected base/group level;
   - normalized remote volume/group URI plus selected base/group level;
   - no UI volume ID, authentication secret, or cache-directory string.
4. Extract source-identity normalization into a shared core helper and make
   cache registration consume it; remove the obsolete VC3D-only overlay
   identity implementation rather than duplicating it. Keep existing
   persistent-cache directory selection and on-disk paths unchanged so this
   in-memory generalization does not strand or rename disk-cache entries.
5. Store levels, transforms, dtype, fill value, fetchers, persistent-cache
   location, and source generation once per registered source. Re-registering
   the same identity must reuse the ID and validate compatible metadata.
6. Make explicit invalidation source-scoped: increment that source's generation
   and remove only its entries. A completed stale-generation fetch must not
   publish after invalidation, but this task does not interrupt the operation.
7. Preserve source-specific listener delivery and diagnostics. Expose global
   decoded RAM/capacity and persistent-budget totals while reporting in-flight
   downloads, throughput, and unresolved levels for the facade's source, so an
   old-volume download cannot corrupt the active volume's `qK` level display.

## VC3D ownership and lifecycle

1. Construct one regular-cache backend at VC3D application bootstrap and pass
   that same instance explicitly through every `CWindow`/`CState` to volumes
   and viewer cache acquisition. Avoid both one-service-per-window behavior and
   a hidden global singleton.
2. Change `Volume` cache acquisition to register its immutable source once and
   return source-bound facades backed by that service. Preserve a private
   backend fallback for volumes used outside VC3D.
3. Stop treating `releaseCacheClient()` as cache invalidation. Releasing the
   current view may release its facade/listeners, but the service retains source
   metadata and decoded entries for its lifetime. Keep `Volume::invalidateCache`
   as explicit source invalidation for writes or source changes.
4. Remove the separate overlay regular-cache registry and lease. Base and
   overlay access to the same source must resolve to the same numeric source ID,
   entry, in-flight fetch, and decoded bytes.
5. Replace VC3D's physically separate Spiral/private regular chunk pools with
   facades over the unified backend. Keep `SurfaceCache` and geometry-cache
   capacities separate; route their raw source-chunk reads through the unified
   backend. Remove or migrate regular-cache settings that no longer represent a
   real independent memory pool rather than leaving misleading controls.
6. Preserve newest-view request priority across sources. In this task,
   `beginViewRequest` may raise demand priority but volume switching must not
   discard queued work. Any existing `discardPending` caller must be made
   non-destructive when attached to the shared backend and documented as such;
   cooperative cancellation/request withdrawal is deferred.

## Testing

- Extend focused core cache tests for:
  - the same canonical source registered twice receiving the same numeric ID;
  - different sources with identical chunk coordinates never colliding;
  - two facades deduplicating one in-flight fetch and sharing the decoded result;
  - numeric/trivially-copyable hot keys with no source string member;
  - releasing all facades and reacquiring the source preserving resident data;
  - switching A -> B -> A reusing A without another fetch while resident;
  - source-scoped invalidation clearing A but not B and rejecting stale A
    publication without cancelling its fetch;
  - compatible local and remote persistent-cache reads/writes retaining their
    current on-disk layout;
  - global byte/LRU enforcement across sources;
  - global storage stats combined with source-specific queue/network stats.
- Add or extend VC3D-facing tests proving base/overlay reuse and that derived
  SurfaceCache/geometry caches remain distinct while their source chunks use the
  shared backend.
- Build and run at minimum:

  ```bash
  cmake --build volume-cartographer/build --target test_chunk_cache test_chunk_cache_persist test_chunked_plane_sampler bench_chunked_plane_sampler VC3D -j4
  ctest --test-dir volume-cartographer/build --output-on-failure -R '^(test_chunk_cache|test_chunk_cache_persist|test_chunked_plane_sampler)$'
  git diff --check
  ```

- Run the deterministic Valgrind/Callgrind CI render gate before and after.
  This is the required virtualized performance regression result and does not
  depend on host load. Use the established `render_valgrind_ci` target and its
  checked score/reference artifacts.

- Also run the direct synthetic sampler benchmark before and after with the
  same build and repetition count. This supplemental check includes the new
  warmed-real-cache scenario that exercises the changed cache key/map path,
  which the established CI fixture does not cover:

  ```bash
  volume-cartographer/build/bin/bench_chunked_plane_sampler
  ```

  Record every direct scenario's minimum and median/spread plus covered-pixel
  count. Treat score-gate or output/coverage changes as correctness failures.
  Investigate any repeatable direct-throughput regression outside run-to-run
  noise; do not accept a meaningful regression merely because a loose floor
  passes.

## Spec update

- Add the process-wide VC3D regular-cache ownership invariant.
- Specify canonical cold-path source interning and numeric-only hot keys.
- Specify cross-volume retention, global LRU limits, source-scoped invalidation,
  shared base/overlay entries, and source-specific queue diagnostics.
- State that derived surface/geometry caches remain separate and that
  switch-time cancellation is intentionally not part of this phase.

## Docs updates

- Add a regular chunk-cache architecture section to
  `docs/remote_file_cache.md` covering service/facade ownership, source IDs,
  heap-backed decoded storage, persistent-disk separation, global eviction, and
  volume-switch lifetime.
- Update Spiral/cache setting documentation and UI text for any removed or
  reinterpreted private regular-cache controls.
- Document that cache retention is capacity-bound, not a guarantee that every
  previously visited volume remains resident indefinitely.

## Changelog update

- Add one entry describing the unified source-qualified regular chunk cache,
  warm volume switching, and base/overlay deduplication after implementation and
  validation are complete.

## Explicitly deferred

- Cooperative cancellation of render, decode, persistent probes, local I/O, or
  network transfers.
- Withdrawing stale per-view demand or guaranteeing bounded switch latency while
  all workers are occupied.
- Combining SurfaceCache, surface geometry, or other derived caches with the
  regular decoded chunk store.
- Memory mapping decoded or persistent chunks; decoded data remains ordinary
  heap-owned byte vectors in this task.
