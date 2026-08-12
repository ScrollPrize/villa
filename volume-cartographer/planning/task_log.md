# Task log

## 2026-08-12

- Confirmed `ChunkKey` currently contains only level and chunk coordinates;
  volume identity is implicit in each independent `ChunkCache::State`.
- Confirmed each cache owns its own entry map, LRU, source metadata, listeners,
  scheduler group, and queue/network counters. `DecodedChunkCacheBudget` only
  coordinates aggregate byte eviction and does not provide shared storage.
- Confirmed `Volume::releaseCacheClient()` currently invalidates and destroys a
  volume's cache when its last current-view client leaves, preventing warm
  switches back to that volume.
- Confirmed VC3D separately constructs regular caches for overlays and Spiral
  private plane/surface-input pools. Derived SurfaceCache and geometry caches
  are separate cache types and remain outside this task.
- Confirmed decoded and persistent cache reads use heap-backed byte vectors, not
  memory mapping.
- Chosen design: stable path/URI identity is normalized and interned only during
  source registration. Render-time lookup uses a fixed-width numeric
  `VolumeSourceId` in the shared entry key; no source strings enter the hot key.
- Cancellation is intentionally deferred at the user's direction. Existing
  work may drain after a switch; this phase provides retention, deduplication,
  and newest-view prioritization but no bounded interruption latency.
- Independent plan review against `task.md`, `spec.md`, and `plan.md` found that
  the existing synthetic plane benchmark uses a fake array and would not time
  the generalized cache key/map path. The plan now adds a warmed real-cache
  scenario before taking the old-code baseline. The review also clarified that
  one service is created at application scope and shared by all windows, and
  that persistent-cache directory naming remains unchanged.
- Added a warmed real-`ChunkCache` scenario to
  `bench_chunked_plane_sampler` before changing the cache implementation. The
  2026-08-12 pre-change baseline (existing build, nine repetitions per
  scenario) was:
  - fake array: trilinear pan 0.094 s / 335.2 Mvoxel/s, trilinear zoom
    0.079 s / 398.2 Mvoxel/s, nearest pan 0.063 s / 497.6 Mvoxel/s;
  - warmed `ChunkCache`: trilinear pan 0.092 s / 343.3 Mvoxel/s, nearest pan
    0.063 s / 502.0 Mvoxel/s.
  All scenarios covered 31,457,280 pixels.
- Clarified that the required load-independent regression check is the
  established Valgrind/Callgrind CI render gate. The direct sampler timings
  above are supplemental because they exercise the warmed real-cache path;
  they are not a substitute for the virtualized gate.
- Added `VolumeSourceId` to the cache hot key and an application-owned
  `ChunkCacheService` that interns cold source identities, assigns monotonic
  numeric IDs, retains source state, and supplies one aggregate decoded-byte
  budget.
- Wired the service from `VCAppMain` through `CWindow` and both main/Spiral
  `CState` instances into `Volume`. Final viewer release retains the volume's
  lightweight source facade and decoded state, avoiding source metadata reopen
  on switch-back; explicit invalidation remains source-scoped.
- Removed the separate overlay cache registry/lease and routed overlays through
  the same source-bound cache. Disabled construction of Spiral private regular
  chunk pools while retaining separate derived surface image/geometry caches,
  and removed the obsolete Spiral plane-chunk-cache setting.
- The service stores one source-owned entry map/LRU per interned source under a
  single service registry and aggregate budget, rather than physically moving
  all entries into one cross-source unordered map. This preserves the existing
  mature source-local scheduling/statistics implementation while eliminating
  competing caches and still puts the required numeric source ID in every hot
  key. This implementation detail deviates from the plan's single-map wording;
  behavior, global capacity enforcement, source isolation, and reuse are the
  same.
- Local and remote identities now both include the selected base scale.
  Duplicate registration validates level transforms, dtype/fill, persistent
  path and budget root, fill detection, compression, and quantization after
  process defaults are applied. Facade-only worker and metadata limits do not
  make a source incompatible; this preserves the bulk-prefill acquisition path.
- Moved view-epoch allocation into `ChunkCacheService`. New views now outrank
  queued work from every older source while retaining that work, rather than
  comparing unrelated source-local epochs.
- Added source-identity invalidation directly on the service. Volume writes now
  invalidate retained decoded state even when the last viewer facade was
  already released, preventing stale data when switching back later.
- Added focused service tests for numeric source interning, cross-source key
  isolation, incompatible metadata, same-source in-flight deduplication,
  A -> B -> A warm reuse, listener lifetime, aggregate service teardown,
  global cross-source eviction, newest-view ordering, source-scoped
  invalidation, and rejected stale publication. Added a `Volume` lifecycle test
  for release/reacquire reuse.
- Final focused validation passed:
  - built `test_chunk_cache`, `test_chunk_cache_persist`, `test_volume_local`,
    `test_chunked_plane_sampler`, `bench_chunked_plane_sampler`, and `VC3D`;
  - `test_chunk_cache`, `test_chunk_cache_persist`, `test_volume_local`, and
    `test_chunked_plane_sampler` all passed;
  - `git diff --check` passed.
- Ran the established GCC 15 Release Valgrind/Callgrind CI gate in an isolated
  integration with its historical benchmark branch. All eight checked cases
  passed the `1.10x` ceiling. Ratios to the frozen reference were: serial
  `full_res=1.039x`, `fallback_3=1.037x`, `mixed_correlated=1.023x`,
  `mixed_shuffled=1.027x`; parallel `full_res=0.966x`,
  `fallback_3=1.063x`, `mixed_correlated=1.051x`, and
  `mixed_shuffled=1.012x`.
- The supplemental post-change direct run retained exact coverage at
  31,457,280 pixels. Fake-array throughput was 322.0/351.9/460.2 Mvoxel/s for
  trilinear pan/zoom and nearest pan; warmed-cache throughput was 311.6 and
  446.7 Mvoxel/s for trilinear and nearest pan. The direct run was slower than
  the earlier host-sensitive baseline across both fake and cache-backed cases;
  the load-independent Valgrind gate is the accepted regression verdict.
