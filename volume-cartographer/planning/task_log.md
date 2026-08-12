# Task log

## 2026-08-12

- Confirmed current render freshness uses service-wide `viewEpoch_`, while
  source `generation_` is invalidation-only.
- Confirmed an already in-flight chunk is not promoted by a newer render unless
  `prefetchChunks()` receives a strictly better scalar base priority; ordinary
  `tryGetChunk()` misses retain their original immutable executor priority.
- Confirmed persistent-cache probe/decode and remote fetch/decode are separate
  executor stages, but both currently carry the priority captured at initial
  submission. Remote source read and payload decode run in one fetcher call.
- Confirmed `SurfaceCache` owns a shared `SurfaceGeometryTileCache`: base and
  overlay fills already reuse each tile's `QuadSurface::gen()` result. Direct
  fallback rendering instead uses the viewer's full-frame
  `GeneratedSurfaceCache`.
- Chosen geometry strategy: direct fallback renders generate full-frame coords
  once before the pre-pass and reuse them for rendering; fully SurfaceCache-
  backed renders sparsely probe the shared geometry tiles so the following tile
  fills reuse exactly that generated geometry.
- Independent plan review found that `requestSurfaceViewForJob()` currently
  starts asynchronous SurfaceCache fills on the UI thread before `renderFrame()`
  can run the pre-pass. The integration plan now moves accepted view admission
  into the worker immediately after demand publication; otherwise those fills
  could queue unlocated work ahead of the metadata intended to prioritize it.

## Implementation

- Added a service-owned keyed scheduler for persistent probes and fetch/decode
  work. Pending tasks can move between background and GUI ordering in place;
  both stages use a work-conserving 7:1 GUI/background policy.
- Added explicit view ID/version request context and per-source, per-view demand
  snapshots to `ChunkCache`. One unresolved chunk keeps all interested view
  slots, and stale snapshot versions are rejected.
- Added `PointIndex::nearestPerCollection()` so a folded surface can retain
  multiple viewport occurrences while scheduler priority uses the nearest one.
- Added sparse viewport dependency collection to `ChunkedPlaneSampler`, with
  exact nearest/trilinear chunk dependency enumeration and per-chunk 8-pixel
  occurrence deduplication.
- VC3D now uses one deterministic random sample per 8x8 viewport cell on every
  accepted render. Plane coordinates are analytic. Direct surface rendering
  generates and reuses its full coordinate/normal matrices. A fully cached
  surface probes `SurfaceGeometryTileCache`, whose tiles are reused by the
  following `SurfaceCache` fills.
- Moved `SurfaceCache::requestView()` into the accepted render worker after
  demand publication and propagated the GUI context into fill prefetches.
- Pointer moves update pending priorities without requesting an additional
  render. Viewer shutdown and source replacement remove only that view's
  demand. Executing work remains non-cancellable by design.

## Validation

- Built `test_chunk_cache`, `test_chunk_cache_persist`,
  `test_chunked_plane_sampler`, `test_volume_local`, and `VC3D`.
- Passed all focused CTest targets: 5/5, including `test_point_index`.
- Direct test runs passed 42 `test_chunk_cache` cases and 12
  `test_chunked_plane_sampler` cases before the final CTest run.
- `git diff --check` passed.

## Synthetic render gate

- Merged `origin/main` at `5b2ca4ff3`, which contains the synthetic rendering
  harness, and configured its documented GCC 15 Release build with native
  architecture tuning disabled.
- Native fixture and no-site harness checks passed: 2/2.
- Full Valgrind/DRD replay gate passed all eight cases against the frozen main
  reference (maximum allowed ratio `1.10x`):

  | Fixture | Scenario | Modeled ns | Ratio |
  | --- | --- | ---: | ---: |
  | serial | full_res | 332507.608 | 1.015x |
  | serial | fallback_3 | 892944.275 | 1.032x |
  | serial | mixed_correlated | 620216.920 | 1.020x |
  | serial | mixed_shuffled | 926286.997 | 1.022x |
  | parallel | full_res | 870894.808 | 0.964x |
  | parallel | fallback_3 | 2644150.191 | 1.036x |
  | parallel | mixed_correlated | 1877335.942 | 1.049x |
  | parallel | mixed_shuffled | 7298099.253 | 1.057x |

- The gate exercises the production `ChunkedPlaneSampler` fine-to-coarse path.
  VC3D logs `prepass_ms`, retained occurrence count, and unique chunk count for
  interactive profiling; the headless gate does not construct a GUI viewer and
  therefore does not assign a standalone wall-time claim to the viewer pre-pass.
