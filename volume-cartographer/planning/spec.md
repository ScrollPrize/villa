# VC3D render and fetch specification

## Invariants

- Rendering values, interpolation, pyramid transforms, and cache contents must
  not change as a side effect of diagnostics or scheduling work.
- Remote fetching must remain asynchronous; UI and render threads must not wait
  on network or persistent-cache I/O.
- Queue diagnostics must come from the existing chunk-cache request state, not
  from a parallel accounting system at viewer call sites.
- A shared cache must report each unresolved chunk once regardless of how many
  viewers requested it.
- Normal download diagnostics belong in the existing application cache status
  bar, not in per-slice overlays or a persistent setting. The explicit
  `--debug-download-queue` process-start diagnostic may additionally render
  active remote chunks over slice images for request-order investigation.

## Regular chunk-cache ownership

- VC3D constructs one `ChunkCacheService` at application startup and passes it
  through every window/workspace state. Main, Spiral, overlay, and derived
  surface-tile input reads must acquire source facades from this service rather
  than construct independent decoded regular-chunk pools.
- A source is identified once on cache registration by its canonical local
  path or normalized remote URI plus selected base level. Authentication and
  persistent-cache-directory strings are not source identity.
- Registration interns the source identity to a monotonic, non-reused
  `VolumeSourceId`. Render-time `ChunkKey` construction, equality, and hashing
  use this fixed-width numeric ID and must not process or retain source strings.
- Re-registering a source reuses its source state and validates immutable level,
  transform, dtype, fill, and persistent-path metadata. Incompatible duplicate
  registrations fail loudly.
- Decoded data is heap-backed and globally constrained by the service's shared
  decoded-byte budget. Source state and resident entries survive A -> B -> A
  volume switches until global eviction or explicit source invalidation.
- Releasing a viewer/cache client does not invalidate service-owned source
  state. Writes and explicit `Volume::invalidateCache()` invalidate only that
  source; stale generation results must not publish afterward.
- Surface image tiles and surface geometry tiles remain derived caches with
  independent budgets. Their raw volume reads use the regular cache service.
- Pending interactive work is owned by per-chunk `(view ID, view version)`
  demand slots. Atomically replacing a view snapshot or closing a view removes
  its stale slots; a pending probe, source read, or decode with neither another
  view owner nor an explicit background owner is canceled by task ID.
- Running work is not interrupted. A running probe or source read may finish,
  but it cannot enqueue the next stage after all of its demand has become stale.
  A chunk independently requested by CLI/batch/background work remains in the
  scheduler's separate background lane after GUI demand is removed.
- Viewers allocate stable numeric view IDs and monotonically increasing request
  versions. The cache service stores only explicit per-view snapshots and the
  active view ID; it does not maintain an implicit frame epoch. Scheduler group
  epochs are internal to source invalidation and do not affect interactive
  priority.

## Interactive chunk scheduling

- Regular chunk work has separate interactive and background pending lanes.
  Existing context-free `IChunkedArray` calls are background work. VC3D render
  requests carry a stable numeric view ID and a monotonically increasing view
  version through direct misses, prefetches, and asynchronous `SurfaceCache`
  fills.
- Before an accepted interactive render samples the volume, it probes the
  viewport on a deterministic stratified 8-pixel grid. Each probe records the
  2-D viewport occurrence of every required requested-level and permitted
  fallback chunk. It queues at most five coarser levels and may stop earlier
  when one average chunk edge spans the larger viewport extent, with both
  quantities measured in level-0/base-volume voxels. Plane and renderable
  `QuadSurface` parameter coordinates use those same units; camera scale is
  framebuffer pixels per base voxel for both. This fallback demand remains
  present on refinement renders until it resolves.
  Nearby occurrences of the same chunk are deduplicated, but distant
  occurrences on folded surfaces are retained.
- The source volume/Zarr pyramid level is VC3D's only render LOD. One constant
  source level is selected analytically for each source over a complete render
  from camera scale and declared level transforms. Base and overlay sources may
  select different numeric levels. Generated volume coordinates, finite
  differences, cache residency, and per-pixel geometry never select LOD.
- `QuadSurface::scale()` is grid samples per level-0/base-volume voxel in each
  parameter direction. It declares the point-grid parameterization and is not
  another LOD. A transient renderable producer must provide a finite positive
  scale; serialized surfaces use their stored declaration.
- Line-annotation ribbons are derived views of the authoritative stored line.
  They uniformly arclength-resample with a default maximum interval of 50 base
  voxels, retain both endpoints, and carry an explicit bidirectional mapping
  between original fractional point positions and ribbon grid columns. Input
  line spacing may be arbitrary; control points, cuts, and persistence remain
  in original line-position coordinates.
- A completed pre-pass atomically replaces that source's previous snapshot for
  the view. Snapshot construction and surface-coordinate generation occur
  without the chunk-cache state lock. Older view versions cannot replace a
  newer snapshot.
- Pending interactive work is ordered by active view, coarser pyramid level,
  nearest retained occurrence to that view's focus, then FIFO. A GUI miss not
  observed by the sparse pre-pass has no location and sorts after located work
  at the same view and relative level. It cannot outrank a located coarser
  fallback because relative level is the primary ordering key. Dependency
  publication is coarse-to-fine so
  workers cannot admit fine work before its coarse entries are visible.
- Mouse interaction marks that view active and updates distances against its
  retained point index. Before any pointer has been observed, viewport center
  is the focus.
- One unresolved source/chunk entry may contain demand from several views.
  New snapshots promote already queued work in place instead of submitting a
  duplicate task. Clearing a view removes only that view's demand.
- Regular chunk work passes through three shared pending queues. A 32-worker
  local probe queue classifies persistent data, empty markers, and misses using
  filesystem metadata only. Cache hits enter an eight-worker CPU read/decode
  queue; misses enter the remote source-read queue. Successful source reads
  then enter the decode queue.
- Normal interactive remote source reads use 64 available workers with a
  dynamic admission limit initially set to two. After the latest
  `current_limit * 4` successful encoded chunk transfers are available, the
  controller measures aggregate bandwidth as their total encoded bytes divided
  by the interval from their earliest start to latest completion. It computes
  average encoded chunk size from the same samples and selects
  `ceil(bandwidth * 0.25 seconds / average_chunk_bytes)`, clamped to `[2,64]`.
  Failed and missing reads are excluded. A new larger limit is held until its
  larger sample window is complete. The controller intentionally performs no
  exploratory increases, so a low current limit may underestimate latent
  bandwidth.
- Adaptive admission is service-wide for normal remote rendering and changes
  only how many source tasks may start; it does not alter pending-task priority.
  A decrease does not cancel running work. Explicit `maxConcurrentReads`
  callers, tests, prefill operations, and local volumes retain fixed
  concurrency.
- Each queue uses the work-conserving 7:1 interactive/background admission
  policy. Current view-relative priority is recalculated at every stage handoff,
  and atomic view-demand publication reprioritizes pending work in all three
  queues. Classification never waits for cached payload reads or decoding, so
  known remote misses can be admitted while cached decodes are busy. There is
  no cross-stage pyramid-level barrier in this phase.
- Running probe, download, decode, and render work is not cancelled. Updated
  priorities affect pending work and stage handoffs only.
- A direct surface render generates its full coordinate/normal matrices before
  the pre-pass and reuses them for normal sampling. A fully SurfaceCache-backed
  render probes the shared `SurfaceGeometryTileCache`; subsequent tile fills
  reuse those geometry tiles and do not allocate a second full-frame matrix.

## Download label

The main window uses one permanent status label for cache diagnostics and
Z-scroll sensitivity; these fields must not be rendered by overlapping status
widgets. RAM and persistent-disk values share one trailing `GiB` unit:

`RAM X/Y disk X/Y GiB`

During active remote downloads, the existing cache status bar appends:

`qK X/Y/Z`

- `K` is the first pyramid scaledown level with unresolved chunk requests.
- `X/Y/Z` are unresolved request counts for consecutive levels from the first
  through the last nonzero level. Interior zero counts are retained; leading
  and trailing zero levels are omitted.
- The queue item is shown only for remote volumes while remote fetches are in
  flight. Remote volumes otherwise show `net idle`; local volumes have no
  network field.
- The displayed MiB/s is the adaptive controller's aggregate successful-chunk
  estimate over its current `parallelism * 4` sample window, not a fixed-time
  status polling window. The `Nx` value remains the actual current number of
  source fetches in flight.
- The full active status is
  `RAM X/Y disk X/Y GiB net Nx XMiB/s qK X/Y/Z Z sens: N.N`.

## Active-download debug overlay

- `--debug-download-queue` is disabled by default and applies uniformly to all
  `CChunkedVolumeViewer` instances, including plane, segment, strip, and
  generated annotation slice views.
- The overlay reflects actual remote source fetches, not unresolved queue
  entries, persistent-cache probes, local decode work, or resident chunks.
- Every accepted debug render maps its full-resolution logical level-0
  coordinates to source-qualified containing chunks for the requested and
  queued fallback levels. Per-pixel storage uses level-local `uint16` IDs with
  compact key tables; zero is invalid and IDs must never alias on overflow.
- The clean rendered framebuffer remains authoritative. Active matching chunks
  are composited over a copy at 50 percent opacity with deterministic colors by
  pyramid level, and removing the last matching active fetch restores the clean
  framebuffer.
- Worker callbacks only publish activity state. Framebuffer composition and Qt
  repaint requests happen on the UI thread, and diagnostics never queue chunks
  or alter request priority.
