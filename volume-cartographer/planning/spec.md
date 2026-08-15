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
  surface-tile input reads must acquire source handles from this service rather
  than construct independent decoded regular-chunk pools.
- `ChunkCacheService::acquireSource()` is the only shared-service source factory.
  It interns or reacquires the source and returns a source-bound `ChunkCache`
  implementing `IChunkedArray`. A standalone `ChunkCache` convenience
  constructor creates and retains a one-source service; it never owns local
  schedulers.
- A source is identified once on cache registration by its canonical local
  path or normalized remote URI plus selected base level. Authentication and
  persistent-cache-directory strings are not source identity.
- Registration interns the source identity to a monotonic, non-reused
  `VolumeSourceId`. Render-time `ChunkKey` construction, equality, and hashing
  use this fixed-width numeric ID and must not process or retain source strings.
- Re-registering a source reuses its source state and validates immutable level,
  transform, dtype, fill, persistent encoding, and persistent-path metadata.
  Compatible registration atomically adopts the newly opened fetchers so
  refreshed temporary credentials take effect without changing source ID or
  evicting decoded chunks. Fetch/decode work retains its captured fetcher and
  may publish only while both cache and fetcher generations match. Incompatible
  duplicate registrations fail loudly.
- Decoded data is heap-backed and globally constrained by the service's shared
  decoded-byte budget. Source state and resident entries survive A -> B -> A
  volume switches until global eviction or explicit source invalidation.
- Probe, source-read, and decode schedulers belong to the service. Source-read
  concurrency is one service-global policy, bounded by the physical worker
  capacity fixed when the service is constructed. Source acquisition options
  contain only source-local metadata and persistent-cache policy and cannot
  change scheduling.
- `configureFetchConcurrency()` changes fixed/adaptive admission on the
  service's existing source-read scheduler. It does not replace the scheduler,
  increment source epochs, cancel or restart tasks, duplicate requests, or
  invalidate running results. Increasing admission wakes existing workers;
  after a decrease, running work drains normally and no new task starts until
  activity falls below the new limit.
- Explicit bounded prefill, redownload, and batch caches use a separate
  `ChunkCacheService`. Their fixed concurrency is isolated from the regular
  interactive service even when both services share one decoded-byte budget.
- `Volume` may receive a service or create a private one, but it neither owns
  nor exposes a per-volume source-read concurrency setting.
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
  version through direct misses and sparse frame demand. Exact dependencies
  admitted by asynchronous `SurfaceCache` tile fills are context-free
  background work: they remain owned until resolved and are not cancelled when
  a viewer replaces its frame snapshot.
- Before an accepted interactive render samples the volume, it probes the
  viewport on a deterministic stratified 8-pixel grid. Each probe records the
  2-D viewport occurrence of every required requested-level and permitted
  fallback chunk. It queues at most five coarser levels and may stop earlier
  when one average chunk edge spans the larger viewport extent, with both
  quantities measured in level-0/base-volume voxels. Plane and renderable
  `QuadSurface` parameter coordinates use those same units; camera scale is
  framebuffer pixels per base voxel for both. This fallback demand remains
  present on refinement renders until it resolves.
  Eight pixels is only the sparse probe-cell spacing. Nearby occurrences of the
  same chunk are deduplicated using that level's projected representative chunk
  footprint, computed analytically from chunk shape, declared level transform,
  and framebuffer pixels per base voxel. Distant occurrences on folded
  surfaces are retained. Invalid scale or transform metadata fails dependency
  publication rather than silently reverting to the probe spacing.
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
  Every distinct control point is an exact ribbon support so generated quads
  cannot shortcut stored bends. Each control-point segment is independently
  subdivided at the interval count whose spacing is closest to the default
  50-base-voxel target. Explicit support arclengths provide the bidirectional
  mapping between original fractional point positions and nonuniform ribbon
  grid columns. Input line spacing may be arbitrary; control points, cuts, and
  persistence remain in original line-position coordinates.
- A completed pre-pass atomically replaces that source's previous snapshot for
  the view. The accepted render's captured focus is used locally to reduce each
  chunk's retained occurrences to its nearest squared distance. Snapshot
  construction, distance calculation, and surface-coordinate generation occur
  without the chunk-cache state lock. Publication installs the complete demand
  and re-sorts pending work atomically; older view versions cannot replace a
  newer snapshot.
- Pending interactive work is ordered by coarser view-relative pyramid level,
  active view,
  nearest retained occurrence to that view's focus, then FIFO. A GUI miss not
  observed by the sparse pre-pass has no location and sorts after located work
  at the same view and relative level. It cannot outrank a located coarser
  fallback because relative level is the primary ordering key. The terminal
  source-pyramid level adds 100 to its relative priority, so reaching the best
  available whole-view fallback always outranks ordinary relative levels even
  for a view that starts near the end of a shallow pyramid, while relative
  ordering remains meaningful between terminal-level demands. Dependency
  publication is coarse-to-fine so
  workers cannot admit fine work before its coarse entries are visible.
- Mouse and Agent Bridge canvas interaction store focus in the viewer and mark
  the service's active view with one atomic O(1) update. They do not scan demand,
  query a retained point index, traverse sources, or explicitly re-sort queues.
  Normal stage handoffs may consult the current active view; full pending-queue
  re-sorting occurs when a completed render demand snapshot is published.
  Before any pointer has been observed, viewport center is the captured focus.
- One unresolved source/chunk entry may contain demand from several views.
  New snapshots promote already queued work in place instead of submitting a
  duplicate task. Whole-view closure removes that view's demand from every
  source. Disabling or replacing a different-source overlay closes only that
  source's current view version, preserving base and other-view demand; a newer
  overlay render version reopens it. Same-source overlay demand remains merged
  with base demand and is removed by the next base-only snapshot.
- Regular chunk work passes through three shared pending queues. A 32-worker
  local probe queue classifies persistent data, empty markers, and misses using
  filesystem metadata only. Cache hits enter an eight-worker CPU read/decode
  queue; misses enter the remote source-read queue. Successful source reads
  then enter the decode queue.
- Normal interactive remote source reads use 64 available workers with an
  adaptive admission limit in `[2,64]`. The common HTTP response callback
  reports encoded body bytes for scoped Zarr chunk reads. Remote fetchers
  declare this capability before invocation, and a service-global
  five-second window aggregates concurrent payload bytes over the union of
  remote request-issue-through-completion intervals. Connection and TTFB time
  are included; intervals with no remote request in flight are excluded.
  Saturated adaptive epochs use the same aggregate measurement plus p90 request
  latency and require both five remote-active seconds and at least one
  successful completion per admitted worker. Local and custom fetchers do not
  update displayed network bandwidth, adaptive history, or persisted remote
  state. Bracketed probes compare the settled limit with higher
  and lower limits; initial discovery uses a 4x step and subsequent refinement
  uses 2x steps. Stable bandwidth stretches periodic exploration toward five
  minutes, while a roughly 2x bandwidth change brings it back toward one minute.
  Failed and missing reads end their own request measurements without erasing
  successful observations from concurrent requests. They may pace an
  already-selected admission ramp but do not create successful payload
  samples. Underfilled-tail reads reset rather than establish a capacity
  epoch.
- VC3D persists the settled admission limit, long-term bandwidth EMA, and
  saturated per-worker capacity model in its versioned per-user settings. A
  later run restores and uses the settled limit immediately. Epoch samples,
  probe phase, direction-turn history, instability, and accumulated stability
  time are never persisted: startup resets those values and immediately resumes
  the frequent initial 4x/2x search around the restored operating point.
- Adaptive admission is service-wide for normal remote rendering and changes
  only how many source tasks may start; it does not alter pending-task priority.
  A decrease does not interrupt running work. Explicit fixed-concurrency
  callers, tests, prefill operations, and local volumes use fixed service
  configurations; independent operations require independent services.
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
- The displayed MiB/s and adaptive controller use the same aggregate encoded
  HTTP body-byte estimator over the last five seconds with a remote request in
  flight. Measurement begins at request issue, includes connection and TTFB
  latency, is updated independently of chunk completion, and excludes only
  intervals with no remote request active.
  The `Nx` value remains the actual current number of source fetches in flight.
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
