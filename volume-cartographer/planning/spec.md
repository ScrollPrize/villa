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
- Cooperative cancellation is not part of this phase. A shared
  `beginViewRequest(discardPending=true)` advances newest-view priority without
  deleting shared unresolved work.
- View epochs are allocated by the application cache service, not independently
  per source, so newly selected volume work outranks older queued source work.

## Interactive chunk scheduling

- Regular chunk work has separate interactive and background pending lanes.
  Existing context-free `IChunkedArray` calls are background work. VC3D render
  requests carry a stable numeric view ID and a monotonically increasing view
  version through direct misses, prefetches, and asynchronous `SurfaceCache`
  fills.
- Before an accepted interactive render samples the volume, it probes the
  viewport on a deterministic stratified 8-pixel grid. Each probe records the
  2-D viewport occurrence of every required requested-level and permitted
  fallback chunk. It queues at most five coarser levels and stops earlier when
  one average chunk edge at a candidate level spans the average viewport edge.
  This fallback demand remains present on refinement renders until it resolves.
  Nearby occurrences of the same chunk are deduplicated, but distant
  occurrences on folded surfaces are retained.
- A completed pre-pass atomically replaces that source's previous snapshot for
  the view. Snapshot construction and surface-coordinate generation occur
  without the chunk-cache state lock. Older view versions cannot replace a
  newer snapshot.
- Pending interactive work is ordered by active view, coarser pyramid level,
  nearest retained occurrence to that view's focus, then FIFO. A GUI miss not
  observed by the sparse pre-pass has no location and sorts after located work
  at the same view and level. Dependency publication is coarse-to-fine so
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
  queue; misses enter the remote source-read queue, whose concurrency is set by
  `maxConcurrentReads`. Successful source reads then enter the decode queue.
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
