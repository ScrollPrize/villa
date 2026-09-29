# VC3D render and fetch specification

## Native Bidirectional Trace Fusion

- All callers use the shared native fusion implementation. Meeting positions
  `Pf`, `Pr` with prefix arclengths `Lf`, `Lr` fuse at
  `Pf + Lf/(Lf+Lr)*(Pr-Pf)`. Each prefix receives a linear arclength warp;
  longer prefixes absorb proportionally more of the gap.
- A zero-length prefix stays at its CP, with full endpoint correction on the
  nonempty side. Never create an artificial connector for an empty prefix.
- Candidate ranking, error metrics, acceptance and output spacing are unchanged.
  This does not impose tangent continuity between separate CP spans.

## Invariants

- Rendering values, interpolation, pyramid transforms, and cache contents must
  not change as a side effect of diagnostics or scheduling work.
- Remote fetching must remain asynchronous; UI and render threads must not wait
  on network or persistent-cache I/O.
- S3 sources are attempted anonymously first. A successful response makes
  anonymous access sticky for that store; an anonymous 401/403 retries with
  available credentials and successful authenticated access becomes sticky.
  A later anonymous 401/403 may upgrade a mixed-access store to authenticated
  access. Not-found, unrelated client/server, and transport failures do not
  select a mode. The access mode must not alter source identity, cache paths,
  response bytes, or serialized locations.
- Queue diagnostics must come from the existing chunk-cache request state, not
  from a parallel accounting system at viewer call sites.
- A shared cache must report each unresolved chunk once regardless of how many
  viewers requested it.
- Normal download diagnostics belong in the existing application cache status
  bar, not in per-slice overlays or a persistent setting. The explicit
  `--debug-download-queue` process-start diagnostic may additionally render
  active remote chunks over slice images for request-order investigation.

## Regular chunk-cache ownership

- Core owns one process-lifetime regular `ChunkCacheService`. Every normal
  `Volume` discovers it internally; applications, windows, workspaces, and
  Python callers must not attach or replace services per volume. Main, Spiral,
  overlay, and derived surface-tile input reads acquire source handles from
  this process service rather than construct independent decoded regular-chunk
  pools.
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
  decoded-byte budget. Sources retain usage, LRU touches, and eviction callbacks
  but do not own independent decoded-byte ceilings. Source state and resident
  entries survive A -> B -> A volume switches until global eviction or explicit
  source invalidation.
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
- `configureDecodedByteCapacity()` changes the existing service budget in
  place. It does not replace source handles, increment source epochs, or cancel
  queued/running work. Increasing capacity preserves all decoded entries;
  decreasing it evicts only globally oldest decoded entries until accounting
  satisfies the new ceiling. Concurrent completions drain normally and are
  accounted and enforced against the new ceiling.
- Open Data prefill and Settings redownload submit persistence-only maintenance
  demand to the process service. Maintenance is below interactive and ordinary
  background work but shares source transfers, workers, adaptive admission,
  and connections. Maintenance alone never decodes or consumes decoded RAM.
- Low-level standalone and test workloads may own a separate service, but that
  service owns its complete scheduling and decoded-memory policy. Services do
  not partially share only a decoded-byte budget.
- `Volume` owns neither a service nor per-volume decoded-byte or source-read
  policy. C++ callers configure the process service directly; Python callers
  use module-level regular-cache configuration APIs. Configuration must not
  invalidate or reacquire a source.
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
  Every annotation control point and both line endpoints are exact ribbon
  supports. Optimized line points between adjacent controls define the path but
  are not mandatory columns. Each control-point span is independently
  resampled by optimized-polyline arclength using the interval count whose
  physical spacing is closest to the 8-base-voxel along-line target; a shorter
  span remains one interval.
  The along-strip grid density is always declared as `1/8` samples per base
  voxel, so a short physical control-point span expands to one nominal display
  interval. Explicit support arclengths provide the bidirectional mapping
  between original fractional point positions and nonuniform ribbon columns.
  Both generated ribbons have seven cross rows at `1/32` samples per base voxel,
  giving a fixed 192-base-voxel first-to-last-row extent close to the previous
  typical width without depending on optimized-line spacing; the along and
  cross spacings are independent constants.
  Generated-view clicks collapse all controls within an inclusive 8-base-voxel
  optimized-polyline arclength radius (equal to the along-line sampling
  distance) into one control at the clicked point.
  Before automatic full optimization, every insertion, replacement, or
  multi-control collapse first reconstructs the replacement's surviving
  adjacent spans from its authoritative line position. A multi-control collapse
  must not optimize the replacement directly against the unchanged old line by
  nearest 3-D position. If only one control survives, its reinitialization
  tangent comes from that authoritative old-line position rather than spatial
  projection. Synchronous preparation failure leaves the pre-edit geometry and
  optimization state unchanged; reciprocal branch changes for multi-collapse
  are committed only after asynchronous optimization succeeds.
  The separate maximum control-point extrapolation setting applies only outside
  the outermost controls, measures optimized-polyline arclength in base voxels,
  and does not restrict insertion between existing controls.
  Input line spacing may otherwise be arbitrary; cuts and persistence remain in
  original line-position coordinates.
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
  callers, tests, and local volumes may use fixed service configurations;
  independent operations require independent services.
- Each stage scheduler uses interactive, ordinary background, and maintenance
  work classes. Interactive/background retain their work-conserving 7:1
  admission; maintenance runs only while neither higher class is pending.
  Current view-relative priority is recalculated at every stage handoff,
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

## Annotation focus bounds

- VC3D may hold one optional session focus bounding box in inclusive
  base-volume XYZ coordinates. It is preserved across volume/channel switches
  within one package, cleared when the package is replaced or closed, and is
  not serialized into project data.
- While enabled, ordinary plane views and annotation-role strip/cut views draw
  the final composed framebuffer outside the box at half RGB brightness without
  a boundary line. Base and attached-volume pixels are dimmed together; Qt
  control markers and other scene items remain full brightness.
- Standard segmentation QuadSurface/SurfaceCache views are outside the current
  display scope. Focus-bounds rendering uses coordinates already generated for
  sampling and must not add volume reads, surface intersections, or render
  resubmission loops. Editing configured bounds while disabled must not start a
  different render job.
- New line seeds, branch-fiber seeds, and control-point placements are accepted
  only inside an active box. Validation occurs before pane creation, dataset
  access, autosave flushing, or persistence. Existing controls remain valid
  outside the box; in particular, an already-saved seed-only fiber can be
  reopened and resumed after the box is enabled or moved.
- Every trace or reoptimization uses the box state captured when that work was
  launched. Changing the box does not cancel, restart, or retroactively mutate
  existing or in-flight annotation geometry.
- For work launched with an active box, the entire optimized path between the
  outer controls is retained, including temporary excursions outside the box.
  Open tails stop after their first outside sample. A lone outside control
  retains each contiguous connector that reaches the box, the in-box run, and
  its first sample after exiting. If neither tail reaches the box, it retains
  one adjacent sample to remain valid. Manual provisional edits follow the
  same tail rule.

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

## Fiber width and display normals

CP normals carry `display_normal_source`: manual for explicit edits,
interpolated for creation inheritance, unknown when provenance was not stored.
Only manual normals qualify as manually annotated training data. Copy/merge
the source with its vector; resets omit both JSON fields.

Ribbons use ordinary QuadSurface geometry, rendering and indexed projection.
Only construction resamples the polyline at the existing support spacing and
builds transported, roll-smoothed frames from the corrected input normals.
No cubic strip upsampling or custom runtime surface evaluator is permitted.
CP correction and cross views share the ordinary central-chord tangent.
CP displacement remains cubic,
with zero derivatives at stationary controls and outer local-edit boundaries.

CP `direction` annotations are signed forward vectors in fiber order, unlike
sign-agnostic display normals. Placement aligns their sign with increasing strip
columns, regardless of drag sign; reversing a fiber negates them. Shift-drag in either
strip edits the local in-plane heading while preserving the perpendicular
component. They apply to native trace starts, Lasagna per-side constraints and
all spline CP tangents, irrespective of span length. At an annotated CP they
also define the cross-section normal; save/sync and CP lifecycle preserve them.
Ordinary tracer initial directions remain prediction-selection hints. Only an
explicit annotation fixes the first step (forward vector or its backward
negation). Fusion must preserve those explicit endpoint directions; ordinary
unannotated fusion is unchanged. Direction markers are 80 pixels long.
Away from a CP, Shift-drag inserts one at the drag start on release, through
ordinary placement including inherited normal correction. Direction and creation
are one edit; cancelling or clicking without dragging creates nothing.
Direction annotations have short persistent strip markers sharing the editing
frame. Ctrl-right-click clears only the selected CP's direction and normal
corrections. The annotation window menu clears them for the whole fiber,
including interpolated normal corrections, without changing width/gap.

Display normals are sign-ambiguous axes, including stored CP corrections.
Offset recovery and interpolation use the shortest rotation modulo pi.
Interpolate target axes in a common transported reference, never by adding
interpolated offsets to independently rotating baselines. Feed the resulting
normal field through ordinary alignment and strip construction with no
post-construction correction rotations. Uncorrected spans retain sampled normals.

The annotation toolbar has **Fiber width** (base voxels; 0 means unset) and
**CP angle offset** (degrees). The angle control selects the nearest CP within
the same arclength radius as CP replacement, and is disabled outside that radius.
Setting an angle to zero removes its manual normal.

Version-3 JSON stores optional top-level `width`, always-written top-level
`width_gap_fraction` (default 0.2), and optional per-control-point
`display_normal: [x,y,z]`. The latter is the baked world-space unit normal,
not an angle. The gap is a dimensionless fraction between 0 and 1; missing
values load as 0.2 and are explicitly written on the next save. Existing values
are retained; there is currently no gap editor in the toolbar. Coordinates
scale the width but never this fraction. The stored normal is projected into
the current cross plane after movement,
reoptimization, or Lasagna reload; the spinbox shows its angle relative to the
current Lasagna normal. An unprojectable stored direction is cleared, saved,
and reported with the number of affected CPs.

Display offsets use bounded C1 smoothstep interpolation in arclength between
CPs (shortest angular path). Unset CPs constrain the offset to zero. The viewer
and annotation code share the cut-plane tangent calculation. Corrections affect
the cross-section and both strips, never tracing, alignment metrics, or
optimization inputs.

Width draws inner and outer guide pairs at `(1-gap)*width` and
`(1+gap)*width`: by default 80% and 120% of the full width
(offsets +/-0.40 and +/-0.60 times width from the center). The top strip shows
continuous boundaries; the cross view shows short ticks. Hovering either view
shows cyan edge ticks in both panes around the same world-space hover position,
projected into each view, to preview placement. Cursor ticks
remain 12 screen pixels long; their spacing follows the physical width and
view scale. They hide on leaving the view, invalid data, unset width or hiding
width guides, independently of linked-cursor mirroring.
Any manual normal enables a horizontal cross-view guide. Both guide types can
be hidden in the annotation menu. Graphics items are reused; width paths change
only when the width changes (short tick lengths also adapt on zoom), with
transforms updated for navigation. Mouse movement only repositions the existing
cursor overlay, without requesting a volume render.

**Ctrl+Space** in the fiber annotation window toggles the existing volume
overlay through the main window's shared toggle action. **Space** continues to
toggle cross-view mouse-follow. The annotation menu also exposes the overlay
toggle. This is independent of width/normal guide visibility.

Edits use the ordinary queued fiber save/sync path without retracing or changing
review tags. CP metadata follows CP movement, reversal and split; each split
inherits width and gap. Joining fibers uses the clicked fiber's nonzero width
and its gap, otherwise the other fiber's width and gap. Three-way sync merges
width, gap and per-CP normal changes
independently of span refits; divergent edits and edits to removed CPs conflict.

The former cross-section polygon/line editor and `cross_sections` storage are
removed. Old records are not migrated and are omitted when rewriting a fiber.

## Cross-view Shift Drag

New CP insertion inherits the existing effective display normal in corrected
regions, evaluated before geometry changes. Baseline-only regions remain unset;
replacement metadata and explicitly dragged normals take precedence.

Shift-left-drag edits the nearest nominal width edge or center in the current
cross view. Center movement preserves width and manual normal; edge movement
uses the original opposite edge for orientation, places the selected edge at
the mouse and the center half the unchanged width behind it. Store only CP
position and baked display normal. Holding Shift accumulates multiple strokes
in the preview, without reslicing or reoptimizing; release Shift to commit once
through the regular CP-edit path (wait for mouse release if still dragging);
cancel on Escape/focus loss/geometry replacement. No new persistent fields.
The preview includes inner/outer width-tolerance ticks using the stored gap;
they move and rotate with the dragged frame, with fixed screen length.
Hide original normal/width guides and hover-width previews for the entire
Shift-held sequence; restore their configured visibility on commit/cancel.
