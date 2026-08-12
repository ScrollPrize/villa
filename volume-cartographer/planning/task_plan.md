# Task plan

## Scope and invariants

- Change request ordering only. Preserve sampled values, level transforms,
  interpolation, fill behavior, persistent-cache formats, and source identity.
- Keep one deduplicated unresolved regular-chunk entry per source/key while
  allowing that entry to carry demand from several viewers.
- Use stable integer view IDs and viewport-space points; do not collapse a
  folded surface's multiple chunk occurrences into one rectangle.
- Treat "lower level" as lower spatial resolution/coarser pyramid data, matching
  the current fallback-first behavior.
- Never hold the shared scheduler/cache lock while generating surface geometry
  or traversing viewport samples.
- Running I/O/decode work is not cancelled. New scheduling applies at the next
  admission decision and at persistent-probe to remote-fetch handoff.

## Core request context and scheduler

1. Add explicit request metadata to `IChunkedArray`: background by default,
   or GUI with stable view ID, view version and optional viewport location.
   Preserve existing overloads for CLI/core callers as background requests.
2. Add an application-cache-service scheduler with separate GUI and background
   pending lanes. Admit work to the existing probe and fetch/decode pools only
   when a worker slot is available, so the next item can be chosen from current
   demand rather than being buried in an immutable executor heap.
3. Track versioned per-view demand slots for each unresolved chunk. Select GUI
   work by active view, coarse level, nearest occurrence, then FIFO. Interleave
   background admissions with GUI admissions using a bounded weighted policy
   while remaining work-conserving.
4. Preserve request identity across stages. Re-evaluate priority when a disk
   probe misses and the chunk enters the remote-fetch lane. Invalidation and
   stale fetch serial checks must continue rejecting obsolete publications.
5. Replacing a view snapshot removes that view's old slots, adds the new slots,
   and queues/promotes all currently missing chunks atomically. Clearing or
   destroying a viewer removes only that viewer's demand.

## View pre-pass and point indexing

1. Add reusable dependency-point collection helpers to
   `ChunkedPlaneSampler`. A dependency point associates every nearest/trilinear
   required chunk with the originating 2-D viewport sample without queueing or
   touching cache LRU state.
2. Sample the viewport on an 8-pixel stratified grid. Derive deterministic
   jitter from view/render version and grid cell so successive renders cover
   different positions while test runs remain reproducible.
3. Deduplicate points per chunk in screen space with an 8-pixel radius/grid,
   then bulk-build a 2-D `PointIndex` (`z=0`) whose collection IDs map to chunk
   keys. Compute each chunk's nearest retained occurrence to the captured focus
   point for scheduler ordering.
4. Publish the locally completed snapshot immediately before normal rendering.
   Actual sampler misses after publication use the same GUI context with no
   position unless the caller has an exact viewport point.

## Surface geometry reuse

1. Move direct-render coordinate generation ahead of the pre-pass whenever the
   full render needs generated coordinates. Reuse the existing
   `GeneratedSurfaceCache` result for both dependency collection and sampling.
2. Add a sparse view-coordinate probe to `SurfaceGeometryTileCache`. It reads
   or generates the same level geometry tiles used by SurfaceCache fills and
   returns coordinates/normals only at requested viewport samples.
3. When base and overlay rendering are fully SurfaceCache-backed, build the
   pre-pass from this geometry-tile probe instead of allocating full-frame
   coordinate mats. The following `requestView`/fill work reuses those tiles.
4. Carry the originating GUI request context through `SurfaceCache::requestView`
   into asynchronous fills and their blocking dependency prefetch calls.

## VC3D integration

1. Assign every `CChunkedVolumeViewer` a stable numeric view ID and track its
   last valid viewport cursor/focus, falling back to viewport center.
2. Mark a view active on mouse interaction and publish active-view state through
   the shared cache service. Capture focus and demand version in accepted render
   jobs; do not run a pre-pass for jobs merely stored in `_pendingRenderJob`.
3. Run the pre-pass inside the accepted render worker before any normal chunk
   miss can queue. Apply it independently to base and active overlay sources,
   sharing one view ID/version.
4. Move accepted `SurfaceCache::requestView()` admission from the UI-side
   `startRenderJob()` setup to the render worker, immediately after snapshot
   publication. This ensures its asynchronous fills inherit the completed GUI
   demand instead of queueing unlocated dependencies ahead of the pre-pass.
5. Clear view demand on volume/source replacement and viewer shutdown so stale
   old-volume work becomes background rather than interactive priority work.

## Testing and measurement

- Add core scheduler/cache tests for:
  - existing queued chunks adopting new GUI demand;
  - active view before inactive view;
  - coarse level before pointer distance;
  - nearest of multiple viewport occurrences;
  - no-location GUI ordering within a level;
  - atomic snapshot replacement and stale-version rejection;
  - weighted, work-conserving GUI/background fairness;
  - priority re-evaluation between persistent probe and remote fetch;
  - source/view isolation and cleanup.
- Add sampler tests for deterministic jitter, dependency-point correctness,
  per-chunk point deduplication, and nearest/trilinear corner coverage.
- Add SurfaceGeometryTileCache tests proving sparse view probes reuse generated
  tiles and match direct surface coordinates within the cache's sampling
  convention.
- Build and run:

  ```bash
  cmake --build volume-cartographer/build --target test_chunk_cache test_chunk_cache_persist test_chunked_plane_sampler VC3D -j4
  ctest --test-dir volume-cartographer/build --output-on-failure -R '^(test_chunk_cache|test_chunk_cache_persist|test_chunked_plane_sampler)$'
  git diff --check
  ```

- Run the synthetic Valgrind/Callgrind rendering gate after implementation and
  compare with the accepted unified-cache result. Also report pre-pass cost and
  point/chunk counts from a deterministic synthetic fixture. The gate is
  virtualized and may run despite unrelated host load.

## Spec update

- Replace scalar newest-render priority with versioned multi-view GUI demand.
- Specify viewport occurrence points, pre-pass publication, active/level/distance
  ordering, no-location semantics, and GUI/background fairness.
- Specify that direct rendering and SurfaceCache-backed rendering reuse their
  respective surface geometry paths for the pre-pass.
- Specify explicit GUI request propagation through asynchronous SurfaceCache
  fills and late render misses.

## Docs updates

- Extend `docs/remote_file_cache.md` with request stages, GUI/background lanes,
  view snapshots, focus ordering, and SurfaceCache geometry reuse.
- Document that running work is not cancelled and that priority changes apply
  to pending work and stage handoffs.

## Changelog update

- Add a dated entry for focus-aware multi-view chunk scheduling and the
  reduced-resolution render pre-pass after implementation and validation.
