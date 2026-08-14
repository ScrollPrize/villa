# Task log

## Planning audit

- PR #1453 has three current unresolved comments: synchronous full demand-point
  scans on cursor movement, discarded refreshed fetchers for reused remote
  source identities, and retained different-source overlay demand after opacity
  becomes zero.
- `ChunkCache::updateViewFocus()` currently enters the scheduler publication
  gate, enumerates every shared source state, clears installed distances, scans
  `PointIndex::nearestPerCollection()`, and reprioritizes demanded entries.
- `CChunkedVolumeViewer` already owns the latest focus in `_lastScenePos` and
  captures it in `PendingRenderJob::renderFocus`. No other component reads a
  cache-service focus value, so adding a service-level focus map would duplicate
  existing state.
- `collectViewportDependencies()` samples on an 8-pixel stratified grid and
  also uses a fixed 8-pixel per-chunk dedup radius. Those are separate concepts;
  the latter should be the declared projected chunk footprint for that level.
- The incorrect design was introduced in `e9416cc21a` (`Prioritize VC3D chunk
  work by view focus`) on 2026-08-12. Its committed task plan explicitly chose
  both the fixed 8-pixel dedup radius and mouse-time point-index recalculation.
  No later commit lost a previously correct implementation.
- Existing source registration validates metadata and returns the retained
  source state without adopting newly opened fetchers. Canonical identity
  intentionally excludes authentication, so temporary credential rotation
  requires an explicit fetcher refresh lifecycle rather than a new identity.
- Fetch/decode tasks currently dereference the source state's fetcher vector
  outside its mutex. Refresh therefore requires tasks to retain an immutable
  per-level fetcher `shared_ptr` captured with a generation under lock; directly
  swapping the vector alone would introduce a data race and mixed-generation
  decode behavior.
- `ChunkCache::clearViewDemand()` currently traverses all source states in the
  service. Disabling one different-source overlay therefore needs a distinct
  source-scoped removal operation; calling the existing method after base
  publication could also remove base demand.
- Same-source overlays merge into base demand and should be removed by the next
  base snapshot, not by clearing the shared source.

## Constraints confirmed

- Mouse and Agent Bridge pointer updates retain focus in the viewer and change
  only the cache service's atomic active-view ID.
- Accepted render publication owns occurrence-distance calculation and queue
  reprioritization.
- The 8-pixel value remains only the randomized sparse sampling interval.
- Deduplication uses declared level transforms, chunk shape, and view scale; it
  does not inspect generated-coordinate finite differences.
- Refreshed credentials preserve decoded chunks and source identity while stale
  source work cannot publish.
- Overlay cleanup is source scoped and does not disturb base, other-view, or
  background ownership.
- Rendering numerics, LOD selection, and cache formats remain unchanged.

## Implementation

- Added a versioned adaptive-download snapshot containing the settled admission
  limit, long-term bandwidth EMA, and saturated per-worker capacity. VC3D loads
  it from `VC3D.ini`, seeds the application-wide scheduler, and explicitly
  writes it before `_Exit` on clean shutdown. Runtime epochs, probe phase,
  direction history, and stability duration are reset, so the restored limit is
  active immediately but startup still performs frequent 4x/2x probes.
- Replaced `updateViewFocus()` with O(1) `markViewActive()`. The viewer retains
  focus and accepted render jobs publish nearest per-chunk distances from their
  local occurrence lists. Persistent cache snapshots no longer retain a
  `PointIndex`; the now-unused `nearestPerCollection()` API and test were
  removed.
- Split sparse prepass spacing from occurrence deduplication. Each source level
  now derives its dedup radius from the shared representative chunk-extent
  helper and the render's declared pixels per base voxel. The same helper is
  used by fallback-range selection.
- Compatible source reuse now adopts new fetchers. A non-destructive fetcher
  generation and immutable per-task fetcher context reject old probe/fetch/
  decode results while retaining decoded chunks, listeners, source ID, LRU,
  accounting, and persistent-cache state. Retained in-flight/error demand is
  deterministically requeued; unowned stale in-flight entries are removed.
- Added source-bound view-demand cleanup. Different-source overlay disable and
  replacement close only that source's current view version, while same-source
  overlays are replaced by the next base-only render. Overlay callbacks carry
  an atomic attachment/opacity generation and verify current identity, opacity,
  and close state on the UI thread.
- Remote activity now tracks individual fetch serials per chunk so overlapping
  old/new credential-generation reads cannot clear one another's activity.

## Deviations

- No functional requirements were deferred. The plan's proposed explicit
  viewer/Agent Bridge overlay integration test was covered at the shared cache
  lifecycle boundary because the existing offscreen bridge harness does not
  expose pending source-demand internals; callback guards remain covered by the
  VC3D compile path and code-level generation contract.

## Validation

- `cmake --build volume-cartographer/build --target test_chunk_cache test_chunked_plane_sampler test_point_index VC3D --parallel 4`
- `volume-cartographer/build/bin/test_chunk_cache` (68 cases passed)
- `volume-cartographer/build/bin/test_chunked_plane_sampler` (19 cases passed)
- `volume-cartographer/build/bin/test_point_index` (26 cases passed)
- `cmake --build volume-cartographer/build/ci-fast-core --target vc_test_core --parallel 4`
- `ctest --test-dir volume-cartographer/build/ci-fast-core --output-on-failure --parallel 4 -L '^vc-core$'`
  passed 130 of 131 tests. The unrelated `test_ink_detection_overlay`
  attempted to write `/home/hendrik/.VC3D/current_project.json.tmp`, which is
  outside the writable test sandbox. The three affected Clang-built tests were
  rerun directly and all passed.
- `ctest --test-dir volume-cartographer/build/ci-fast-core --output-on-failure -R '^(test_chunk_cache|test_chunked_plane_sampler|test_point_index)$'`
- `ctest --test-dir volume-cartographer/build/ci-render-benchmark --output-on-failure -R '^(test_render_synthetic_fixture|test_render_valgrind_ci_no_site)$'`
- `cmake --build volume-cartographer/build/ci-render-benchmark --target render_valgrind_ci --parallel 4`
  passed all serial and parallel scenarios. Modeled cost ranged from `0.967x`
  to `1.048x` of the frozen reference.
- `git diff --check`

## Independent review

- Completed against task, specification, overarching plan, current code, and
  all PR #1453 comments.
- The review proposed separate requested and scheduler-installed active-view
  IDs. This was rejected after user review because no component needs a
  separately installed value. There is one active view: mouse input changes it
  in O(1), render publication performs explicit queue re-sorting, and normal
  stage handoffs may consult the latest value when computing their own priority.
- Follow-up code audit confirmed that scheduler code receives only a materialized
  `ChunkWorkPriority::activeView` boolean and no other cache consumer reads the
  active view ID. It also confirmed that focus is already viewer-owned and
  captured per render job, eliminating the proposed service focus state.
- It required `fetcherGeneration_` to remain separate from destructive cache
  invalidation, immutable fetcher handles for persistent and fetched decode,
  and guaranteed replacement of retained stale work.
- It required removing the fixed-radius dependency API rather than merely
  changing its default.
- Its source-ID cleanup recommendation is represented as a source-bound facade
  operation; an explicit source-ID argument would be redundant and could permit
  a facade/source mismatch.
- It added overlay callback generation checks and explicit same-source
  opacity-zero behavior. All findings are incorporated in `task_plan.md`.
- Final review found no rendering-numeric changes: sparse dependency discovery
  changes only scheduler demand metadata, while sampling and compositing paths
  are unchanged. The new state uses standard mutexes, atomics, containers, and
  `shared_ptr` ownership without platform-specific assumptions.
