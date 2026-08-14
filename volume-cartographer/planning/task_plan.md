# Task plan

## Scope and history

The focus-priority implementation was introduced by `e9416cc21a` on
2026-08-12. That commit's plan conflated the randomized 8-pixel prepass spacing
with the chunk-occurrence deduplication radius and explicitly placed
point-index distance recalculation in the mouse path. The implementation has
matched that incorrect plan since its introduction; this is not a later
regression. The atomic publication changes in `0f4f98c5c1` preserved the
problem but did not introduce it.

This task corrects all three unresolved PR #1453 findings together because they
share the application-wide chunk service and per-view demand lifecycle.

## Implementation

### 1. Keep focus in the viewer and active view in the cache service

1. Keep viewport focus ownership in `CChunkedVolumeViewer` through the existing
   `_lastScenePos`/`_haveChunkFocus` state. Continue to use viewport center when
   no pointer has been observed.
2. Keep capturing that focus in `PendingRenderJob::renderFocus` when a render
   job is accepted. The cache service does not need a per-view focus map or
   another focus-generation mechanism.
3. Replace `IChunkedArray`/`ChunkCache::updateViewFocus(viewId, focus,
   makeActive)` with `markViewActive(viewId)`, which only stores the one atomic
   active-view ID. Remove the unused focus and boolean parameters rather than
   keeping a compatibility API with misleading semantics. It must not enumerate
   source states, inspect demand points, walk entries, or submit scheduler
   reprioritization.
4. `workPriorityLocked()` and scheduler stage handoffs read that single active
   view directly. Pointer input may update it immediately; existing pending
   task priorities remain unchanged until normal stage handoff or render demand
   publication recalculates them.
5. Ensure physical mouse and Agent Bridge canvas interaction paths use the same
   O(1) active-view operation while retaining focus locally in the viewer. Mark
   the shared service once through the base cache facade; do not repeat the same
   active-view write through a different-source overlay facade.
6. On actual viewer closure, clear the active ID only if it names that view.
   Source-local overlay cleanup does not affect active-view state.

### 2. Deduplicate by declared chunk footprint

1. Keep `stratifiedViewportSamples()` at its randomized 8-pixel cell spacing.
   Document this solely as sparse coverage sampling.
2. Extract one shared helper for a source level's representative chunk extent
   in level-0/base-volume voxels from `chunkShape` and the declared
   `LevelTransform`. Reuse the established transform validation and the same
   representative extent policy used by viewport fallback bounds rather than
   duplicating the calculation.
3. Convert that level-specific base-voxel chunk extent to framebuffer pixels
   using the render's declared `pixelsPerLevel0VolumeVoxel`. Do not infer it
   from generated coordinates, finite differences, cache residency, or local
   surface distortion.
4. Remove the optional fixed `dedupRadiusPixels` parameter from
   `collectViewportDependencies()`. Require the declared
   `pixelsPerLevel0VolumeVoxel` input and compute each level's radius internally
   from chunk shape and level transform. Nearest and trilinear dependencies
   continue to deduplicate independently by exact source-qualified chunk key.
5. Preserve points for the same chunk when their screen-space separation is at
   least one projected chunk footprint. This retains genuinely separate
   appearances on folded surfaces while reducing planar demand toward one
   occurrence per visible chunk region.
6. Compute base and overlay footprints independently from each source's chunk
   shapes, transforms, selected start level, and common declared view scale.
   Handle absent/invalid level metadata by failing the dependency publication
   clearly rather than silently reverting to an 8-pixel dedup radius.

### 3. Make accepted render publication own reprioritization

1. After the prepass has produced and deduplicated its local occurrence list,
   use `PendingRenderJob::renderFocus` and group occurrences by chunk. Do not
   retrieve focus from cache-service state.
2. Compute each chunk's nearest squared viewport distance while grouping the
   local list. This work remains on the render worker and outside the global
   scheduler gate and source-state lock.
3. Atomically replace the view's previous chunk slots with the completed
   `{view version, relative level, nearest distance}` snapshot. Preserve stale
   version rejection and cancellation of entries with no remaining GUI or
   background owner.
4. During that same publication, reprioritize affected pending entries across
   source states using the current active view. This is the render-time re-sort
   required by the view model; it must never be initiated directly by mouse
   movement. Independent stage handoffs may also use the current active view
   when naturally calculating the next stage's priority.
5. Keep queue-stage handoff priority recalculation so probe, source, and decode
   stages use current view state. Preserve active view, relative level, pointer
   distance, and FIFO ordering, including the terminal-level bonus.
6. Remove focus, `PointIndex`, and collection-key storage from persistent
   `ViewSnapshot` state. Keep only snapshot lifecycle/relative-level metadata
   still needed by direct misses and stale-version rejection. Remove
   `nearestPerCollection()` and its dedicated test only if repository-wide
   search confirms this demand path is its sole user; retain the general
   `PointIndex` and all independently used APIs.
7. Keep snapshot construction and distance calculation local, followed by one
   scheduler-gated publication. Queue workers must see either the old complete
   snapshot or the new complete snapshot.

### 4. Refresh fetchers for an existing remote source identity

1. Audit fetch, probe, decode, error, and generation ownership before editing
   the constructor reuse path. Introduce one source-state refresh operation
   rather than copying invalidation logic into `Volume`.
2. When `ChunkCache` registers a metadata-compatible existing source identity,
   atomically adopt the newly opened fetcher set instead of discarding it.
   Keep the existing source ID, decoded ready entries, LRU position, byte
   accounting, listeners, persistent-cache configuration, and view demand.
3. Add `fetcherGeneration_` independently from the existing destructive cache
   `generation_`. Probe, persistent decode, source fetch, and fetched decode
   work records both generations and cannot publish or advance stages unless
   both still match. Refresh increments only `fetcherGeneration_`; it must not
   use `invalidateState()` or clear ready entries, LRU state, decoded-byte
   accounting, listeners, or persistent-cache state.
4. Never replace or read the mutable fetcher vector concurrently. Capture the
   level fetcher's `shared_ptr` under the source-state lock together with both
   generations and carry that immutable handle through probe metadata use,
   persistent decode, source read, and fetched decode. A refresh swaps the
   vector under the same lock; running work may retain an old fetcher handle but
   its old fetcher generation cannot publish.
5. Cancel and requeue unresolved work which depended on the old fetchers while
   preserving its GUI/background ownership and priority. Clear retryable
   authentication/error state so the refreshed source can recover. Do not
   discard compatible decoded ready chunks or valid persistent encoded data.
6. Assign each retained unresolved entry a new fetch serial and enqueue exactly
   one replacement task immediately during refresh. A stale running completion
   checks both generations and fetch serial; it must either observe that
   replacement or enqueue one under the state lock when retained demand still
   has no current-generation task. It cannot overwrite status, bytes,
   persistence flags, or remote-activity state belonging to the replacement.
7. Make repeated refreshes deterministic and safe while old source reads are
   returning. A stale completion may drain but cannot overwrite a newer result,
   strand retained demand, or reintroduce an expired fetcher.
8. Keep source identity canonical URL plus selected base level. Do not hash,
   compare, persist, or log access keys, secret keys, session tokens, signed
   query strings, or credential fingerprints.

### 5. Clear inactive overlay demand at source scope

1. Separate service-wide view closure from source-scoped view-demand removal.
   The existing `clearViewDemand()` traverses all service sources, so it must
   not be used unchanged to disable one overlay after base demand publication.
2. Add a source-bound `clearSourceViewDemand(viewId, minimumVersion)` operation
   on the `ChunkCache` facade. Its bound state/source ID determines the only
   source it may modify. It removes that source's slots, cancels newly unowned
   pending work, preserves demand from other views and background callers, and
   neither marks the whole view closed nor changes active-view state. Keep
   service-wide `clearViewDemand()` for actual viewer closure.
3. When opacity transitions to zero or an overlay otherwise becomes inactive:
   - for a different-source overlay, clear only the old overlay source demand;
   - for a same-source overlay, do not clear the shared source, because the next
     base snapshot naturally omits overlay-only dependencies.
4. Use source-scoped removal when replacing an old different-source overlay.
   Retain service-wide view closure only for actual viewer destruction.
5. Guard overlay chunk-ready and remote-activity callbacks with current overlay
   identity, active opacity, and an overlay generation token so already-queued
   UI callbacks from earlier overlay states cannot trigger redundant renders or
   debug-overlay refreshes after disablement. Increment the token on disable,
   replacement, and re-enable.
6. Setting opacity to zero always schedules a newer render. For a same-source
   overlay, that base-only render's `replaceViewDemand()` is the sole mechanism
   which removes overlay-only occurrences; no source clear is issued.
7. Re-enabling the overlay must publish a newer view version normally; source
   cleanup must not permanently mark the entire viewer closed.

### 6. Cleanup and compatibility

1. Remove obsolete snapshot members, comments, and compatibility helpers made
   dead by render-owned distance calculation.
2. Preserve public API compatibility where there are real external callers.
   Do not retain private dead APIs solely for compatibility.
3. Preserve local-volume behavior, background callers, persistent-cache file
   formats, cache status metrics, debug download overlays, and rendering
   numerics.
4. Keep all new data structures portable across Ubuntu/macOS and amd64/arm64;
   do not depend on nonportable packed-atomic assumptions without a guarded
   fallback.

## Spec update

Update `planning/spec.md` during implementation:

- Distinguish the randomized 8-pixel prepass sampling interval from the
  level-specific projected chunk-footprint deduplication radius.
- Define chunk footprint analytically from declared level transforms, chunk
  shape, and framebuffer pixels per base voxel.
- State that mouse/Agent Bridge pointer movement only updates service-level
  active view in O(1), while viewport focus remains viewer-owned and is captured
  in accepted render jobs. Mouse movement never scans demand or explicitly
  reprioritizes scheduler entries. Scheduler stage handoffs may consult the
  current active view when naturally computing their next-stage priority.
- State that accepted render publication computes nearest occurrence distances
  and atomically re-sorts pending work using current view information.
- Remove the current requirement that mouse interaction updates distances
  against a retained point index.
- Define reopening a compatible canonical source as refreshing fetchers while
  retaining decoded entries and rejecting stale fetcher-generation results.
- Define source-scoped overlay demand removal separately from service-wide
  viewer closure, including same-source overlay behavior.

## Documentation updates

- Update `docs/remote_file_cache.md` with:
  - render-owned focus-distance calculation;
  - chunk-footprint occurrence deduplication;
  - viewer-owned focus and O(1) physical/Agent Bridge active-view updates;
  - credential/fetcher refresh semantics for an existing source identity;
  - source-scoped overlay demand cleanup.
- Update `ChunkCache`, `ChunkedPlaneSampler`, `IChunkedArray`, and viewer API
  comments with the same ownership and unit contracts.
- Update Agent Bridge-facing documentation only if it currently promises
  different cursor or overlay lifecycle behavior; no MCP protocol change is
  planned.
- Replace the active task log/status as implementation proceeds and add one
  concise changelog entry after validation.

## Testing

### Adaptive download restart state

1. Expose a small core snapshot containing only settled admission, long-term
   bandwidth, and saturated per-worker capacity.
2. Seed the shared adaptive scheduler from that snapshot, clamped to its current
   worker bounds. Do not restore epochs, phase, turn count, or stability time.
3. Load and save a versioned snapshot through VC3D's existing per-user INI;
   explicitly sync it before the application's `_Exit` path.
4. Test that restoration starts at the saved admission limit and that the first
   completed baseline immediately begins the normal initial probe search.

### Interactive priority and deduplication

- Add a cache/scheduler regression proving the narrowed active-view operation
  does not query point data, enumerate source states, mutate demand slots, or
  submit scheduler updates, even with a deliberately large demand set.
- Prove a pointer update changes the active view in O(1) without walking or
  re-sorting pending entries. Permit later stage handoffs to observe it through
  their normal priority calculation.
- Prove an accepted newer render captures the viewer's latest focus and changes
  pending chunk order only when its complete snapshot is published.
- Prove render publication re-sorts pending work across shared source
  schedulers using the current active view without exposing partial snapshots.
- Add sampler tests with known chunk shapes, transforms, and view scales which
  distinguish the 8-pixel sampling grid from much larger projected chunk
  footprints.
- Prove planar occurrences collapse according to chunk footprint and that two
  distant appearances of one chunk on a folded fixture remain distinct.
- Cover anisotropic transforms, multiple fallback levels, nearest/trilinear
  dependencies, base/overlay differences, invalid metadata, and deterministic
  jitter.

### Fetcher refresh

- Register one remote source with a fetcher that blocks or returns an auth
  error, then reopen the same identity with a working fetcher. Verify pending
  demand completes through the new fetcher without restarting the service.
- Verify stale old-generation completion cannot publish after refresh.
- Exercise remote fetch, persistent decode, and fetched decode across a refresh
  to prove all use immutable fetcher handles and both generation checks.
- Verify already decoded chunks remain resident and are not fetched again.
- Verify source ID, decoded-byte accounting, listeners, and persistent-cache
  hits remain stable across refresh.
- Verify logs and identities contain no credential material.

### Overlay lifecycle

- Add viewer/cache tests proving opacity zero on a different-source overlay
  removes only overlay demand while base and other-view demand remain.
- Prove same-source base/overlay demand is not incorrectly cleared.
- Prove already-running inactive-overlay callbacks do not submit renders and
  re-enabling publishes a newer valid demand snapshot.
- Exercise the same behavior through the Agent Bridge overlay command contract
  where the existing test harness supports it.

### Validation commands

- Build the affected core and VC3D targets with the existing developer build.
- Run focused `test_chunk_cache`, chunk scheduler, point-index, chunked plane
  sampler, volume, overlay/viewer, and Agent Bridge contract tests.
- Reproduce the Linux Clang QuickBuild core CI configuration, using
  `VC_USE_SCCACHE=OFF` only when local `sccache` is unavailable:

  ```bash
  cmake -S . -B build/ci-fast-core -G Ninja \
    -DCMAKE_BUILD_TYPE=QuickBuild \
    -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_Fortran_COMPILER=flang \
    -DCMAKE_AUTOGEN_PARALLEL=4 \
    -DVC_QUICKBUILD_OPT_LEVEL=0 -DVC_USE_SCCACHE=OFF \
    -DVC_TESTING=ON -DVC_BUILD_APPS=OFF \
    -DVC_BUILD_UI_TRACER=ON -DVC_BUILD_FLATBOI=OFF
  cmake --build build/ci-fast-core --target vc_test_core --parallel 4
  ctest --test-dir build/ci-fast-core --output-on-failure \
    --parallel 4 -L '^vc-core$'
  ```

- Run the virtualized synthetic rendering benchmark before and after. Record
  command, dataset, build type, repetitions, and mean plus min/median/max or
  p50/p95. Confirm rendered outputs remain identical.
- Run `git diff --check`.

## Changelog update

After implementation and validation, add one dated entry describing:

- render-owned focus priority with projected chunk-footprint deduplication;
- recoverable credential refresh for shared remote sources;
- source-scoped cleanup for disabled overlays.

## Explicitly deferred

- Cancelling network requests already executing inside the HTTP client; stale
  work may drain but must be prevented from publishing.
- Changing interpolation, selected source levels, rendered values, cache file
  formats, or adaptive download policy.
- Changing the Agent Bridge protocol or tool schemas.

## Independent plan review

- Completed against `planning/spec.md`, `planning/plan.md`, `planning/task.md`,
  current cache/scheduler/viewer code, and all three PR #1453 comments.
- Incorporated findings: a distinct non-destructive fetcher generation,
  immutable fetcher capture for every fetch/decode path, guaranteed stale-work
  replacement, removal of the fixed dedup-radius API, explicit source-bound
  cleanup semantics, overlay callback generation checks, and same-source
  opacity-zero publication behavior.
- The suggested requested-versus-installed active-view split was reviewed and
  intentionally omitted: no component needs a separately installed value.
  Mouse input updates the one active view cheaply; render publication performs
  the expensive queue re-sort, while normal stage handoffs may consult the
  latest active view directly.
- The viewer already owns focus and captures it in `PendingRenderJob`, so no
  cache-service focus map or focus synchronization API is added.
