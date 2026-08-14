# Task: restore render-owned chunk priority and remote-source lifecycle

Address the three unresolved review findings on PR #1453 without changing
rendered values, interpolation, source-level selection, or cache identity.

## Follow-up: adaptive download restart state

Persist VC3D's reusable adaptive download operating point across clean runs.
Restore the settled queue limit immediately, but reset the stability window and
probe-phase history so startup explores frequently around the restored value.

## Interactive demand and focus

- Mouse movement stores the latest viewport pointer in the viewer and updates
  the cache service's single active-view ID. It must not scan retained demand
  points, traverse source states, mutate per-chunk demand, or reprioritize
  scheduler entries.
- Every accepted interactive render continues to run the randomized,
  stratified 8-pixel dependency prepass. Eight pixels is the sampling interval,
  not the per-chunk deduplication radius.
- Dependency occurrences must be deduplicated separately for each chunk and
  source level using that level's declared chunk footprint in screen pixels.
  Multiple occurrences of one chunk remain only when they are separated by at
  least one chunk footprint, as can occur on folded surfaces.
- An accepted render captures the viewer's current focus in its render job.
  The completed prepass uses that captured focus, computes nearest retained
  occurrence distances locally, and atomically replaces and reprioritizes that
  view's demand. Mouse input may change the active view immediately, but it
  does not walk or re-sort pending work itself.
- The retained point collection must not be queried from the GUI event path.
  Remove persistent point-index state and specialized APIs if they have no
  remaining users.

## Reopened remote sources

- Reopening the same canonical remote source with refreshed temporary
  credentials must adopt the new fetchers while retaining compatible decoded
  chunks and the same numeric `VolumeSourceId`.
- Pending, failed, and running work created with superseded fetchers must not
  publish stale results or keep the source permanently bound to expired
  credentials.
- Fetcher refresh must use a generation independent of destructive source/cache
  invalidation so ready decoded chunks remain resident.
- Authentication material must remain absent from source identities,
  persistent-cache paths, diagnostics, and logs.

## Overlay lifecycle

- Disabling a different-source overlay, including by setting opacity to zero,
  must remove that source's demand for the view without removing base-volume
  demand or another view's demand.
- Completion callbacks from already-running inactive-overlay work must not
  schedule redundant renders.
- Same-source base/overlay demand remains merged and is replaced by the next
  base snapshot; it must not be cleared as a separate source.

## Constraints

- Preserve rendering numerics and deterministic request ordering for equal
  priorities.
- Preserve application-wide decoded-cache sharing and warm volume switching.
- Keep expensive dependency traversal and distance calculation in the accepted
  render worker, outside the scheduler/cache lock.
- Atomically publish complete demand snapshots; queue workers must never
  observe partially replaced per-view state.
