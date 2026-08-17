---
name: vc3d-atlas
description: Open, inspect, search, remap, and optimize snap candidates for a VC3D fiber Atlas through MCP. Load for vc3d_atlas_* calls or after creating an Atlas from a saved fiber.
---

# Fiber Atlas workflow

Assume `vc3d-bridge-session`, `vc3d-fiber-tracing`, and `vc3d-open-data`. An
Atlas maps saved fibers onto an unwrapped base mesh; it is not a volume atlas.

## Preconditions

1. Select the volume whose provenance matches a manifest-backed `lasagna`
   dataset. A running Lasagna fit service is not a substitute.
2. Save fibers before searching. An unsaved trace is invisible to the search.
3. When available in the resolved manifest metadata, verify it provides
   `init_shell_dir`. This is a dataset field in the `.lasagna.json`, not a
   service setting: its path is resolved relative to the manifest and must name
   a directory containing `shell_*.tifxyz` candidate base surfaces. Atlas maps
   the saved fiber onto one of those initial shells. The field is not guaranteed
   to be exposed by catalog summaries. Without it, `vc3d_fiber_create_atlas`
   fails `-32005` with “Lasagna manifest is missing init_shell_dir”; there is no
   bridge fallback.
4. Create an Atlas from an eligible saved fiber with
   `vc3d_fiber_create_atlas`, or resolve an existing atlas directory.
5. Open it with `vc3d_atlas_open`, then verify `vc3d_atlas_status` reports the
   expected directory/name.

A stale atlas that requires an interactive rebuild cannot be recovered through
the bridge. Preserve the load error and do not continue as if an atlas were
open.

## Search and inspect

1. Page and preserve any existing results before a new search; starting clears
   them.
2. Choose `atlas_to_non_atlas` only when the open Atlas has saved mappings;
   otherwise use `non_atlas_only` if it matches the task.
3. Pass `max_distance` explicitly so persistent GUI settings cannot change the
   search. Add tag filters only from current saved-fiber metadata.
4. Start with `vc3d_atlas_search_start(..., wait=true)`, then verify terminal
   job state and `vc3d_atlas_status.search.resultCount`.
5. Page `vc3d_atlas_search_results` using returned offsets and retain each
   row's current `index`; indices are invalidated by the next search.
6. Open a result with `vc3d_atlas_open_result`. This switches to the
   Intersections workspace. Return to `main` before capturing a main viewer.

The bridge can inspect candidates but cannot record the human verdict
same-winding/different-winding/uncertain. State that boundary explicitly.

Cancellation normally terminates the Atlas job as failed and discards partial
results; report it as a requested cancellation rather than a successful
search.

## Remap and snap candidates

- `vc3d_atlas_remap` returns when work is launched, not completed. It has no
  job id or completion signal; verify only through later status/search behavior
  and report the observability limit.
- `vc3d_atlas_optimize_snap_candidates` also returns request acceptance only.
  It requires the Lasagna fit service and can restart an internal service to
  match a manifest directory, dropping that service's queue. Never call it
  while an optimization that matters is in flight.

Record current volume, Lasagna dataset identity, Atlas directory, saved fiber
ids, search parameters, result count, and all non-job observability limits.
