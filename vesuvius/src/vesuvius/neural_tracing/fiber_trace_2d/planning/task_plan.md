# Plan: reload replay artifacts in napari

## Reload contract

1. Add a replay-only `Reload artifacts` command to the existing dock. It rereads
   the original root `fiber_replay.json`, following its newly published
   content-addressed generation, and runs the same strict bundle/artifact
   loaders used at startup.
2. Require unchanged failed-status class, prediction manifest content hash,
   prediction shape/scale, selected Zarr level transform, base crop,
   extraction-tube radius, failed artifact-kind set, and five-stage identity.
   Permit changed counts including empty/nonempty transitions, geometry,
   metrics, reference/trace/failure data, and generation paths.
3. Keep the exact existing lazy Zarr source object, crop, image layer, selected
   level, scale, and translation. Do not call the Zarr opener, crop loader, or
   resolver during reload. Replacing the image layer's derived mask graph is
   allowed only when it is built from that identical source object.

## Layer updates

4. Extract shared artifact loading, topology, compatibility, and feature helpers
   so startup and reload cannot diverge. In replay mode, create stable typed
   layers for every artifact kind even when its initial population is empty.
   Update the existing reference, trace, failure,
   final-anchor, five stage, cell-center, refinement-offset, and fiberlet layers
   in place. Update stage names/counts and feature tables and fiberlet features;
   clear stale item selection before row-count changes.
5. Preserve layer identity, order, visibility, clipping, widths/sizes, path
   colormap, and current radius values. Restore the fixed diagnostic colors
   after data replacement so per-item arrays match the new counts.
6. Recompute the presence distance transform and exact anchor representative
   distances from the reloaded reference/trace artifacts. Reapply the current
   independent presence and anchor radii. This is derived display state, not a
   Zarr reload.
7. Split reload into prepare and commit. Preparation strictly loads and
   validates every file and computes EDT, exact anchor distances, features,
   colors/visibility, and names before touching Napari. Commit snapshots all
   affected layer/controller fields, clears item selections, applies under
   blocked events, and refreshes once. Any setter failure restores every layer
   and derived controller field before reporting a terminal/viewer error.

## Tests and validation

8. Add focused tests for prediction-fingerprint and topology compatibility,
   changed-count layer replacement, feature alignment, selection clearing,
   style preservation, incompatible reload rejection, and unchanged lazy Zarr
   source identity/no-opener behavior. Test derived-mask/distance recomputation,
   current-radius reuse, repeated reload, zero/nonzero transitions, and injected
   commit failure rollback.
9. Run focused viewer pytest, Ruff, Python compilation, diff hygiene, and a
   strict reload of two local replay generations when fixtures permit. A live
   napari smoke remains conditional on the installed GUI environment.

## Spec update

- Define the replay-only reload command, strict same-Zarr/same-layout contract,
  in-place layer/state preservation, derived-distance recomputation, and
  failure behavior.

## Docs updates

- Document how `Reload artifacts` follows the root bundle, what it refreshes,
  what display state it preserves, and which incompatibilities require restart.

## Changelog update

- Record in-process replay-artifact reload without Zarr reopening.
