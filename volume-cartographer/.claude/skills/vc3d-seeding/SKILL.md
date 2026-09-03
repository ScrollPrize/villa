---
name: vc3d-seeding
description: Running batch seeding in VC3D through the bridge (vc3d_seeding_run / vc3d_seeding_expand) — the local paths/seed.json config those commands resolve, why its normal_grid_path does not understand remote stores, and where the source point collection comes from. Load before any vc3d_seeding_* call.
---

Assumes `vc3d-bridge-session`.

`seeding.run` / `seeding.expand` spawn `vc_grow_seg_from_seed` once per source
point or expansion iteration — the same tool behind
`vc3d_grow_patch_from_seed`. Both share one `seeding` job source. Preview,
cast, reset, and path analysis are synchronous over the widget's existing
state.

## Workflow

1. Open/select the intended volume and attach the required normal-grid
   representation.
2. Commit verified source points and retain their collection identity.
3. Enable winding annotation mode only if the chosen panel workflow requires
   it; record the prior state.
4. Call `vc3d_seeding_preview_rays`, inspect returned state and visual evidence,
   then `vc3d_seeding_cast_rays` only when the preview is plausible.
5. Launch `vc3d_seeding_run(wait=true)` or
   `vc3d_seeding_expand(wait=true)`. Both share the `seeding` job slot.
6. Use `vc3d_seeding_analyze_paths` to inspect the current generated paths.
7. `vc3d_seeding_cancel` requests cancellation; confirm the terminal state.
8. `vc3d_seeding_reset_points` is destructive widget-state cleanup. Preserve
   required point/evidence state first and restore the annotation mode.

Ray preview can create a session point collection such as `ray_preview`; it is
not necessarily view-only. List point collections before and after preview,
retain the exact new collection id, and remove only that verified id when the
test must leave point state unchanged. Never use a global point clear as generic
preview cleanup.

A successful preview/cast does not prove a batch produced a surface. Verify the
terminal job, output paths, segment list, and generated geometry.

## The RPCs take no parameters at all

Seven of the eight `seeding.*` methods — `run`, `expand`, `analyze_paths`,
`cast_rays`, `preview_rays`, `reset_points`, `cancel` — accept **zero
parameters**. Only `set_winding_annotation_mode` takes one.

They fire the Seeding widget's *current* state. You cannot set the source
collection, parallel process count, OMP threads, intensity threshold, peak
detection window, max size, or expansion iterations from the bridge. The
substantive configuration is the on-disk `seed.json` / `expand.json` below, and
everything else is whatever a human last left in the panel.

Say so when reporting a seeding run: you drove it, you did not configure it.

Two more things bite, and both are about the config VC3D resolves rather than
the points you commit.

## `paths` / `seed.json` resolution is local-only

It is relative to VC3D's own working directory when the open project has no
segmentations yet — the common case for a freshly attached catalog sample. If
neither exists there you get a clean `-32007` (`data.kind:"file"`); nothing in
`catalog.open_sample` sets it up for you.

For a remote-only project, create a real `paths/` directory and a `seed.json`
(or `expand.json`) yourself in whatever directory VC3D was launched from.
Schema: `{"cache_root", "thread_limit", "normal_grid_path", "min_area_cm",
"generations"}`.

## `normal_grid_path` is a static local path

It has no awareness of remote or streaming normal-grid stores. This is a real,
currently-open gap — unlike `vc3d_grow_patch_from_seed`, which resolves normal
grids dynamically and does support remote stores.

To seed a remote volume, point `normal_grid_path` at the already-fetched local
cache directory for that sample's normal grid:

```
~/.VC3D/remote_cache/normal_grids/<sampleId>/<volumeId>/L<level>-<hash>/
```

`catalog.open_sample` creates it when you pass a `normal_grids`-kind entry in
`representationRefs`. Get the exact hash-suffixed directory name from
`vc3d_get_state` or the catalog attach result — do not guess it.

The volume argument itself (the CT data source) does resolve correctly for
remote volumes; that part needs no workaround.

## The source point collection is implicit

`seeding.run` uses whatever collection `points.commit` last created — it
becomes the widget's combo-box selection. Commit real points obtained from
`vc3d_get_cursor_point` after a screenshot, not fabricated coordinates.
