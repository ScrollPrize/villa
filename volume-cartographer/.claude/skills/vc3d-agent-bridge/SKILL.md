---
name: vc3d-agent-bridge
description: Cheat sheet for driving a running VC3D instance via the vc3d-bridge MCP tools (VC3D Agent Bridge). Use before or while calling any vc3d_* / vc3d-bridge MCP tool — covers job concurrency rules, editing-session preconditions, and footguns that span multiple tool calls, none of which are captured by any single tool's own docstring.
---

This is a cross-call cheat sheet, not a tool reference. Per-tool params/return
shapes are already well documented in each `vc3d_*` tool's own MCP
description — read those for call-level detail. For wire-protocol/implementer
detail, see `apps/VC3D/agent_bridge/SPEC.md`. This skill only covers the
gotchas that only show up across *multiple* calls.

## 1. Always call `vc3d_get_state` first

Before doing anything else, see what's actually loaded: volume package,
current volume, active segment, viewers, editing mode, running job. Don't
assume state from a previous turn — another process (or the human) may have
changed it.

## 2. Jobs are tagged by `source`, and concurrency is per-source

Long-running operations return a `jobId` and are tagged with a `source`
(`tool` / `growth` / `lasagna` / `catalog` / `flatten` / `import` /
`export` / ...). Starting a
*second* job of the **same** source while one is active fails with
`-32004 JOB_RUNNING`. Jobs of *different* sources run fine concurrently — a
Lasagna optimization and a render job can both be in flight at once. Poll
`vc3d_job_status` (optionally filtered by source) rather than assuming
serialization.

## 3. Footguns

- `vc3d_grow_segment(method="manual_add")` is rejected outright — use
  `vc3d_manual_add_begin` / `vc3d_manual_add_finish` instead.
- `vc3d_open_catalog_sample` treats the call itself as consent to replace the
  currently open project — there is no confirmation step, and it will
  discard unsaved local work silently. Don't call it casually if the current
  session has state worth keeping.
- `vc3d_corrections_set_point_mode(active=true)` turns every subsequent plain
  click on the surface into a correction point until you explicitly turn it
  back off — it does not auto-clear on release.
- A correction drag longer than ~1 voxel silently kicks off a `source:"growth"`
  job (the corrections solver runs asynchronously). Poll `vc3d_job_status`
  before issuing further editing calls, or you'll race it.

## 4. The editing-session precondition triad

Several tools (`vc3d_manual_add_begin`, `vc3d_corrections_set_point_mode`,
`vc3d_push_pull_start`, and similar editing RPCs) silently require ALL of:

1. Segmentation editing enabled (`vc3d_enable_editing(true)` first — else you get an error).
2. An active edit session on the target segment (else `-32007`, `data.kind:"session"`).
3. No `source:"growth"` job currently running (else `-32004`).

This triad is repeated piecemeal across individual tool docstrings — treat it
as one precondition to check up front rather than discovering each piece
one error at a time.

## 5. Async correlation: jobs vs. plain slow replies

Not everything slow is a job. A `jobId`-bearing result means poll
`vc3d_job_status`. Some Lasagna calls (`vc3d_lasagna_list_datasets`,
`vc3d_lasagna_jobs`, `vc3d_lasagna_ensure_service` in external-host mode) have
no `jobId` at all — the RPC reply itself just arrives late (up to ~10-15s).
The MCP tool call simply blocks; there's nothing to poll and nothing wrong if
it takes a few seconds.

## 6. Position-then-act pattern

`vc3d_push_pull_start` acts at the last recorded pointer position, not a
position you pass it. Use `vc3d_drag` with `button="none"` to hover-position
the cursor first (no press/release, just interpolated move events), then
call `vc3d_push_pull_start`, wait, then `vc3d_push_pull_stop`. The same rule
applies to keyboard holds: hover first, then `vc3d_key(key="A", hold_ms=...)`.
A `vc3d_key(action="press")` stays down until a matching `action="release"`
or until its `hold_ms` (10 s by default) runs out, and keys are also released
if the MCP connection drops; `vc3d_get_state` lists held keys in `heldKeys`.
A key that opens a dialog or menu has it answered as Escape would (listed in
`dismissed`), and so is any dialog a person at the window opens during that
call. Keys and wheel turns need a visible viewer: switch to its workspace
first. Prefer the typed tool whenever one exists.

## 7. Fiber tracing (`vc3d_fiber_launch` + a fiber workspace pane)

- Not every catalog sample has fiber data: `vc3d_fiber_launch` needs a
  resolvable Lasagna dataset for the active volume, and some samples (e.g.
  PHercParis4, verified live) have none published at all — check
  `vc3d_describe_catalog_sample` for a `lasagna`-kind representation before
  opening, or you'll hit `-32005` no matter what you try.
- On a fiber workspace pane, plain `vc3d_click` adds a control point;
  `vc3d_shift_click` does **not** — it triggers a "predicted snap point"
  gesture instead. This inverts the usual shift-click-to-place-a-point
  convention that holds everywhere else in the bridge.
- The pane's `"v<N>"` viewer ids are **not stable across edits** — each
  control point placed rebuilds the workspace and reassigns ids. Re-call
  `vc3d_get_state` before targeting a pane after any edit, or you'll get
  `-32002` on a now-stale id.
- `vc3d_fiber_launch`'s `replace_owning` defaults to `true` and silently
  discards the currently open fiber's unsaved control points. Pass
  `replace_owning=false` to trace several fibers before one combined
  `vc3d_fiber_save`.

## 8. `vc3d_screenshot` fails loudly on a hidden target (by design)

If the target viewer is on a non-frontmost tab (e.g. a fiber/lasagna
workspace pane while the main tab is active, or vice versa), `vc3d_screenshot`
now fails `-32009` instead of silently returning a near-zero-size image — this
used to return a genuinely degenerate (e.g. 15×50px) but "successful" PNG,
which cost real debugging time before anyone noticed. If you hit this, switch
to the right tab/workspace (or activate that pane) before capturing, or pass
`reveal=True` to switch and show it for that one capture and put the GUI back
afterwards. A revealed viewer may still be loading tiles; for a reliable image
of a viewer, switch first and capture on a later call.

## 9. Seeding batch runs (`vc3d_seeding_run`/`_expand`) need a writable surface store and a local `seed.json` — even on a remote volume

`seeding.run`/`seeding.expand` spawn `vc_grow_seg_from_seed` once per source
point / expansion iteration in store mode, same underlying tool as
`vc3d_grow_surface_from_seed`. Two things that will bite you, both about the
config VC3D resolves, not the points you commit:

- **New surfaces go to the surfaces' store (the project's `.vcobj`, unless
  another store is chosen for surfaces: `state.get` → `routes.surface`), and
  `seed.json` (or `expand.json`) is read from the project directory** (the
  directory holding the project's `.vcobj`). Without a writable store, or
  without that file, you get a clean error — not a crash, but also not
  something `catalog.open_sample` sets up for you (a sample's project lives in
  VC3D's projects directory, `open_data/<sample>.vcobj`). Write `seed.json`
  (schema: `{"cache_root", "thread_limit", "normal_grid_path", "min_area_cm",
  "generations"}`) next to the project file yourself.
- **`normal_grid_path` inside that file is a static local path, with no
  awareness of remote/streaming normal-grid stores** — this is a real,
  currently-open gap (unlike `vc3d_grow_surface_from_seed`, which resolves
  normal grids dynamically and does support remote stores). To seed a remote
  volume, point `normal_grid_path` at the already-fetched local cache dir for
  that sample's normal grid: its directory under the remote cache root (created by `catalog.open_sample`
  when you pass a `normal_grids`-kind `representationRefs` entry — check
  `vc3d_get_state` or the catalog attach result for the exact directory).
- The volume argument itself (the actual CT data source) *does* now resolve
  correctly for remote volumes (`Volume::remoteLocator()`, issue #1188) — you
  do not need to work around that part.
- `seeding.run`'s source point collection is whatever `points.commit` last
  created (it becomes the widget's combo-box selection) — commit real points
  from `canvas.get_cursor_volume_point` after a screenshot, not fabricated
  coordinates.

## 10. Objects live in stores: import, then route, then use

Surfaces, fibers and point collections are objects of the project's `.vcobj`
stores. `vc3d_get_state` lists the store references (`stores`, each with
`writable`) and the store each kind is shown from and edited in (`routes`).

- **Load a folder of tifxyz segments**:
  `vc3d_import_objects(kind="surface", path=<folder>, bind={"scale": 1}, wait=True)`,
  then `vc3d_list_segments()` → `vc3d_activate_segment(<uuid>)` →
  `vc3d_screenshot()`. The job result's `uuids` are the new segment ids; no
  need to diff listings.
- **Import never changes routing.** It writes to the kind's routed store
  unless `store` names another one; objects imported into a store that is not
  routed for their kind do not show up in `vc3d_list_segments` /
  `vc3d_fiber_list` / `vc3d_list_points` until you route it
  (`vc3d_route_store(kind, store_id=<result.store>)`). Edits of an existing
  object always stay in the store that holds it — routing only picks what is
  shown and where new objects go. Check `routes` and each row's `store`.
- **`bind.scale` has no default.** It is stored units per *current-volume*
  voxel (1 when the segments were traced on this volume at this resolution).
  Without `bind`, sources keep the coordinates they declare; a source that
  declares none stays unbound and is not shown on any volume.
- **A finished import can still have imported nothing.** The job succeeds
  once the whole source was traversed; read `problems` and `existing`.
  `fatalError` (job failed) means the store itself failed partway.
- **Legacy `.volpkg` data**: there is no volpkg open. Create a project on its
  volume, then import its `paths/` with a namespace:
  `vc3d_create_project(path=<p.vcproj>, volume=<volpkg>/volumes/<v>)`,
  `vc3d_open_volume(path=<p.vcproj>)`,
  `vc3d_import_objects(kind="surface", path=<volpkg>/paths, uuid_namespace=<one fresh UUID per volpkg>, bind={"scale": 1}, wait=True)`.
  Record the namespace and reuse it for every later import of that volpkg;
  a retry then reports `existing` instead of creating duplicates. Without a
  namespace, UUID-less VC3D-era sources are listed as problems.
- **Export by UUID**: `vc3d_export_object(uuid, path)` takes a segment id or
  the `uuid` from `vc3d_fiber_list` / `vc3d_list_points` (next to the numeric
  ids). The parent directory must exist; pending edits are saved first.
- **Detach is not delete.** `vc3d_detach_store` and
  `vc3d_detach_project_entry` only drop the project's reference; files and
  stored objects stay. `vc3d_remove_umbilicus` is the exception: it retires
  the umbilicus *in the store*, for every project sharing it — use
  `vc3d_route_store(kind="umbilicus", store_id=None)` to hide it for this
  project only.
- **Store switches save first.** Route, detach and Save As save pending
  edits before switching and fail `-32004` (changing nothing) while growth
  or a line optimization runs; finish or wait for those first.


## 10. Ink labels (`vc3d_ink_labels_*`)

- Call `vc3d_ink_labels_open` after activating the segment; every other ink
  label tool fails `-32007` (`data.kind:"ink_label_session"`) until then. Open
  never prompts: an autosave newer than the last save fails `-32602`
  (`data.kind:"autosave_pending"`) under the default `recover="fail"`, so
  decide between `"restore"` and `"discard"` and call again.
- Coordinates are label pixels by default: the segment's `vc_render_tifxyz`
  canvas at full resolution, constants in `state.canvas`. Use
  `space="volume"` for voxels on the segment (they must lie on the surface the
  segmentation viewer shows). A `-32009` with `data.kind:"viewer"` means the
  segmentation viewer is not showing the labeled segment flattened: call
  `vc3d_activate_segment` for it, or work in label space.
- The user may be labeling at the same time. Pass `expect={"segment",
  "serial"}` from your last `vc3d_ink_labels_state` so a paint against a moved
  session is refused instead of applied, and note that undo stops at the
  user's edits unless `force=true`.
- To check unsaved work, use `vc3d_ink_labels_read` (a PNG of a rectangle),
  not `vc3d_ink_labels_save`, which overwrites the trainer TIFFs. Save blocks
  until the write lands and returns `saved:false` when nothing changed.
- To edit a region by hand, read it with `format="values"`, change the image,
  and paint it back with `mask={"path", "origin", "format": "values"}`: one
  undo entry with exactly those values.
- `vc3d_ink_labels_fill`, `_suggest` and `_cleanup` answer when done.
  Suggestions need the single-channel prediction in `state.prediction`
  (`labelPxPerPixelX/Y` give its pixel size in label px); accept them with
  `vc3d_ink_labels_accept_suggestions(ids=...)` or drop them with
  `_discard_suggestions`. `vc3d_ink_labels_import`, `_versions`,
  `_dataset_entry`, `_set_tool_state` and `_close` cover the rest of the dock.
