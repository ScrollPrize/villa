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
(`tool` / `growth` / `lasagna` / `atlas` / `catalog` / `flatten`). Starting a
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
call `vc3d_push_pull_start`, wait, then `vc3d_push_pull_stop`.

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
to the right tab/workspace (or activate that pane) before capturing.

## 9. Seeding batch runs (`vc3d_seeding_run`/`_expand`) need a real local `paths`/`seed.json` — even on a remote volume

`seeding.run`/`seeding.expand` spawn `vc_grow_seg_from_seed` once per source
point / expansion iteration, same underlying tool as `vc3d_grow_patch_from_seed`.
Two things that will bite you, both about the config VC3D resolves, not the
points you commit:

- **`paths`/`seed.json` (or `expand.json`) resolution is local-only, and is
  relative to VC3D's own working directory when the open project has no
  segmentations yet** (the common case for a freshly-attached catalog sample).
  If neither exists there, you get a clean `-32007` (`data.kind:"file"`) —
  not a crash, but also not something `catalog.open_sample` sets up for you.
  For a remote-only project you must create a real `paths/` directory and a
  `seed.json` (schema: `{"cache_root", "thread_limit", "normal_grid_path",
  "min_area_cm", "generations"}`) yourself, in whatever directory VC3D was
  launched from.
- **`normal_grid_path` inside that file is a static local path, with no
  awareness of remote/streaming normal-grid stores** — this is a real,
  currently-open gap (unlike `vc3d_grow_patch_from_seed`, which resolves
  normal grids dynamically and does support remote stores). To seed a remote
  volume, point `normal_grid_path` at the already-fetched local cache dir for
  that sample's normal grid: `~/.VC3D/remote_cache/normal_grids/<sampleId>/
  <volumeId>/L<level>-<hash>/` (created by `catalog.open_sample` when you pass
  a `normal_grids`-kind `representationRefs` entry — check `vc3d_get_state` or
  the catalog attach result for the exact hash-suffixed directory name).
- The volume argument itself (the actual CT data source) *does* now resolve
  correctly for remote volumes (`Volume::remoteLocator()`, issue #1188) — you
  do not need to work around that part.
- `seeding.run`'s source point collection is whatever `points.commit` last
  created (it becomes the widget's combo-box selection) — commit real points
  from `canvas.get_cursor_volume_point` after a screenshot, not fabricated
  coordinates.
