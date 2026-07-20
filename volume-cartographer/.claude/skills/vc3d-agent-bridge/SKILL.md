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
