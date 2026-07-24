# VC3D Agent Bridge Protocol

Status: current protocol reference.

The bridge is an opt-in JSON-RPC 2.0 server embedded in VC3D. A separate
FastMCP process translates agent-facing tools into bridge calls. The bridge
executes the same application operations as the UI while avoiding modal
dialogs and nested event loops.

## Sources of truth

The compiled `AgentBridgeMethod` descriptors are authoritative for:

- registered RPC names;
- parameter types, required fields, defaults, enums, and rejecting bounds;
- documented JSON-RPC error codes;
- RPC-to-MCP mappings and parameter renames.

`rpc.describe` exposes those descriptors over the live socket.
`rpc_description.json` is a generated snapshot of that response. The offscreen
smoke test byte-compares the compiled response with the snapshot, and the host
MCP tests compare FastMCP schemas and the mapping table below with the same
snapshot.

This document owns the parts that are not mechanical schemas: transport
behavior, lifecycle rules, cross-field semantics, mutation ordering, headless
behavior, and known limitations.

After changing a descriptor, rebuild VC3D and regenerate the snapshot in the
bridge container:

```sh
docker exec vc3d-bridge bash -lc 'cd /work && QT_QPA_PLATFORM=offscreen python3 \
  apps/VC3D/agent_bridge/test/smoke_offscreen.py \
  --vc3d build/ci-release-gcc/bin/VC3D --update-description-snapshot'
```

## Activation and discovery

The bridge is disabled unless VC3D receives one of these options:

```text
--agent-bridge
--agent-bridge-name <name>
```

`--agent-bridge` uses `vc3d-agent-<pid>`. An explicit name is used verbatim.
The endpoint is restricted to the current user because it grants full control
of the running application.

VC3D publishes a discovery record under `~/.vc3d/agent_bridge/` after a
successful listen and removes its own record on clean shutdown. The record
contains the process id, server name, resolved socket path, and start time.
Clients discard malformed records and records whose process is no longer
alive, then choose the newest live entry.

An existing live socket is never removed. On Unix, listen recovery is guarded
by a name-derived advisory lock. VC3D probes a failed name first and removes it
only when no live peer accepts a connection. A requested bridge that still
cannot listen exits with code 2.

On success VC3D prints one machine-readable line:

```text
VC3D-AGENT-BRIDGE: listening name=<serverName> path=<fullServerName>
```

## Wire protocol

- UTF-8, newline-delimited JSON; one JSON-RPC message per line.
- Requests must use `"jsonrpc":"2.0"` and a non-empty string method.
- `params` may be absent, null, or an object. Arrays and scalars are rejected.
- Batch arrays are not supported.
- A framed request and an unterminated receive buffer are each limited to
  1 MiB. An oversized client receives a best-effort invalid-request response
  and is disconnected without affecting other clients.
- Multiple clients may connect. Requests are dispatched in frame order;
  deferred responses may complete out of order and are correlated by `id`.
- Handlers run serially on the Qt GUI thread.
- Server notifications have no `id` and are broadcast to all connected
  clients.
- A notification request never receives a response, including on error.

Handlers must not open dialogs or run nested event loops. Operations that may
outlive one GUI-thread turn return jobs or use a deferred response completed by
an application signal.

## Method descriptions

`rpc.describe` accepts an optional string `prefix`. Its result contains:

- `methods`: matching descriptor objects;
- `undocumented`: registered handlers without descriptors;
- `coverage.described` and `coverage.registered`;
- `coverage.complete`.

The unfiltered response must report complete coverage. Registration rejects
duplicate names and malformed descriptors at startup.

Descriptors validate mechanical inputs before handlers run. Handlers perform
semantic validation, resolve live application state, and apply mutations only
after all fallible preconditions have passed.

## Common values

### Coordinates

Volume coordinates are full-resolution voxel coordinates:

```json
{"x": 1.0, "y": 2.0, "z": 3.0}
```

Scene coordinates are viewer-local `{"x": number, "y": number}` values.
Conversions use the selected viewer's `volumeToScene` and `sceneToVolume`
operations. Values narrowed to floats must remain finite and within float
range. Volume-space operations reject points outside the current volume.

Canvas operations round-trip converted points and reject a volume point that
does not lie on the selected viewer's current plane or surface.

### Viewer selection

VC3D assigns every live base viewer a stable process-local id (`v1`, `v2`, …).
Ids are never reused during that process. A viewer parameter resolves in this
order:

1. exact bridge viewer id;
2. unique `surfName`;
3. the `segmentation` surface when the parameter is optional and omitted.

An ambiguous surface name returns the matching viewer candidates. Canvas
methods additionally require a chunked-volume viewer.

### Mouse input

Buttons are `left`, `right`, `middle`, or, where declared, `none`. Modifiers
are arrays containing `shift`, `ctrl`, `alt`, `meta`, or `keypad`. The bridge
delivers the corresponding Qt input to the real viewer so existing signal
wiring and tool behavior remain authoritative.

### Identifiers

Segment, volume, sample, and job identifiers are strings. Fiber identifiers
are serialized as decimal strings because their native type is `uint64_t`.
Point and collection ids use JSON-safe positive integers. Methods that accept
both a collection id and name reject ambiguous selectors.

### Errors

Errors use the JSON-RPC error object:

```json
{"code": -32602, "message": "…", "data": {"param": "…"}}
```

| Code | Name | Meaning |
|---:|---|---|
| -32700 | Parse error | Malformed JSON. |
| -32600 | Invalid Request | Invalid envelope, unsupported batch, or oversized request. |
| -32601 | Method not found | No registered method. |
| -32602 | Invalid params | Missing, mistyped, conflicting, or out-of-range input. |
| -32000 | NO_VOLPKG | No volume package is open. |
| -32001 | NO_VOLUME | No current volume is selected. |
| -32002 | INVALID_VIEWER | Viewer selection failed or was ambiguous. |
| -32003 | INVALID_COORDINATES | Point is outside the volume or selected view. |
| -32004 | JOB_RUNNING | The relevant job source is busy. |
| -32005 | JOB_FAILED | Launch, persistence, download, or deferred operation failed. |
| -32006 | TOOL_NOT_FOUND | A required external executable is unavailable. |
| -32007 | NOT_FOUND | Requested application object does not exist. |
| -32008 | EDITING_REQUIRED | Segmentation editing is not enabled. |
| -32009 | UNSUPPORTED | The selected target or operation is unsupported. |
| -32010 | INTERNAL | An unexpected application or bridge failure occurred. |

`data.param` identifies invalid input. Lookup failures use `data.kind` and,
where useful, `data.id`. Internal and launch failures use `data.detail`.

## Jobs and deferred responses

Job ids are monotonically increasing `job-<n>` strings. At most one active job
per source is tracked:

| Source | Authority |
|---|---|
| `tool` | `CommandLineToolRunner` lifecycle |
| `growth` | segmentation growth lifecycle |
| `lasagna` | Lasagna optimization lifecycle |
| `atlas` | bridge-started atlas search lifecycle |
| `catalog` | Open Data sample open lifecycle |
| `flatten` | SLIM, ABF, and straighten lifecycle |
| `seeding` | bridge-started run or expand batch |
| `autosave` | explicit dirty-segment save |

Application work started outside the bridge is represented when its lifecycle
is observable, but only bridge-started atlas and seeding operations are
registered as jobs. A source retains its eight most recent terminal records.

`job.status` accepts `jobId`, `source`, or neither. With neither it returns the
most recently started job across all sources. Its record contains:

```text
jobId, kind, source, label, state, message, outputPath, externalId,
consoleTail, progressHistory, startedAtMs, finishedAtMs, result
```

`state` is `running`, `succeeded`, or `failed`. `consoleTail` and
`progressHistory` are bounded. `outputPath`, `externalId`, and `finishedAtMs`
are null when unavailable. `result` is an operation-specific object when a
terminal job has structured output, otherwise null.

### Progress

`job.progress` is a notification. Every update contains `jobId`, `kind`,
`source`, a monotonically increasing per-job `seq`, and `phase`. `message` is
present when the job has non-empty text. Terminal updates also contain
`success` and `result`, plus `outputPath` when one is available.

Each job retains the last 64 notifications. Output text is rate-limited to ten
notifications per second and may be coalesced. Delivery is ordered and
best-effort for a live connection; it is not durable or exactly-once.
`job.status` is authoritative for terminal state.

### Cancellation

`job.cancel` selects a running job by id or source. Cancellation dispatches to
the operation's real cancellation authority when one exists. Unsupported
cancellation returns an error without changing the job. Cancelling an MCP wait
does not cancel the underlying VC3D job.

### Deferred calls

Some request/response methods wait for an existing asynchronous application
signal without becoming jobs. The bridge stores the caller, arms a bounded
timer, and completes the response exactly once from the signal or timeout.
Disconnecting a client discards its pending replies without affecting other
clients or application work.

## Headless application behavior

An explicit RPC is treated as consent for the requested mutation, never as
consent for unrelated prompts. Bridge calls use dialog-free application cores:

- project and catalog opens never ask whether to replace the current project;
- segment, fiber, atlas, rendering, tracing, and flatten operations report
  failures through JSON-RPC or job state;
- interactive menu actions remain responsible for file pickers, confirmation
  prompts, status widgets, and message boxes;
- shared operation cores retain the same validation, persistence, progress,
  and UI-synchronization behavior for both callers.

Bridge handlers suppress line-annotation error dialogs for the duration of
dispatch. Other reusable cores take explicit non-interactive options or
optional status/error sinks. A bridge handler must never call `exec()` on a
dialog.

## Domain semantics

### Session, volumes, and catalog

- `ping` reports process, application, and protocol identity.
- `state.get` is a non-mutating snapshot of the current package, volume,
  active segment, editing modes, viewers, point state, and active jobs.
- `project.create` writes a new `.volpkg.json` referencing one local zarr
  volume or remote `.zarr` URL. It does not change the current session;
  remote availability is checked by `volume.open` when the new project is
  opened.
- `volume.open` opens a local project and may select a volume. Failed opens do
  not discard the current project.
- `volume.select` is a no-op success when the requested volume is already
  current; otherwise it uses the same state and selector synchronization as
  the GUI.
- `catalog.list_samples` and `catalog.describe_sample` use the cached Open Data
  manifest unless `refresh` is true. A refresh is a deferred response.
- `catalog.open_sample` validates the entire optional resource selection before
  starting work. It is a `catalog` job and refuses concurrent interactive or
  bridge catalog opens.
- Resource selections may filter volume ids, derived-representation refs, and
  representation kinds. An explicit empty volume selection is invalid.

### Segments and review state

- `segments.list` reports package segments, whether they are loaded, and the
  active segment.
- `segments.fetch` materializes an Open Data placeholder. Already-materialized
  segments return synchronously; downloads return a `catalog` job.
- `segments.activate` validates before changing selection. Activating the
  current segment is a no-op success. Placeholder materialization remains an
  explicit fetch operation.
- Delete and rename use dialog-free controller cores and preserve editing and
  active-selection invariants.
- `segments.review` derives review status and optional geometric/tag filters
  without changing the current selection.

### Canvas and viewers

- `canvas.click`, `canvas.shift_click`, and `canvas.drag` deliver real viewer
  input. A buttonless drag moves the cursor without starting a gesture.
- `viewer.rotate` accepts normalized plane spellings and supports relative or
  absolute angles. Axis-aligned slices must be enabled.
- Wrap annotation mode, commit, and undo target the selected chunked viewer and
  use the same preview and point-collection state as keyboard interaction.
- Render-setting updates validate and normalize every supplied value before
  applying any setter. Opacity is clamped to `[0,1]`; non-negative sizes remain
  non-negative; normal-arrow controls are clamped to their GUI ranges.
- Viewer-manager settings remain meaningful with zero viewers. Per-viewer
  toggles are broadcast to every live base viewer and their persisted defaults
  are updated where applicable.
- Overlay updates are atomic. A clear request removes the overlay; explicit
  volumes must resolve in the open package. Intersection sets always include
  the segmentation surface and are not applied to the segmentation viewer
  itself.

### Segmentation editing

- Editing must have an active materialized segment.
- `segmentation.grow` accepts tracer, corrections, and patch-tracer growth.
  Manual add is an editing mode, not a growth method.
- Explicit save returns `jobId:null` when nothing is dirty; otherwise it
  creates an `autosave` job completed by the real save signal.
- Manual-add begin/finish, line mode, interpolation, undo, correction point
  mode, push/pull, and synthetic canvas input all use the active
  `SegmentationModule` session.
- `segmentation.grow_patch_from_seed`, tracing, rendering, and reoptimization
  use the existing command-runner lifecycle and never open the interactive
  console or parameter dialogs.

### Points

Point collection mutations resolve every selector and validate all points
before changing the collection. Bulk operations are all-or-nothing. Winding
values may be finite floats or null where the method declares clearing.
Metadata and tags are idempotent setters. Save/load methods use explicit paths
and never open file pickers.

### Lasagna, atlas, and fibers

- Lasagna service and job queries use deferred responses from the service
  manager. Optimization is tracked as a `lasagna` job.
- Atlas open, remap, result selection, and candidate optimization use
  dialog-free `CWindow` operations. Only bridge-started searches become
  `atlas` jobs.
- Fiber ids are strings on the wire. Bulk deletion validates every id before
  removing any fiber. Import/export paths are explicit. Save waits for the
  controller's persistence completion signal.
- Fiber launch and create-atlas operations suppress dataset pickers and atlas
  rebuild prompts; missing prerequisites become ordinary errors.

### Seeding, rendering, flattening, and mesh operations

- Seeding run and expand share one `seeding` source. Preview, cast, reset, and
  path analysis are synchronous requests over the existing widget state.
- `render.tifxyz`, trace, reoptimize, and alpha-composition refinement use the
  shared `tool` source.
- SLIM, ABF, and straighten share the `flatten` source and application
  lifecycle.
- Crop and area recalculation are synchronous. Mask generation and append use
  deferred completion from the in-process renderer.
- Result paths are returned only after successful launch or completion.

## MCP behavior

The MCP server connects to an explicit socket, discovers the newest live
registry entry, or launches VC3D and parses its handshake. It rejects a bridge
with an incompatible protocol version.

Tool wrappers are intentionally thin. They translate Python snake_case names,
remove omitted optional values, preserve bridge errors, and otherwise return
the bridge result.

`wait` is an MCP-only convenience on job-returning tools. A wait:

1. subscribes before its first status read;
2. replays the server's bounded progress tail;
3. merges live updates by sequence;
4. polls status as a delivery fallback;
5. returns the authoritative terminal record.

Progress reporting is observational. Unsupported or failing MCP contexts
disable further reporting for that wait. Buffered replay shares one bounded
reporting-time budget; live messages use the same per-report cap. Reporting
failure never changes the job result, while task cancellation still
propagates. Waits cap at 30 minutes and return `waitTimedOut:true` with the
still-running job id.

`vc3d_wait_job` applies the same behavior to an existing job.

Without `file_path`, `vc3d_screenshot` decodes inline PNG data into FastMCP
image content. With `file_path`, it returns the bridge's file result and does
not include inline image data.

## MCP tool map

The following rows are checked against the descriptor snapshot. `rpc.describe`
is the sole bridge method without an MCP tool.

| MCP tool | RPC method | Agent-facing purpose |
|---|---|---|
| `vc3d_activate_segment` | `segments.activate` | Make a segment the active editing target (the programmatic equivalent of clicking it in the segment list). Required before `vc3d_enable_editing` / `vc3d_grow_segment` and after any segment switch. |
| `vc3d_add_point_collection` | `points.add_collection` |
| `vc3d_append_segment_mask` | `segment.append_mask` | Append a volume-image layer to a segment's mask. Blocks until the render completes (130 s client timeout); requires a current volume. |
| `vc3d_apply_anchor_offset` | `points.apply_anchor_offset` |
| `vc3d_atlas_open_result` | `atlas.open_result` |
| `vc3d_atlas_open` | `atlas.open` |
| `vc3d_atlas_optimize_snap_candidates` | `atlas.optimize_snap_candidates` |
| `vc3d_atlas_remap` | `atlas.remap` |
| `vc3d_atlas_search_cancel` | `atlas.search_cancel` |
| `vc3d_atlas_search_results` | `atlas.search_results` |
| `vc3d_atlas_search_start` | `atlas.search_start` |
| `vc3d_atlas_status` | `atlas.status` |
| `vc3d_auto_fill_windings` | `points.auto_fill_windings` |
| `vc3d_cancel_job` | `job.cancel` | Request cancellation of a running job by `jobId` (or by `source`). Only `tool`/`atlas`/`seeding`/`lasagna` jobs are cancellable; `growth`/`flatten`/`catalog`/`autosave` return an error. Request-only — poll `vc3d_job_status` for the terminal state. |
| `vc3d_center_viewer` | `viewer.center_on_point` | Center a viewer pane on a 3D volume point. |
| `vc3d_clear_all_points` | `points.clear_all` |
| `vc3d_clear_point_collection` | `points.clear_collection` |
| `vc3d_click` | `canvas.click` | Synthesize a mouse click in a viewer at a volume-space (or scene-space) position, with button and modifiers (e.g. `{"modifiers": ["shift"]}` to place a point / set focus). |
| `vc3d_commit_points` | `points.commit` | Add annotation points (volume space) to a named collection, optionally with a winding annotation. |
| `vc3d_commit_wrap_annotation` | `wrap_annotation.commit` | Commit the seeded same-wrap annotation preview into the point collection (the tutorial's shift+E). Requires the mode enabled. |
| `vc3d_corrections_set_point_mode` | `segmentation.corrections.set_point_mode` |
| `vc3d_create_project` | `project.create` | Create a `.volpkg.json` that references one local zarr volume or remote `.zarr` URL without opening it. |
| `vc3d_crop_segment_bounds` | `segment.crop_bounds` | Crop a segment's surface grid to its tightest valid bounds. Synchronous. |
| `vc3d_delete_segment` | `segments.delete` | Delete a segment from disk. **Irreversible** — requires `confirm=true`. Fails while segmentation editing is enabled; deleting the active segment is allowed. |
| `vc3d_describe_catalog_sample` | `catalog.describe_sample` |
| `vc3d_drag` | `canvas.drag` |
| `vc3d_enable_editing` | `segmentation.enable_editing` | Turn segmentation editing mode on/off for the active segment. |
| `vc3d_fetch_segment` | `segments.fetch` | Download ("materialize") an open-data placeholder segment so it can be activated/edited. Sync if already materialized; else a `"catalog"` job (`wait` defaults true). |
| `vc3d_fiber_create_atlas` | `fiber.create_atlas` |
| `vc3d_fiber_delete` | `fiber.delete` |
| `vc3d_fiber_export` | `fiber.export` |
| `vc3d_fiber_import` | `fiber.import` |
| `vc3d_fiber_launch` | `fiber.launch` |
| `vc3d_fiber_list` | `fiber.list` |
| `vc3d_fiber_open` | `fiber.open` |
| `vc3d_fiber_save` | `fiber.save` |
| `vc3d_fiber_set_follow` | `fiber.set_follow` |
| `vc3d_fiber_set_tag` | `fiber.set_tag` |
| `vc3d_flatten_abf` | `flatten.abf` |
| `vc3d_flatten_slim` | `flatten.slim` |
| `vc3d_flatten_straighten` | `flatten.straighten` |
| `vc3d_generate_segment_mask` | `segment.generate_mask` | Render a segment's binary mask. Blocks until the render completes (130 s client timeout); no job to poll. |
| `vc3d_get_cursor_point` | `canvas.get_cursor_volume_point` | Resolve a viewer scene position (or the current cursor) to a 3D volume point + surface normal. |
| `vc3d_get_overlay` | `viewer.get_overlay` | Read the current overlay-volume settings. |
| `vc3d_get_render_settings` | `viewer.get_render_settings` | Read the shared viewer render/overlay settings (intersection lines, overlay opacity, surface normals, direction hints, highlighted surfaces). |
| `vc3d_get_state` | `state.get` | Snapshot of VC3D: open volume package, current volume, active segment, viewers (ids/names), editing mode, running job. Call this first. |
| `vc3d_grow_patch_from_seed` | `segmentation.grow_patch_from_seed` | Create a brand-new segment by growing a patch from a 3D seed point (headless GrowPatch). Async: returns a jobId and outputDir. |
| `vc3d_grow_segment` | `segmentation.grow` | Grow the active segmentation surface (method: tracer/corrections/patch_tracer; direction; steps). Async: returns a jobId. |
| `vc3d_job_status` | `job.status` | Poll a job by id (or the latest job): state, message, console tail. |
| `vc3d_lasagna_cancel` | `lasagna.cancel` |
| `vc3d_lasagna_ensure_service` | `lasagna.ensure_service` |
| `vc3d_lasagna_jobs` | `lasagna.jobs` |
| `vc3d_lasagna_list_datasets` | `lasagna.list_datasets` |
| `vc3d_lasagna_repeat_last` | `lasagna.repeat_last` |
| `vc3d_lasagna_select_output` | `lasagna.select_output_segment` |
| `vc3d_lasagna_service_status` | `lasagna.service_status` |
| `vc3d_lasagna_start_optimization` | `lasagna.start_optimization` |
| `vc3d_list_catalog_samples` | `catalog.list_samples` |
| `vc3d_list_overlay_volumes` | `viewer.list_overlay_volumes` | List every volume id in the open package, for picking an overlay volume. |
| `vc3d_list_points` | `points.list` | List point collections and their points. |
| `vc3d_list_segments` | `segments.list` | List segments in the open volume package with loaded/active flags. |
| `vc3d_list_volumes` | `volume.list` | List the volume ids in the open volume package and which one is current, without force-loading any (possibly remote) volume. |
| `vc3d_load_points_json` | `points.load_json` |
| `vc3d_load_points_segment_path` | `points.load_segment_path` |
| `vc3d_manual_add_begin` | `segmentation.manual_add.begin` |
| `vc3d_manual_add_finish` | `segmentation.manual_add.finish` |
| `vc3d_manual_add_set_interpolation` | `segmentation.manual_add.set_interpolation` |
| `vc3d_manual_add_set_line_mode` | `segmentation.manual_add.set_line_mode` |
| `vc3d_manual_add_undo_constraint` | `segmentation.manual_add.undo_constraint` |
| `vc3d_open_catalog_sample` | `catalog.open_sample` | Open an Open Data catalog sample by its manifest sample id. |
| `vc3d_open_volume` | `volume.open` | Open a volume package (.volpkg / .volpkg.json / zarr project) and optionally select a volume id. |
| `vc3d_ping` | `ping` | Check the VC3D bridge is alive; returns pid, app version, and protocol version. |
| `vc3d_push_pull_set_config` | `segmentation.push_pull.set_config` |
| `vc3d_push_pull_start` | `segmentation.push_pull.start` |
| `vc3d_push_pull_stop` | `segmentation.push_pull.stop` |
| `vc3d_recalc_segment_area` | `segment.recalc_area` | Recompute surface area for one or more segments. Synchronous. |
| `vc3d_refine_segment_alpha_comp` | `segment.refine_alpha_comp` | Alpha-composite refinement of a segment. Asynchronous (`source:"tool"`); rejects remote volumes; supports `wait: bool = false` (30-minute cap). |
| `vc3d_remove_point_collection_tag` | `points.remove_collection_tag` |
| `vc3d_remove_point` | `points.remove_point` |
| `vc3d_rename_point_collection` | `points.rename_collection` |
| `vc3d_rename_segment` | `segments.rename` | Rename a segment (id + folder). `new_name` must match `^[a-zA-Z0-9_-]+$`, differ from the current name, and not collide. Fails while editing is enabled. |
| `vc3d_render_tifxyz` | `render.tifxyz` |
| `vc3d_reoptimize_segment` | `segment.reoptimize` | Resume-opt local reoptimization of a segment. Asynchronous (`source:"tool"`); supports `wait: bool = false` (30-minute cap). |
| `vc3d_reset_windings` | `points.reset_windings` |
| `vc3d_review_segments` | `segments.review` | List segments with review-tag state and optional server-side filtering (the programmatic equivalent of the surface panel's review filter checkboxes). |
| `vc3d_rotate_viewer` | `viewer.rotate` | Rotate the "seg xz"/"seg yz" axis-aligned slice plane (middle-drag equivalent). Relative delta by default. |
| `vc3d_run_trace` | `tracer.run_trace` |
| `vc3d_save_points_json` | `points.save_json` |
| `vc3d_save_points_segment_path` | `points.save_segment_path` |
| `vc3d_save_segment` | `segmentation.save` | Force the active segment's pending autosave to disk. Idle no-op (`jobId:null`) when nothing is dirty; else an `"autosave"` job (`wait` defaults true). |
| `vc3d_screenshot` | `screenshot.capture` | Capture a PNG of the whole VC3D window or one viewer pane. Returns the PNG as MCP image content when `file_path` is omitted, or a dict with the on-disk path when `file_path` is set (MCP-layer note below). |
| `vc3d_seeding_analyze_paths` | `seeding.analyze_paths` |
| `vc3d_seeding_cancel` | `seeding.cancel` |
| `vc3d_seeding_cast_rays` | `seeding.cast_rays` |
| `vc3d_seeding_expand` | `seeding.expand` |
| `vc3d_seeding_preview_rays` | `seeding.preview_rays` |
| `vc3d_seeding_reset_points` | `seeding.reset_points` |
| `vc3d_seeding_run` | `seeding.run` |
| `vc3d_seeding_set_winding_annotation_mode` | `seeding.set_winding_annotation_mode` |
| `vc3d_select_volume` | `volume.select` |
| `vc3d_set_auto_fill_mode` | `points.set_auto_fill_mode` |
| `vc3d_set_axis_aligned_slices` | `viewer.set_axis_aligned_slices` | Enable/disable axis-aligned slice mode (checkbox equivalent) — prerequisite for `viewer.rotate`. |
| `vc3d_set_intersects` | `viewer.set_intersects` | Set which surfaces' intersection lines a viewer draws. |
| `vc3d_set_overlay` | `viewer.set_overlay` | Update the overlay-volume settings (volume, colormap, opacity, threshold, window, resolution cap, composite); any subset. |
| `vc3d_set_point_collection_color` | `points.set_collection_color` |
| `vc3d_set_point_collection_metadata` | `points.set_collection_metadata` |
| `vc3d_set_point_collection_tag` | `points.set_collection_tag` |
| `vc3d_set_point_windings_linked` | `points.set_windings_linked` |
| `vc3d_set_render_settings` | `viewer.set_render_settings` | Set any subset of the viewer render/overlay settings; returns the full settings after applying. Viewer-specific toggle fields are no-ops when no viewer is open; highlighted surface ids are retained by the surface panel. |
| `vc3d_set_segment_tag` | `tags.set` |
| `vc3d_set_wrap_annotation_mode` | `wrap_annotation.set_mode` | Enable/disable "Same-wrap annotation mode" (prerequisite for the shift+E commit workflow; seed the preview via `vc3d_shift_click`). |
| `vc3d_shift_click` | `canvas.shift_click` | Shift+click convenience: the canonical place-point / set-focus gesture. |
| `vc3d_switch_workspace` | `workspace.switch` |
| `vc3d_undo_wrap_annotation` | `wrap_annotation.undo` | Undo the same-wrap annotation (Ctrl+Z equivalent): clear the preview or undo the last committed collection. |
| `vc3d_update_point` | `points.update_point` |
| `vc3d_zoom_viewer` | `viewer.zoom` | Multiply a viewer's zoom by a factor (>1 zooms in). Returns the new scale. |

## Verification

Host-side MCP tests:

```sh
cd tools/vc3d-mcp
/tmp/vcmcp/bin/python -m unittest discover -v
```

C++ contract and lifecycle tests:

```sh
docker exec vc3d-bridge bash -lc 'cd /work && \
  ninja -C build/ci-release-gcc VC3D test_agent_bridge_contract && \
  ctest --test-dir build/ci-release-gcc \
    -R "agent_bridge_contract|seeding_batch_tracker|fiber_save_batch_tracker" \
    --output-on-failure'
```

Live offscreen integration:

```sh
docker exec vc3d-bridge bash -lc 'cd /work && QT_QPA_PLATFORM=offscreen python3 \
  apps/VC3D/agent_bridge/test/smoke_offscreen.py \
  --vc3d build/ci-release-gcc/bin/VC3D'
```

The manual fixture suite is documented in `test/README.md`. It requires local
volume-package fixtures and is intentionally separate from hermetic CI.

## Known limitations

- Progress is bounded and best-effort, not durable delivery.
- MCP wait cancellation does not cancel application work.
- Some jobs expose no cancellation authority.
- Viewer settings that exist only on live viewer instances fall back to
  persisted defaults when no viewer exists.
- External tools, remote catalogs, atlas data, Lasagna services, and real
  fixture geometry require their corresponding runtime resources.
- The local bridge is a trusted-user control surface, not a remote security
  boundary.
