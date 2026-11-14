# VC3D Agent Bridge Protocol

Status: current protocol reference. Protocol version 2.

Version history: v2 removed `traceNeedsReview`/`traceVerified` from
`fiber.list` — human review state is the ordinary `reviewed` tag in `tags`.

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
| `catalog` | Open Data sample open lifecycle |
| `flatten` | SLIM lifecycle |
| `seeding` | bridge-started run or expand batch |
| `autosave` | explicit dirty-segment save |
| `import` | `StoreImportController` (bridge and Stores menu imports) |
| `export` | bridge-started object export |
| `manifest` | Lasagna manifest attachment (bridge and File menu) |

Application work started outside the bridge is represented when its lifecycle
is observable, but only bridge-started seeding operations are registered as
jobs. A source retains its eight most recent terminal records.

`job.status` accepts `jobId`, `source`, or neither. With neither it returns the
most recently started job across all sources. Its record contains:

```text
jobId, kind, source, label, state, message, outputPath, externalId,
consoleTail, progressHistory, startedAtMs, finishedAtMs, result, progress
```

`state` is `running`, `succeeded`, or `failed`. `consoleTail` and
`progressHistory` are bounded. `outputPath`, `externalId`, and `finishedAtMs`
are null when unavailable. `result` is an operation-specific object when a
terminal job has structured output, otherwise null; a failed job keeps any
partial report there. `progress` is `{completed, total, detail}` for a job
that counts its work (imports), otherwise null.

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
- segment, fiber, rendering, tracing, and flatten operations report
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
  active segment, editing modes, viewers, point state, and active jobs. It
  reports `author` (the name edits are recorded under: the username setting,
  else the login name, never empty) and `authorConfigured` (the username
  setting names an author). `project.entries` lists the detachable volume,
  normal-grid and regular/fiber Lasagna entries as `{kind, location, scope}`
  (`kind` `volume`, `normal_grid`, `lasagna` or `fiber_lasagna`; `scope`
  `everyone` or `only you`). `umbilicus` is `{volume, store, uuids}` for the
  annotation volume's umbilicus objects in the store routed for umbilici, or
  null without a volume or umbilicus store.
  `segmentationGrowthMethod` reports the selected method even outside an active
  editing session. `manualAddFollowingSheet` indicates that an extension is
  computing; editing remains locked until its completed preview is available.
- A project is a JSON `.vcproj` file referencing volumes, normal grids, Lasagna
  datasets, and one or more `.vcobj` stores. Each of surfaces, fibers, point
  collections, umbilici, and registrations selects one attached store for
  display and editing. `state.get.project` describes the project file;
  `stores` includes reference IDs and availability, and `routes` maps all five
  kinds to those reference IDs or null.
- `project.create` writes a `.vcproj` referencing at most one local Zarr
  volume or remote `.zarr` URL (`volume` is optional; without it the result's
  `volume` is null and segments are listed only once a volume is attached).
  `store` selects an existing store, recorded read-only when this user cannot
  write it; `newStore` creates a store at the given path. If neither is
  supplied, create a sibling `.vcobj`. Supplying both is invalid. The project
  is staged and published atomically; a store created for it is removed again
  on failure. `overwrite` replaces only the project file and never deletes or
  overwrites a referenced store; the open project is never overwritten. The
  result includes the store path. Creation does not open the project; remote
  availability is checked on open. The File menu's New Project shares the same
  core (`vc::project::createProject`).
- `volume.open` opens a project (`.vcproj`); the method keeps its historical
  name.
- `volume.open` may select a volume. Failed opens do not discard the current
  project.
- `volume.attach` loads one local zarr or remote `.zarr` URL asynchronously,
  then persists it in the project only if the same project is still open. It
  preserves the current primary volume, is idempotent by location, and rejects
  a different location that resolves to an existing volume id.
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

Segments are surfaces of the store surfaces are kept in: the project's, or the
store chosen for surfaces. An Open Data sample is opened as a project of its
own whose catalog surfaces are ordinary surfaces of it. A segment id is the
surface's store UUID. Methods on segments need an open project and volume (-32000).

- `segments.list` reports the listed surfaces: `id`, `name`, record `tags`,
  `catalog` (an Open Data catalog surface), `bound` (bound to the open volume),
  `active`, `store` (the project store reference it is in, null for a store
  the project does not reference) and `writable`; `loading` is true while a
  listing is still being read. `segments.activate` and `segments.review`
  report the same fields.
- `segments.activate` validates before changing selection. Activating the
  current segment is a no-op success. A catalog surface whose geometry is still
  remote is fetched first (the call blocks while it downloads).
- Catalog surfaces are edited, renamed, tagged and deleted in place like any
  other; delete, rename and `tags.set` fail with -32009 only on a read-only store.
- `segments.review` derives review status from the record tags, with optional
  tag filters, without changing the current selection.

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
- `viewer.set_hud` switches the measurement overlays for every viewer, like
  View > Measurement Overlays, whose items follow it. A `profileLine` is given in fractions
  of the target viewer's viewport, turns profile mode on and is read from the
  rendered frame, so the values are display gray levels after the volume
  window. The grid draws only on plane views.
- Overlay updates are atomic. A clear request removes the overlay; explicit
  volumes must resolve in the open package. Intersection sets always include
  the segmentation surface and are not applied to the segmentation viewer
  itself.

### Segmentation editing

- Editing must have an active materialized segment.
- `segmentation.grow` accepts tracer and corrections growth.
  Manual add is an editing mode, not a growth method.
- Explicit save returns `jobId:null` when nothing is dirty; otherwise it
  creates an `autosave` job completed by the real save signal.
- Manual-add begin/finish, line mode, interpolation, undo, correction point
  mode, push/pull, and synthetic canvas input all use the active
  `SegmentationModule` session.
- `segmentation.grow_surface_from_seed`, tracing, rendering, and reoptimization
  use the existing command-runner lifecycle and never open the interactive
  console or parameter dialogs.

### Points

Point collection mutations resolve every selector and validate all points
before changing the collection. `points.list` and the responses naming a
`collectionId` also carry the collection's object `uuid` and the `store`
reference it is kept in (null for a surface's corrections, which are not
objects of their own). The UUID is stable across reopen and is what
`objects.export` takes. Bulk operations are all-or-nothing. Winding
values may be finite floats or null where the method declares clearing.
Metadata and tags are idempotent setters. Save/load methods use explicit paths
and never open file pickers.

### Lasagna and fibers

- Lasagna service and job queries use deferred responses from the service
  manager. Optimization is tracked as a `lasagna` job.
- Fiber ids are strings on the wire. Bulk deletion validates every id before
  removing any fiber. Export paths are explicit. Save waits for the
  controller's persistence completion signal.
- `fiber.list` and `fiber.open` also report each fiber's object `uuid` and
  the `store` reference of the fiber store, connecting numeric editing ids to
  store identities (`objects.import` results, `objects.export`).
- Fibers are imported with `objects.import kind=fiber`; `fiber.export` writes
  every fiber of the fiber store as one collection file.
- Fiber launch suppresses dataset pickers; missing prerequisites become
  ordinary errors.

### Stores and imports

Objects live in the project's `.vcobj` stores; `state.get` lists the
references (`stores`) and each kind's selected store (`routes`).

- `objects.import` copies source files into a store as new objects: tifxyz
  surfaces (one directory or a folder of them), fiber JSON (file, collection
  or folder), or a `vc_pointcollections` JSON. It writes to the store routed
  for the kind, or the attached writable `store` named, and never changes
  routing. It is an `import` job (`StoreImportController`, shared with the
  Stores menu's Import from Folder), one at a time; a fiber import is refused
  (-32004) during a fiber merge, split or link.
  - `tags` absent tags each object with the last path component; `[]` adds
    none.
  - `bind.scale` is stored units per current-volume voxel; objects are bound
    to the annotation volume at `scale * annotationVolume.scale`. There is no
    default scale; without `bind` sources keep the coordinates they declare
    (or stay unbound). An explicit binding conflicting with a source's own
    declaration is a problem of that source.
  - `uuidNamespace` (a UUID per dataset) derives identities for sources
    without canonical UUIDs. Bridge imports require source UUIDs otherwise:
    a UUID-less fiber is a problem rather than named after its file, as the
    fiber widget and Spiral still do.
  - The result is `{store, ingested, edgesImported, problems, uuids,
    existing, cancelled, fatalError}`. A completed traversal succeeds even
    with problems or existing identities. A store failure stops the import:
    the job fails with `fatalError: {message}` and the committed part of the
    report. `job.cancel` stops it before the next object; the job fails with
    `cancelled: true` and the partial report. Fiber and point-collection
    sources commit as one batch: one existing identity rejects the batch and
    `existing` lists every clash.
  - Closing the project or detaching the store stops the import between
    objects; what was committed stays. When the project changed before the
    import finished the job fails with `projectReplaced: true` and
    `store: null` (store references belong to the old project).
- `objects.export` writes one surface (tifxyz directory), fiber (fiber JSON)
  or point collection (`vc_pointcollections` JSON with that collection) of an
  attached project store, read-only stores included; objects of a hosted store
  cannot be exported (-32602). Without `store` the UUID
  is looked up across the project's attached stores; several matches fail
  -32602 with the candidates. Catalog-only objects are not candidates. Pending
  edits of the object are saved first (-32004 when an edit operation prevents
  it, -32005 when saving fails, without export). It is an `export` job, not
  cancellable; its result is `{path, uuid, kind, store, version,
  geometryRevision}` from one read snapshot, `version` being the exported
  `{store, generation, object, version}` reference. The files carry the stored frame:
  `uuid`, `coordinates: {volume_uuid, scale_factor}` (stored units per voxel
  of that volume, null when unknown) and `coordinate_binding` when recorded.
  Re-importing them keeps UUID and coordinates.

#### Attaching, routing and detaching stores

- `stores.attach` attaches an existing local store (`readOnly` for read
  access); a store already attached returns its reference unchanged. Its
  result `readOnly` is the actual mode. `stores.create` creates a store at a
  new path and attaches it, removing the file again when attaching fails.
  Neither changes routing.
- `stores.route` selects the store a kind (`surface`, `fiber`,
  `point_collection`, `umbilicus`, `volume_transform`) is shown from and new
  objects go to, as the Stores menu does; null routes none. It validates the
  kind and reference first and persists on its own; a refused route leaves
  the previous one.
- `stores.detach` removes a reference, including an unavailable one; kinds
  routed to it are routed to none. The store file is untouched. It is
  refused (-32004) while an import or export job uses the store; the last
  store of a project cannot be detached (-32005).
- Store switches (detach, route, Save As) first save pending edits: point
  collections, a pending segment autosave and idle line-annotation sessions.
  They are refused with -32004 (`StoreSwitchRefused`) while growth or a line
  optimization runs or when saving fails, and nothing changes.

All of these persist in the `.vcproj` and survive reopening it.

#### Project file and resources

- `project.save_as` saves to a new `.vcproj` (refusing an existing file,
  -32005) and reopens it, keeping the current volume. Resource locations are
  rebased; stores are shared, not copied. Refused (-32004) while an import or
  export job runs, since reopening detaches the stores.
- `project.attach_normal_grid` validates and adds a normal-grid entry;
  attaching the same location again succeeds with `alreadyAttached`.
- `project.attach_lasagna_manifest` loads and validates a local or remote
  manifest as a `manifest` job, as the File menu does, with the explicit
  `role` (`regular` or `fiber`); it never infers the role. Completion after
  the project was replaced fails without attaching. Not cancellable.
- `project.detach_entry` removes a volume, normal-grid or Lasagna entry by
  the exact `location` of a `state.get` `project.entries` item (-32007 when
  there is none), with no confirmation step. Pending edits are saved first
  (-32004 when they cannot be). The result is `{location, scope}`. Detaching
  the current volume selects another (or none) through the same refresh as
  the GUI. Source data is untouched.
- `umbilicus.import` reads an umbilicus file and binds it to the current
  volume: a stated frame is used as is; an inferred or missing one needs
  `unitsPerVoxel` (-32602 with `data.inferred` and `data.frame` otherwise).
  An unreadable file fails -32602 (`param: path`); a file that does not fit,
  an existing umbilicus without `replace`, or a failed read of the umbilicus
  store fails -32005. It needs a selected volume (-32001) and a writable umbilicus store
  (-32009).
- `umbilicus.remove` retires the annotation volume's umbilicus objects in the
  umbilicus store in one transaction, so other projects sharing the store see
  it too; `stores.route(kind="umbilicus", storeId=null)` hides umbilici for
  this project only. No umbilicus is a success with `removed: []`.

### Seeding, rendering, flattening, and mesh operations

- Seeding run and expand share one `seeding` source. Reset and path analysis
  are synchronous requests over the existing widget state.
- `render.tifxyz`, trace, reoptimize, and alpha-composition refinement use the
  shared `tool` source.
- SLIM uses the `flatten` source and application
  lifecycle.
- Crop and area recalculation are synchronous.
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
| `vc3d_apply_anchor_offset` | `points.apply_anchor_offset` |
| `vc3d_attach_lasagna_manifest` | `project.attach_lasagna_manifest` | Attach a regular or fiber Lasagna dataset by manifest, with an explicit role. A `manifest` job. |
| `vc3d_attach_normal_grid` | `project.attach_normal_grid` | Attach a normal-grid root to the open project. |
| `vc3d_attach_store` | `stores.attach` | Attach an existing local store to the open project without changing routing. |
| `vc3d_attach_volume` | `volume.attach` | Attach one local zarr volume or remote `.zarr` URL to the open project without changing the current primary volume. Returns a `volume` job; compose with `vc3d_list_overlay_volumes` and `vc3d_set_overlay` to display it. |
| `vc3d_auto_fill_windings` | `points.auto_fill_windings` |
| `vc3d_cancel_job` | `job.cancel` | Request cancellation of a running job by `jobId` (or by `source`). Only `tool`/`seeding`/`lasagna`/`import` jobs are cancellable; `growth`/`flatten`/`catalog`/`volume`/`autosave`/`export` return an error. Request-only — poll `vc3d_job_status` for the terminal state. |
| `vc3d_center_viewer` | `viewer.center_on_point` | Center a viewer pane on a 3D volume point. |
| `vc3d_clear_all_points` | `points.clear_all` |
| `vc3d_clear_point_collection` | `points.clear_collection` |
| `vc3d_click` | `canvas.click` | Synthesize a mouse click in a viewer at a volume-space (or scene-space) position, with button and modifiers (e.g. `{"modifiers": ["shift"]}` to place a point / set focus). |
| `vc3d_commit_points` | `points.commit` | Add annotation points (volume space) to a named collection, optionally with a winding annotation. |
| `vc3d_commit_wrap_annotation` | `wrap_annotation.commit` | Commit the seeded same-wrap annotation preview into the point collection (the tutorial's shift+E). Requires the mode enabled. |
| `vc3d_corrections_set_point_mode` | `segmentation.corrections.set_point_mode` |
| `vc3d_create_project` | `project.create` | Create a `.vcproj` project, optionally referencing one local zarr volume or remote `.zarr` URL, without opening it. |
| `vc3d_create_store` | `stores.create` | Create a new store and attach it to the open project. |
| `vc3d_crop_segment_bounds` | `segment.crop_bounds` | Crop a segment's surface grid to its tightest valid bounds. Synchronous. |
| `vc3d_delete_segment` | `segments.delete` | Retire a surface in its store (it leaves the listing; store history keeps it). Requires `confirm=true`. Fails while segmentation editing is enabled and on read-only stores. |
| `vc3d_describe_catalog_sample` | `catalog.describe_sample` |
| `vc3d_detach_project_entry` | `project.detach_entry` | Remove a volume, normal-grid or Lasagna entry from the open project. |
| `vc3d_detach_store` | `stores.detach` | Remove a store reference from the open project; the store is untouched. |
| `vc3d_drag` | `canvas.drag` |
| `vc3d_enable_editing` | `segmentation.enable_editing` | Turn segmentation editing mode on/off for the active segment. |
| `vc3d_export_object` | `objects.export` | Export one surface, fiber or point collection of an attached store as native files, in its stored frame. An `export` job. |
| `vc3d_fiber_delete` | `fiber.delete` |
| `vc3d_fiber_export` | `fiber.export` |
| `vc3d_fiber_launch` | `fiber.launch` |
| `vc3d_fiber_list` | `fiber.list` | Per fiber includes `traceState` (`legacy`/`predictions`/`mixed`) and per-span `interpMode` (`C`/`L`/`T`) plus `fiberManifest` (predictions provenance, trace spans only). Human review state is just the ordinary `reviewed` tag in `tags`. |
| `vc3d_fiber_open` | `fiber.open` |
| `vc3d_fiber_save` | `fiber.save` |
| `vc3d_fiber_set_follow` | `fiber.set_follow` |
| `vc3d_fiber_set_tag` | `fiber.set_tag` |
| `vc3d_flatten_slim` | `flatten.slim` |
| `vc3d_get_cursor_point` | `canvas.get_cursor_volume_point` | Resolve a viewer scene position (or the current cursor) to a 3D volume point + surface normal. |
| `vc3d_get_overlay` | `viewer.get_overlay` | Read the active workspace's overlay-volume settings, shared with the GUI controls. |
| `vc3d_get_render_settings` | `viewer.get_render_settings` | Read the shared viewer render/overlay settings (intersection lines, overlay opacity, surface normals, highlighted surfaces). |
| `vc3d_get_state` | `state.get` | Snapshot of VC3D: open project and its stores, current volume, active segment, viewers (ids/names), editing mode, running job. Call this first. |
| `vc3d_grow_surface_from_seed` | `segmentation.grow_surface_from_seed` | Create a new surface by growing it from a 3D seed point (headless GrowPatch). Async: returns a jobId; the new surface's UUID is the finished job's output. |
| `vc3d_grow_segment` | `segmentation.grow` | Grow the active segmentation surface (method: tracer/corrections; direction; steps). Async: returns a jobId. |
| `vc3d_import_objects` | `objects.import` | Import tifxyz surfaces, fiber JSON or point collections into a project store as new objects. An `import` job reporting new UUIDs, existing identities and problems. |
| `vc3d_import_umbilicus` | `umbilicus.import` | Import an umbilicus file for the current volume. |
| `vc3d_job_status` | `job.status` | Poll a job by id (or the latest job): state, message, console tail. |
| `vc3d_lasagna_cancel` | `lasagna.cancel` |
| `vc3d_lasagna_ensure_service` | `lasagna.ensure_service` |
| `vc3d_lasagna_jobs` | `lasagna.jobs` |
| `vc3d_lasagna_list_datasets` | `lasagna.list_datasets` |
| `vc3d_lasagna_repeat_last` | `lasagna.repeat_last` |
| `vc3d_lasagna_select_output` | `lasagna.select_output_segment` |
| `vc3d_lasagna_service_status` | `lasagna.service_status` |
| `vc3d_lasagna_start_optimization` | `lasagna.start_optimization` |
| `vc3d_list_catalog_samples` | `catalog.list_samples` | List samples available to open from the Open Data catalog; use this for discovery when no volume package is loaded. |
| `vc3d_list_overlay_volumes` | `viewer.list_overlay_volumes` | List every volume id in the open package and the active workspace's selection. |
| `vc3d_list_points` | `points.list` | List point collections and their points. |
| `vc3d_list_segments` | `segments.list` | List the surfaces of the open volume (your store and Open Data sample stores) by UUID, with name, tags and active/catalog/bound flags. |
| `vc3d_list_attached_volumes` | `volume.list` | List volume ids already attached to the open package; this does not discover catalog data available to open. |
| `vc3d_manual_add_add_constraint` | `segmentation.manual_add.add_constraint` |
| `vc3d_manual_add_begin` | `segmentation.manual_add.begin` |
| `vc3d_manual_add_extend` | `segmentation.manual_add.extend` |
| `vc3d_manual_add_finish` | `segmentation.manual_add.finish` |
| `vc3d_manual_add_set_interpolation` | `segmentation.manual_add.set_interpolation` |
| `vc3d_manual_add_set_line_mode` | `segmentation.manual_add.set_line_mode` |
| `vc3d_manual_add_undo_constraint` | `segmentation.manual_add.undo_constraint` |
| `vc3d_open_catalog_sample` | `catalog.open_sample` | Open a sample returned by `vc3d_list_catalog_samples`, using its manifest sample id. |
| `vc3d_open_volume` | `volume.open` | Open a project (.vcproj) and optionally select a volume id. |
| `vc3d_ping` | `ping` | Check the VC3D bridge is alive; returns pid, app version, and protocol version. |
| `vc3d_push_pull_set_config` | `segmentation.push_pull.set_config` |
| `vc3d_push_pull_start` | `segmentation.push_pull.start` |
| `vc3d_push_pull_stop` | `segmentation.push_pull.stop` |
| `vc3d_refine_segment_alpha_comp` | `segment.refine_alpha_comp` | Alpha-composite refinement of a segment. Asynchronous (`source:"tool"`); rejects remote volumes; supports `wait: bool = false` (30-minute cap). |
| `vc3d_remove_point_collection_tag` | `points.remove_collection_tag` |
| `vc3d_remove_point` | `points.remove_point` |
| `vc3d_remove_umbilicus` | `umbilicus.remove` | Retire the current volume's umbilicus in the umbilicus store. |
| `vc3d_rename_point_collection` | `points.rename_collection` |
| `vc3d_rename_segment` | `segments.rename` | Rename a surface (its UUID stays). `new_name` must differ from the current name. Fails in a read-only store. |
| `vc3d_render_tifxyz` | `render.tifxyz` |
| `vc3d_reoptimize_segment` | `segment.reoptimize` | Resume-opt local reoptimization of a segment. Asynchronous (`source:"tool"`); supports `wait: bool = false` (30-minute cap). |
| `vc3d_reset_windings` | `points.reset_windings` |
| `vc3d_review_segments` | `segments.review` | List segments with review-tag state and optional server-side filtering (the programmatic equivalent of the surface panel's review filter checkboxes). |
| `vc3d_rotate_viewer` | `viewer.rotate` | Rotate the "seg xz"/"seg yz" axis-aligned slice plane (middle-drag equivalent). Relative delta by default. |
| `vc3d_route_store` | `stores.route` | Choose the store a kind of object is shown from and created in, or none. |
| `vc3d_run_trace` | `tracer.run_trace` |
| `vc3d_save_project_as` | `project.save_as` | Save the open project to a new file and reopen it. |
| `vc3d_save_segment` | `segmentation.save` | Write the active segment's pending edits to the store now. Idle no-op (`jobId:null`) when nothing is dirty; else an `"autosave"` job (`wait` defaults true). |
| `vc3d_screenshot` | `screenshot.capture` | Capture a PNG of the whole VC3D window or one viewer pane. Returns the PNG as MCP image content when `file_path` is omitted, or a dict with the on-disk path when `file_path` is set (MCP-layer note below). |
| `vc3d_seeding_analyze_paths` | `seeding.analyze_paths` |
| `vc3d_seeding_cancel` | `seeding.cancel` |
| `vc3d_seeding_expand` | `seeding.expand` |
| `vc3d_seeding_reset_points` | `seeding.reset_points` |
| `vc3d_seeding_run` | `seeding.run` |
| `vc3d_seeding_set_winding_annotation_mode` | `seeding.set_winding_annotation_mode` |
| `vc3d_select_volume` | `volume.select` |
| `vc3d_set_auto_fill_mode` | `points.set_auto_fill_mode` |
| `vc3d_set_axis_aligned_slices` | `viewer.set_axis_aligned_slices` | Enable/disable axis-aligned slice mode (checkbox equivalent) — prerequisite for `viewer.rotate`. |
| `vc3d_set_hud` | `viewer.set_hud` | Turn the slice viewers' measurement overlays (histogram, line profile, coordinate grid, loupe) on or off; with a profile line, returns the gray values, peaks and length along it. |
| `vc3d_set_intersects` | `viewer.set_intersects` | Set which surfaces' intersection lines viewers draw; stored surfaces (UUIDs) are drawn on every plane viewer. |
| `vc3d_set_overlay` | `viewer.set_overlay` | Update the active workspace and GUI overlay controls together (volume, colormap, opacity, window, intensity-as-alpha, resolution cap, composite); any subset. |
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
- External tools, remote catalogs, Lasagna services, and real fixture
  geometry require their corresponding runtime resources.
- The local bridge is a trusted-user control surface, not a remote security
  boundary.
