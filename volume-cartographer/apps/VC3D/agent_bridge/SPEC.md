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
MCP tests compare FastMCP schemas and mappings with the same snapshot.

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
| `volume` | local or remote volume attachment lifecycle |
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

- `segments.attach` adds one absolute local tifxyz path to the open package.
  The path may identify a segment or a directory of segments. Attachment uses
  the same validation, persistence, and UI refresh as the GUI and is
  idempotent by normalized location. It selects the attached segment source by
  default; `select: false` preserves the current source. A source whose
  directory name is already used case-insensitively by another attachment is
  rejected because the VC3D source picker identifies entries by that name.
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
- `lasagna.attach_manifest` reuses the GUI's existing manifest loader,
  authentication, cache configuration, and transactional package attachment;
  it adds no alternate cache or sidecar format.
- Atlas open, remap, result selection, and candidate optimization use
  dialog-free `CWindow` operations. Only bridge-started searches become
  `atlas` jobs.
- Fiber ids are strings on the wire. Bulk deletion validates every id before
  removing any fiber. Import/export paths are explicit. Save waits for the
  controller's persistence completion signal.
- Fiber launch and create-atlas operations suppress dataset pickers and atlas
  rebuild prompts; missing prerequisites become ordinary errors.

### Spiral

- Spiral methods are thin calls into the existing project-scoped
  `SpiralServiceManager` used by the GUI workspace.
- Connections use saved GUI profiles. Direct profiles take their API key from
  `SPIRAL_API_KEY`; credential values are never RPC parameters or results.
- Input uploads and checkpoint transfers are deferred responses completed by
  existing service signals. Run, stop, preview, and save requests return once
  accepted and do not create bridge jobs or operation records.

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

Spiral tools call the existing workspace `SpiralServiceManager` directly.
They are not VC3D jobs and expose no MCP wait, operation id, or event cursor;
observe the service through `spiral.status`, `spiral.dataset`, and the shared
Spiral workspace.

`vc3d_wait_job` applies the same behavior to an existing job.

`workspace.switch` accepts `spiral` in addition to the existing workspaces, so
agents can inspect the same application-lifetime controller state and preview
that the GUI uses before calling `screenshot.capture`.

Without `file_path`, `vc3d_screenshot` decodes inline PNG data into FastMCP
image content. With `file_path`, it returns the bridge's file result and does
not include inline image data.

## MCP tool coverage

The generated `rpc_description.json` records every RPC-to-MCP mapping,
parameter rename, and MCP-only extra parameter. FastMCP exposes each registered
tool description and input schema at runtime. `rpc.describe` is the sole bridge
method without an MCP tool; `vc3d_wait_job` is the sole MCP-only convenience
tool.

Contract tests compare the generated descriptors directly with the registered
FastMCP tools and schemas. Do not duplicate that generated mapping in this
narrative specification.

## Verification

Host-side MCP tests:

```sh
cd tools/vc3d-mcp
python -m unittest discover -v
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
