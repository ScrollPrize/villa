# VC3D Agent Bridge — Command Surface Specification

Status: **binding design** (Phase 0). Implementers follow this exactly; deviations require
updating this document first.

The agent bridge is an in-process JSON-RPC 2.0 server inside VC3D, listening on a local
socket (`QLocalServer` / `QLocalSocket`). A separate MCP server process (Phase 3) connects
as a client and translates MCP tool calls from an AI agent into these RPCs.

All grounding references below are to real code on branch `feature/vc3d-agent-bridge`,
repo root `volume-cartographer/`, app dir `apps/VC3D/`.

---

## 1. Transport, activation, and socket naming

### 1.1 CLI flags (opt-in, off by default)

Added to the existing `QCommandLineParser` block in `apps/VC3D/VCAppMain.cpp`
(same pattern as `--record` / `--replay`, lines ~217–322):

```
--agent-bridge                 Enable the agent bridge on the default socket name.
--agent-bridge-name <name>     Enable the agent bridge on an explicit QLocalServer name
                               (implies --agent-bridge).
```

Both are plain `QCommandLineOption`s. If neither flag is present, **no bridge object is
constructed at all** — zero runtime cost and zero listening sockets in normal use.

- Default socket name when only `--agent-bridge` is given:
  `vc3d-agent-<pid>` where `<pid>` is `QCoreApplication::applicationPid()`.
  The PID suffix makes concurrent test runs safe by construction.
- `--agent-bridge-name foo` uses `foo` verbatim (for tests that want a predictable name).
  Before listening the server sets `QLocalServer::UserAccessOption` so only the current
  user can connect (the bridge grants full control of the running app). If
  `QLocalServer::listen(name)` fails, the server does **not** blindly unlink the endpoint:
  it first probes `name` with a `QLocalSocket`. A successful connect means a **live**
  bridge already owns the name — the server reports the collision and returns false
  **without** removing it (unlinking would strand the other server's clients). Only when
  the probe fails (a demonstrably stale socket file from a crashed run) does it call
  `QLocalServer::removeServer(name)` once and retry; if it still fails, print an error to
  stderr and exit with code 2 (a test that asked for a bridge must not silently run without
  one). On Unix the whole probe→remove→listen sequence is serialized by an advisory
  `flock` on a name-derived lock file (`<tmp>/<name>.listen.lock`): without it two VC3D
  processes starting the **same** name *concurrently* could both probe before either
  listens (each sees "not live"), and the loser's `removeServer` would then unlink the
  winner's live endpoint. The lock is held across probe+listen and released once the
  process is the live owner (or has refused), so the second acquirer's probe sees the first
  as live and refuses — the live socket is still never unlinked.

On successful listen, VC3D prints exactly one machine-parseable line to stdout:

```
VC3D-AGENT-BRIDGE: listening name=<serverName> path=<QLocalServer::fullServerName()>
```

The MCP server discovers the socket either from `--agent-bridge-name` (agreed out of band)
or by parsing this line when it spawned VC3D itself.

The bridge is constructed in `VCAppMain.cpp` right after `CWindow aWin(...)` (~line 389),
taking `CWindow&` (the bridge class is added to CWindow's friend list, following the
existing `friend class RenderBenchReplay;` precedent at `CWindow.hpp:123` — add
`friend class AgentBridge;` next to it).

### 1.2 Wire framing

- One JSON-RPC 2.0 message per line, UTF-8, LF-terminated (newline-delimited JSON).
  No embedded newlines inside a message (standard compact `QJsonDocument::Compact`).
- A single framed request is bounded to **1 MiB**. If a client's un-framed read buffer
  grows past the bound without a newline, or a newline-terminated line itself exceeds it,
  the server sends a best-effort `-32600 Invalid Request` and disconnects **only that
  client**; other connections are unaffected (the per-socket buffer cannot grow without
  limit). The residual after consuming complete lines is bounded too: a valid line followed
  by a >1 MiB unterminated tail (e.g. `"{}\n"` + junk) is caught by a post-framing check, so
  the pre-loop guard being skipped whenever a newline is present cannot leave an oversized
  tail buffered indefinitely.
- Multiple concurrent client connections are allowed. Requests are executed strictly
  serially on the Qt main thread (queued via the socket's `readyRead` on the GUI thread);
  there is no request pipelining guarantee beyond FIFO per connection.
- Server → client **notifications** (no `id`) are broadcast to *all* connected clients.
- Batch requests (JSON array) are **not supported**; respond with `-32600 Invalid Request`.

### 1.3 Threading model

Every RPC handler runs on the Qt GUI thread (the bridge lives on the main thread and
QLocalSocket signals are delivered there). Handlers must never call `exec()` on a dialog
or otherwise spin a nested event loop. Long-running operations (growth, external tools)
are **jobs**: the RPC returns immediately with a `jobId` and progress is delivered via
`job.progress` notifications.

---

## 2. Common conventions

### 2.1 Coordinate spaces

- `"volume"` — full-resolution voxel coordinates `{x, y, z}` (floats), the space of
  `cv::Vec3f vol_loc` used throughout VC3D (e.g. `CWindow::onVolumeClicked`).
- `"scene"` — a viewer's QGraphicsScene coordinates `{x, y}` (floats), the space of
  `CChunkedVolumeViewer::onMousePress(QPointF, ...)`.

Conversions use the real viewer API (`VolumeViewerBase.hpp:55-58`):
`volumeToScene(cv::Vec3f)` and `sceneToVolume(QPointF)`.

### 2.2 Viewer targeting (multi-viewer scheme)

`ViewerManager` (apps/VC3D/ViewerManager.hpp) owns all viewers:
`const std::vector<VolumeViewerBase*>& baseViewers() const` (line 64), plus
`baseViewerCreated(VolumeViewerBase*)` / `baseViewerClosing(VolumeViewerBase*)` signals
(lines 151–152). Each viewer reports its bound surface slot via
`std::string surfName() const`. The default workspace creates viewers on the well-known
slots `"segmentation"` (Surface view), `"xy plane"`, `"seg xz"`, `"seg yz"`
(CWindow.cpp ~4313–4316). Additional viewers with duplicate slot names can exist
(fiber/annotation viewers), so slot name alone is not a unique key.

**Scheme:** the bridge maintains its own registry, populated from
`ViewerManager::baseViewers()` at attach time and kept current via
`baseViewerCreated` / `baseViewerClosing`. Each viewer gets a stable string id
`"v<N>"` (`v1`, `v2`, … in registration order, never reused within a process).

Every RPC that targets a viewer takes a `"viewer"` string param resolved as follows:

1. If it matches a registry id (`"v3"`), use that viewer.
2. Otherwise treat it as a surface-slot name and match against `surfName()` of live
   viewers. Exactly one match → use it. Multiple matches → error `-32002` with
   `data.candidates` listing `{viewerId, surfName, title}` for each match. Zero matches
   → error `-32002`.
3. If the param is omitted where marked optional, the default is the slot
   `"segmentation"` (resolved via rule 2).

`state.get` returns the full registry so agents can enumerate before targeting.

Canvas RPCs additionally require the resolved viewer to be a `CChunkedVolumeViewer`
(checked via `dynamic_cast`); a non-chunked viewer yields error `-32009 UNSUPPORTED`.

### 2.3 Mouse button and modifier mapping

JSON → Qt mapping used by `canvas.click` / `canvas.shift_click` (and echoed in results):

| JSON `button`  | Qt::MouseButton     |
|----------------|---------------------|
| `"left"`       | `Qt::LeftButton`    |
| `"right"`      | `Qt::RightButton`   |
| `"middle"`     | `Qt::MiddleButton`  |

| JSON modifier string | Qt::KeyboardModifier   |
|----------------------|------------------------|
| `"shift"`            | `Qt::ShiftModifier`    |
| `"ctrl"`             | `Qt::ControlModifier`  |
| `"alt"`              | `Qt::AltModifier`      |
| `"meta"`             | `Qt::MetaModifier`     |
| `"keypad"`           | `Qt::KeypadModifier`   |

`modifiers` is always a JSON array of the strings above (empty array = `Qt::NoModifier`);
the bridge ORs them together. Unknown button/modifier strings → `-32602 Invalid params`.
Example: `{"button": "left", "modifiers": ["shift", "ctrl"]}` →
`Qt::LeftButton`, `Qt::ShiftModifier | Qt::ControlModifier`.

### 2.4 Job model

The bridge tracks at most **one active job** (matching reality: `CommandLineToolRunner`
is a single QProcess with `isRunning()`, and `SegmentationGrower::running()` is a single
flag). Job ids are `"job-<n>"`, monotonically increasing. Sources of job lifecycle:

- `CommandLineToolRunner::toolStarted` / `toolFinished` / `consoleOutputReceived`
  (CommandLineToolRunner.hpp:119–122).
- `CWindow::onSegmentationGrowthStatusChanged(bool running)` (CWindow.hpp:274) /
  `SegmentationGrower::running()` for in-process growth.

Starting any job-producing RPC while a job is active → error `-32004 JOB_RUNNING`
(with `data.jobId` of the active job).

### 2.5 Error object

Standard JSON-RPC error member: `{"code": <int>, "message": <string>, "data": {...}}`.
`data` is always an object when present; fields are error-specific but `data.detail`
(free-form human string) is always allowed.

| Code    | Name                 | Meaning / typical `data` fields |
|---------|----------------------|---------------------------------|
| -32700  | Parse error          | malformed JSON line |
| -32600  | Invalid Request      | not a JSON-RPC 2.0 object, or batch array |
| -32601  | Method not found     | |
| -32602  | Invalid params       | wrong type/missing field; `data.param` names the offender |
| -32000  | NO_VOLPKG            | no volume package loaded (`CState::hasVpkg()` false) |
| -32001  | NO_VOLUME            | vpkg loaded but `CState::currentVolume()` is null |
| -32002  | INVALID_VIEWER       | viewer id/slot unresolvable; `data.candidates` on ambiguity |
| -32003  | INVALID_COORDINATES  | point outside volume bounds / not on surface; `data.point` |
| -32004  | JOB_RUNNING          | a job is already active; `data.jobId` |
| -32005  | JOB_FAILED           | synchronous launch failure of a job; `data.detail` |
| -32006  | TOOL_NOT_FOUND       | external executable not found (e.g. `vc_grow_seg_from_seed`) |
| -32007  | NOT_FOUND            | named segment / sample / collection / point / job absent; `data.kind`, `data.id` |
| -32008  | EDITING_REQUIRED     | op requires segmentation editing to be enabled |
| -32009  | UNSUPPORTED          | viewer/target does not support the operation |
| -32010  | INTERNAL             | unexpected C++ exception; `data.detail` |

---

## 3. JSON-RPC method reference

Types below: `str`, `int`, `float`, `bool`, `obj`, `arr`. `?` marks optional params.
`Vec3` means `{"x": float, "y": float, "z": float}`.

### 3.1 `ping`

Liveness check. No app-state requirements.

- **params:** none (omit or `{}`).
- **result:** `{"pong": true, "pid": int, "version": str, "protocolVersion": 1}` —
  `version` comes from `QApplication::applicationVersion()`; `protocolVersion`
  identifies the bridge/MCP wire contract.
- **errors:** none.

### 3.2 `state.get`

Snapshot of app state. Never errors on missing volume — reports what exists.

- **params:** none.
- **result:**
  ```json
  {
    "vpkg":   null | {"path": str},
    "volume": null | {"id": str, "path": str, "voxelSize": float,
                      "volumeIds": [str]},
    "activeSurface": null | {"id": str},
    "segmentationGrowthVolumeId": str,
    "segmentationEditingEnabled": bool,
    "viewers": [
      {"viewerId": "v1", "surfName": "segmentation", "title": str,
       "kind": "chunked" | "other", "scale": float}
    ],
    "job": null | {"jobId": str, "kind": str, "label": str, "running": true},
    "focusPoi": null | {"position": Vec3, "normal": Vec3, "surfaceId": str}
  }
  ```
  Sources: `CState::vpkgPath/currentVolume/currentVolumeId/activeSurfaceId/`
  `segmentationGrowthVolumeId` (CState.hpp), viewer registry (§2.2),
  `VolumeViewerBase::getCurrentScale()`, `CState::poi("focus")`.
  `volume.volumeIds` is an **additive v2-era field** (source `state->vpkg()->volumeIDs()`)
  nested inside the `volume` block, so it is absent whenever `volume` is `null`; the
  current-volume-independent discovery path is `volume.list` (§22.1). See §22.2 for the
  placement rationale.
- **errors:** none.

### 3.3 `segments.list`

- **params:** `{"onlyLoaded"?: bool = false}`
- **result:**
  ```json
  {"segments": [
     {"id": str, "path": str, "loaded": bool, "active": bool}
  ]}
  ```
  From `VolumePkg` segmentation listing plus `CState::surfaceNames()` for loaded state;
  `active` = `CState::activeSurfaceId() == id`.
- **errors:** `-32000 NO_VOLPKG`.

### 3.4 `screenshot.capture`

- **params:**
  ```json
  {"target"?: str = "window",   // "window" (whole CWindow) or a viewer ref (§2.2)
   "filePath"?: str,            // absolute path; when set, PNG is written to disk
   "maxDim"?: int}              // optional downscale (longest side, aspect-preserving)
  ```
- **result:**
  `{"width": int, "height": int, "format": "png",
    "filePath": str | null, "base64": str | null}`
  — `base64` is set iff `filePath` was omitted. Implementation: `QWidget::grab()` on
  `CWindow` or the resolved viewer widget (`CChunkedVolumeViewer` is a `QWidget`).
  Before grabbing, the target's `isVisible()` is checked, and the grabbed pixmap's
  size is checked against a minimum (8px/side) — a widget on a non-frontmost tab
  (verified live: a fiber-workspace pane while a different tab was active) is
  hidden by its `QStackedWidget`/`QTabWidget` and would otherwise `grab()` a
  meaningless near-zero-size image (observed: 15×50px) with no error.
- **errors:** `-32002` (bad viewer target), `-32005` (file write failed, reuse with
  `data.detail`), `-32009` (target widget not visible, or captured size degenerate;
  `data.detail`).

### 3.5 `canvas.get_cursor_volume_point`

Resolve a viewer position to a volume point via the viewer's real sampling path
(`CChunkedVolumeViewer::sampleSceneVolume(QPointF)` — CChunkedVolumeViewer.hpp:192).

- **params:**
  ```json
  {"viewer"?: str,             // §2.2, default "segmentation"
   "scene"?: {"x": float, "y": float}}  // omitted = viewer->lastScenePosition()
  ```
- **result:**
  ```json
  {"volumePoint": Vec3, "normal": Vec3,
   "scene": {"x": float, "y": float},
   "surfName": str}
  ```
- **errors:** `-32001`, `-32002`, `-32009`, `-32003` (scene position does not hit the
  surface/volume — `sampleSceneVolume` returned `nullopt`).

### 3.6 `canvas.click`

Synthesize a full click at a position, through the viewer's public mouse slots
(`CChunkedVolumeViewer::onMousePress/onMouseRelease/onVolumeClicked`,
CChunkedVolumeViewer.hpp:230–233), so all real signal wiring
(`sendVolumeClicked` → `CWindow::onVolumeClicked`, point placement, tools) fires
exactly as for a human click.

- **params:**
  ```json
  {"viewer"?: str,                       // §2.2, default "segmentation"
   "position": Vec3 | {"x": float, "y": float},
   "space"?: "volume" | "scene",         // default "volume"; must match position shape
   "button"?: str = "left",              // §2.3
   "modifiers"?: [str] = []}             // §2.3
  ```
  With `space:"volume"`, the bridge converts via `volumeToScene()` and verifies the
  round-trip (`sceneToVolume`) lands within 2.0 voxels of the request; farther → `-32003`
  (the point isn't on this viewer's current slice/surface view).
  Dispatch order per click: `onMousePress(scenePos, button, modifiers)` →
  `onMouseRelease(scenePos, button, modifiers)` → `onVolumeClicked(scenePos, button, modifiers)`.
- **result:**
  ```json
  {"clicked": true,
   "scene": {"x": float, "y": float},
   "volumePoint": Vec3 | null,           // from sampleSceneVolume at the click position
   "button": str, "modifiers": [str]}
  ```
- **errors:** `-32001`, `-32002`, `-32003`, `-32009`, `-32602`.

### 3.7 `canvas.shift_click`

Convenience alias: identical to `canvas.click` with `"shift"` unioned into `modifiers`.
(Shift+click is the canonical "place point / set focus POI" gesture in VC3D.)

- **params:** same as `canvas.click` minus no need to pass `"shift"`.
- **result / errors:** same as `canvas.click`.

### 3.8 `viewer.center_on_point`

- **params:** `{"viewer"?: str, "point": Vec3, "forceRender"?: bool = true}`
- Maps to `VolumeViewerBase::centerOnVolumePoint(cv::Vec3f, bool)`
  (VolumeViewerBase.hpp:83).
- **result:** `{"centered": true, "viewerId": str}`
- **errors:** `-32001`, `-32002`, `-32003` (point outside volume bounds).

### 3.9 `viewer.zoom`

- **params:** `{"viewer"?: str, "factor": float}` — `factor` > 0; >1 zooms in,
  <1 zooms out. A **true scale multiplier** applied in one call (new scale =
  `clamp(scale × factor, kMinScale, kMaxScale)`), centered on the viewport — not
  a fixed wheel notch. A very large factor saturates at the viewer's max zoom, so
  the returned `scale` may reflect less than the full multiply; compare it to
  gauge remaining headroom.
- Maps to `VolumeViewerBase::adjustZoomByFactor(float)` (VolumeViewerBase.hpp:85),
  which shares `CChunkedVolumeViewer::zoomByFactorAt` with the wheel zoom.
- **result:** `{"scale": float}` — post-zoom `getCurrentScale()`.
- **errors:** `-32002`, `-32602` (factor ≤ 0 or non-finite).

### 3.9b `viewer.rotate`

- **params:** `{"plane": "seg xz" | "seg yz", "degrees": float, "relative"?: bool = true}`
  — the programmatic equivalent of the human middle-drag rotation on the axis-aligned
  slice panes. `plane` accepts the `"xz"`/`"yz"` shorthands. When `relative` is true
  (default) `degrees` is a delta added to the plane's current angle; when false it is
  an absolute angle. Only the two axis-aligned seg planes rotate — the main xy/segment
  view is not rotatable.
- Maps to `AxisAlignedSliceController::setRotationDegrees(std::string, float)` +
  `scheduleOrientationUpdate()` + `flushOrientationUpdate()` (AxisAlignedSliceController.cpp:483,493,509),
  the same path the middle-drag handler drives; angle read back via `currentRotationDegrees`.
  The `scheduleOrientationUpdate()` call is required: it sets the `_orientationDirty`
  flag, and `flushOrientationUpdate()` early-returns unless that flag is set — without it
  the angle field updates but `applyOrientation()`/the viewer repaint never run, so the
  plane state changes with no visible rotation.
- **result:** `{"plane": str, "degrees": float, "previousDegrees": float, "relative": bool}`
  — `degrees` is the normalized post-rotation angle (`remainder(·, 360)`).
- **errors:** `-32000` (slice controller unavailable), `-32002` (axis-aligned slice mode
  not active), `-32602` (unknown plane, or non-finite/missing `degrees`).

### 3.9c `viewer.set_axis_aligned_slices`

- **params:** `{"enabled": bool}` — turns axis-aligned slice mode on/off, the same
  checkbox (`chkAxisAlignedSlices`) and keyboard shortcut a human uses to make
  `"seg xz"`/`"seg yz"` the rotatable canonical slice planes. This is the prerequisite
  for `viewer.rotate` (§3.9b), which returns `-32002` when the mode is off; there was no
  prior RPC to enable it.
- Drives `CWindow::chkAxisAlignedSlices->setChecked(enabled)` (CWindow is a friend), which
  emits `toggled` synchronously into `CWindow::onAxisAlignedSlicesToggled` (CWindow.cpp:10601)
  → `AxisAlignedSliceController::setEnabled(...)` + persists the `USE_AXIS_ALIGNED_SLICES`
  QSetting + syncs the overlay checkbox. Idempotent: setting the current value is a no-op.
  Falls back to `setEnabled(enabled)` directly if no checkbox exists.
- **result:** `{"enabled": bool}` — the mode state after toggling (`isEnabled()`).
- **errors:** `-32000` (slice controller unavailable), `-32602` (missing/non-bool `enabled`).
- The current mode + per-plane rotation angles are also reported in `state.get` under
  `"axisAlignedSlices": {"enabled": bool, "segXZRotationDeg": float, "segYZRotationDeg": float}`.
### 3.9d `wrap_annotation.set_mode` / `wrap_annotation.commit` / `wrap_annotation.undo`

The **same-winding wrap annotation** workflow — the tutorial's **shift+E**. Human flow:
enable "Same-wrap annotation mode" (a checkbox in the Wrap Annotation panel) → shift-click
on a chunked volume viewer pane to seed preview points → **shift+E** to commit them into
the point collection (Ctrl+Z clears the preview / undoes the last committed collection).
The preview is seeded **only** by `canvas.shift_click` on a chunked viewer, exactly as for
a human — there is no seeding RPC here.

**`wrap_annotation.set_mode`**
- **params:** `{"enabled": bool}` — required boolean.
- Maps to `WrapAnnotationWidget::setSameWrapAnnotationEnabled(bool)`, which drives the
  `_chkSameWrapAnnotation` checkbox's `toggled()` signal → `CWindow` →
  `CChunkedVolumeViewer::setSameWrapAnnotationMode(bool)` (the same wiring a human click
  fires). Setting the current state is an inert no-op.
- **result:** `{"enabled": bool}` — the effective `sameWrapAnnotationEnabled()` after the call.
- **errors:** `-32000` (wrap annotation widget unavailable), `-32602`
  (`data.param:"enabled"` — missing / non-boolean).

**`wrap_annotation.commit`**
- **params:** `{"viewer"?: str}` — optional viewer id / surface-slot. When given, commit on
  that viewer (must be a chunked volume viewer); when omitted, iterate the base viewers like
  the shift+E handler and commit on the first chunked viewer that reports success.
- Maps to `CChunkedVolumeViewer::commitSameWrapAnnotationPreview()` (the same call the
  shift+E key handler makes). Requires **same-wrap annotation mode enabled**. A commit with
  no seeded preview is a no-op that returns `committed:false`.
- **result:** `{"committed": bool, "hadPreview": bool}`.
- **errors:** `-32000` (widget/viewer-manager unavailable), `-32002` (same-wrap annotation
  mode is not enabled), `-32009` (resolved `viewer` is not a chunked volume viewer).

**`wrap_annotation.undo`**
- **params:** `{"viewer"?: str}` — same viewer resolution as `commit`.
- Maps to `CChunkedVolumeViewer::undoSameWrapAnnotation()` (the Ctrl+Z equivalent: clears an
  uncommitted preview, or undoes the last committed same-wrap collection).
- **result:** `{"undone": bool}`.
- **errors:** `-32000` (viewer manager unavailable), `-32009` (resolved `viewer` is not a
  chunked volume viewer).

`state.get` additionally reports `"sameWrapAnnotation": {"enabled": bool, "hasPreview": bool}`
(or `null` when the widget / viewer manager is unavailable), where `hasPreview` is true when
any chunked viewer currently holds an uncommitted preview.

### 3.10 `segmentation.enable_editing`

- **params:** `{"enabled": bool}`
- Maps to `SegmentationWidget::setEditingEnabled(bool)`
  (apps/VC3D/segmentation/SegmentationWidget.hpp:81); the widget's existing signal wiring
  propagates to `CWindow::onSegmentationEditingModeChanged` and
  `ViewerManager::setSegmentationEditActive`.
- **result:** `{"enabled": bool}` (the effective state after the call).
- **errors:** `-32000`, `-32007` (`data.kind:"segment"` — no active surface to edit).

### 3.11 `segmentation.grow`

Asynchronous. Maps to `CWindow::onGrowSegmentationSurface(SegmentationGrowthMethod,
SegmentationGrowthDirection, int steps, bool inpaintOnly)` (CWindow.hpp:132), executed
by `SegmentationGrower::start(...)` (segmentation/growth/SegmentationGrower.hpp:72).

- **params:**
  ```json
  {"method"?: str = "tracer",     // "tracer"|"corrections"|"patch_tracer"  (see §8.1)
   "direction"?: str = "all",     // "all"|"up"|"down"|"left"|"right"|"fill"
   "steps": int,                  // >= 1
   "inpaintOnly"?: bool = false}
  ```
  Enum mapping (segmentation/growth/SegmentationGrowth.hpp:23–66):
  `tracer=Tracer(0)`, `corrections=Corrections(1)`, `patch_tracer=PatchTracer(3)`;
  directions map in declared order `All..Fill`. **`manual_add` is not accepted here**
  — it is an interactive editing mode, not a growth invocation, and is rejected with
  `-32009 UNSUPPORTED` (amendment §8.1; use `segmentation.manual_add.*`). The C++ enum
  value `ManualAdd(4)` is unchanged; only the RPC surface drops it.
- **result:** `{"jobId": str, "kind": "segmentation.grow"}` — completion is signaled by a
  `job.progress` notification with `phase:"finished"` when
  `onSegmentationGrowthStatusChanged(false)` fires.
- **errors:** `-32000`, `-32001`, `-32004`, `-32007` (no active segmentation surface),
  `-32008` (editing not enabled), `-32009` (`method:"manual_add"`, §8.1),
  `-32602` (bad enum string / steps < 1).

### 3.11c `segmentation.save`

Asynchronous. Forces the active segment's pending autosave to disk immediately and
reports the flush as a job. Maps to `SegmentationModule::flushAutosave()`
(`markAutosaveNeeded(true)` -> `performAutosave()`); the underlying save runs in a
`QtConcurrent` worker and completion is signaled by `SegmentationModule::autosaveCompleted(bool)`.

- **params:** `{}` (none; any params are ignored).
- **result (nothing to flush):** when no save is pending and none is in flight — or when
  the flush cannot start a disk write (no resolvable surface / surface missing file
  metadata) — an idle body:
  ```json
  {"jobId": null, "kind": "segmentation.save", "state": "idle",
   "pending": bool, "saveInProgress": false, "dirtyAfterSave": bool}
  ```
- **result (save running):** when a save is pending or already in flight, the flush is
  registered as a `source:"autosave"` job and the running `job.status` body is returned
  (the `jobStatusJson` shape of §3.17: `{"jobId", "source":"autosave",
  "kind":"segmentation.save", "state":"running", ...}`). The job is closed by a
  `job.progress` notification with `phase:"finished"` (`success:true` on a successful disk
  write, `false` on save failure) when `autosaveCompleted(bool)` fires. A second explicit
  save while one is running is rejected with `-32004` (`data.source:"autosave"`).
- **errors:** `-32000` (segmentation module unavailable), `-32004` (an `"autosave"` job is
  already running).

Only bridge-initiated explicit saves are tracked as `"autosave"` jobs; the periodic /
edit-driven autosaves do not create jobs.

### 3.12 `segmentation.grow_patch_from_seed`

Asynchronous; headless twin of the "Create Segment (GrowPatch)" dialog flow. See §4 for
the interactive/headless split. Runs `vc_grow_seg_from_seed` via
`CommandLineToolRunner::executeCustomCommand`.

- **params:**
  ```json
  {"seed": Vec3,                 // volume-space seed point (required)
   "volumeId"?: str,             // vpkg volume id; default: current volume id
   "iterations"?: int = 200,     // "generations" param, 1..100000
   "minAreaCm"?: float = 0.002,  // min_area_cm, >= 0
   "outputDir"?: str}            // absolute, or relative to the volpkg root;
                                 // default: same choice list head the dialog uses
  ```
- **result:** `{"jobId": str, "kind": "segmentation.grow_patch_from_seed",
   "outputDir": str, "volumeId": str}`
  Terminal `job.progress` (`phase:"finished"`) carries `success` and, on success, the
  refreshed segment list is observable via `segments.list`.
- **errors:** `-32000`, `-32001`, `-32004`, `-32005` (process failed to start),
  `-32006` (`vc_grow_seg_from_seed` not found), `-32007` (`data.kind:"volume"` — bad
  `volumeId`), `-32005` with `data.detail` for normal-grid-missing / output-dir-create
  failures (mirroring the dialog path's validation messages), `-32602`.

### 3.13 `points.commit`

Add annotation points to a named collection in the live `VCCollection`
(`CState::pointCollection()`; `PointCollections::addPoint/addPoints`,
core/include/vc/core/PointCollections.hpp). The collection is created if absent
(same semantics as `addPoint(collectionName, ...)`).

- **params:**
  ```json
  {"collection": str,            // collection name (required, non-empty)
   "points": [Vec3],             // >= 1 point, volume space
   "winding"?: float}            // optional winding_annotation applied to each point
  ```
- **result:**
  ```json
  {"collectionId": int, "pointIds": [int]}   // uint64 ids as JSON numbers
  ```
  (`winding` set via `updatePoint` on each returned `ColPoint` when provided.)
- **errors:** `-32000`, `-32602` (empty name / empty points / non-finite coords).

### 3.14 `points.list`

- **params:** `{"collection"?: str}` — omit to list all collections.
- **result:**
  ```json
  {"collections": [
    {"id": int, "name": str, "color": [float, float, float],
     "points": [
       {"id": int, "position": Vec3, "winding": float | null}
     ]}
  ]}
  ```
  From `PointCollections::getAllCollections()` / `getPoints(name)`;
  `winding` is `null` when `winding_annotation` is NaN.
- **errors:** `-32007` (`data.kind:"collection"` — named collection absent).

### 3.15 `volume.open`

Open a volume package, optionally selecting a specific volume in it.

- **params:**
  ```json
  {"path": str,          // .volpkg dir, .volpkg.json, or zarr project path
   "volumeId"?: str}     // switch current volume after load
  ```
- Maps to `MenuActionController::openVolpkgAt(const QString&)`
  (MenuActionController.hpp:41). The call is synchronous from the bridge's point of view
  (openVolpkgAt runs load on the GUI thread); the response is sent after it returns.
- **result:** `{"opened": true, "vpkgPath": str, "volumeId": str,
   "volumeIds": [str]}`
- **errors:** `-32005` (`data.detail` — load failed / path missing),
  `-32007` (`data.kind:"volume"` — `volumeId` not in the package).

### 3.16 `catalog.open_sample`

Open an Open Data catalog sample by manifest id — the headless twin of double-clicking a
sample in the catalog window.

- **params:** `{"sampleId": str}`
- Implementation contract: fetch/lookup via `OpenDataManifest::findSample(id)`
  (apps/VC3D/OpenDataManifest.hpp:167), then call a **new public wrapper**
  `bool MenuActionController::openOpenDataSampleById(const QString& sampleId)` that
  resolves the sample and forwards to the existing private
  `openOpenDataSample(const vc3d::opendata::OpenDataSample&)`
  (MenuActionController.hpp:91). Do not make the bridge a friend of
  MenuActionController; the wrapper is the supported surface.
- **result:** `{"opened": true, "sampleId": str, "vpkgPath": str,
   "volumeIds": [str]}`
- **errors:** `-32007` (`data.kind:"sample"`), `-32005` (manifest fetch / open failed).

### 3.17 `job.status`

- **params:** `{"jobId"?: str}` — omit for "the current/most recent job".
- **result:**
  ```json
  {"jobId": str, "kind": str, "label": str,
   "state": "running" | "succeeded" | "failed",
   "message": str,                 // last status/finish message
   "outputPath": str | null,      // from toolFinished when applicable
   "consoleTail": [str]}          // last <=50 console lines (consoleOutputReceived)
  ```
  The bridge retains the last **8** completed job records for late polling.
- **errors:** `-32007` (`data.kind:"job"` — unknown id and no job has ever run).

### 3.18 `job.progress` — server-push notification (no `id`)

Broadcast to all connected clients. Never sent as a response.

```json
{"jsonrpc": "2.0", "method": "job.progress",
 "params": {
   "jobId": str,
   "kind": str,                       // "segmentation.grow", "segmentation.grow_patch_from_seed", ...
   "phase": "started" | "output" | "finished",
   "message"?: str,                   // toolStarted message / console line chunk
   "success"?: bool,                  // present iff phase == "finished"
   "outputPath"?: str                 // present iff phase == "finished" and applicable
 }}
```

Sources: `CommandLineToolRunner::toolStarted` → `started`;
`consoleOutputReceived` → `output` (rate-limited to ≤10 notifications/sec, coalescing
lines); `toolFinished` → `finished`. For in-process growth,
`onSegmentationGrowthStatusChanged(true/false)` → `started` / `finished`.

### 3.19 `segments.fetch` — materialize an open-data placeholder segment

Segments attached from the Open Data catalog are **lazy placeholders** holding
only metadata (`isOpenDataSegmentPlaceholder`); `segments.activate` (§17.3)
refuses them with `-32005` / `data.detail` ending `"…placeholder; fetch it
first"`. This RPC downloads the actual surface data — the headless counterpart
to the GUI's fetch-on-click (`SurfacePanelController::fetchOpenDataSegmentAsync`,
which reuses `materializeOpenDataSegment` and, on success, reloads the surface
list so the segment becomes activatable). It does **not** activate the segment.

- **params:** `{"segmentId": str}` — an id as returned by `segments.list`.
- **result (already materialized — synchronous):**
  ```json
  {"fetched": true, "alreadyMaterialized": true,
   "segment": {"id": str, "path": str, "placeholder": false}}
  ```
- **result (placeholder — async, SPEC §18.4 job):**
  ```json
  {"jobId": str, "source": "catalog", "fetched": false,
   "alreadyMaterialized": false,
   "segment": {"id": str, "path": str, "placeholder": true}}
  ```
  Progress and completion arrive as `job.progress` (§3.18) on the `"catalog"`
  source; the terminal state is pollable via `job.status`.
- **concurrency:** runs on the `"catalog"` job source and shares
  `SurfacePanelController`'s single-flight materialize guard, so it is rejected
  with `-32004` while a `catalog.open_sample`, another `segments.fetch`, or a
  GUI-initiated fetch is in flight.
- **errors:** `-32602` (missing `segmentId`); `-32007` (`data.kind:"segment"` —
  unknown id); `-32004` (`data.source:"catalog"` — a fetch is already running);
  `-32000` (no volume package loaded); `-32010` (surface panel unavailable).

---

## 4. Interactive/headless split: `onCreateSegmentGrowPatchFromSeed`

Current reality (`SegmentationCommandHandler.cpp:2518–2790`): the slot
`onCreateSegmentGrowPatchFromSeed(const QVector3D& seedPoint)` does, in one block:

1. Preconditions: vpkg loaded, current volume set, `_cmdRunner` present and not
   `isRunning()`, no `_growPatchSeedJob` active, `vc_grow_seg_from_seed` findable.
2. Builds volume option list (`buildVolumeOptionList`), open-data patches roots, normal
   grid path, volpkg root, output-dir choice list.
3. Calls the modal dialog `selectGrowPatchSeedParams(...)`
   (SegmentationCommandHandler.cpp:272–417) which collects: **volume (id+path),
   iterations (1..100000, default 200), min size `minAreaCm` (default 0.002), output
   folder** — then `dlg.exec()`.
4. Validates/creates the output dir, registers the segments entry, writes the params
   JSON temp file (`mode:"seed"`, `step_size:20`, `min_area_cm`, `generations`,
   `thread_limit:1`, `normal_grid_path`, `cache_root`, optional `voxelsize`,
   optional `normal3d_zarr_path`), connects a one-shot `toolFinished` lambda
   (meta.json coordinate-identity fixup, `refreshSegmentations`,
   `reloadSurfacesFromDisk`), and launches `executeCustomCommand`.

**Binding refactor** in `SegmentationCommandHandler.{hpp,cpp}` — steps 1, 2, 4 move into
a shared non-interactive method; step 3 (the dialog) stays only in the slot. No logic is
duplicated: the slot becomes "gather defaults → dialog → call the headless method",
and dialogs/`QMessageBox`es remain exclusively in the interactive path.

New public types/method on `SegmentationCommandHandler`:

```cpp
struct GrowPatchSeedParams {
    QString volumeId;          // vpkg volume id; empty => current volume
    int     iterations{200};   // clamped/validated to [1, 100000]
    double  minAreaCm{0.002};  // >= 0
    QString outputDir;         // absolute or relative to volpkg root; empty => default
                               // head of the same choice list the dialog shows
};

/// Headless GrowPatch-from-seed launch. Performs ALL validation and execution that
/// onCreateSegmentGrowPatchFromSeed performs today, but reports failures through
/// `errorMessage` (never via QMessageBox) and never opens a dialog. Returns true when
/// the vc_grow_seg_from_seed process was started (job accepted), false otherwise.
/// On success, completion is observable via CommandLineToolRunner::toolFinished
/// exactly as in the interactive path (same one-shot handler: meta.json coordinate
/// identity fixup, VolumePkg::refreshSegmentations, surface panel reload).
bool startGrowPatchFromSeed(const QVector3D& seedPoint,
                            const GrowPatchSeedParams& params,
                            QString* errorMessage = nullptr);
```

The refactored slot:

```cpp
void SegmentationCommandHandler::onCreateSegmentGrowPatchFromSeed(const QVector3D& seedPoint)
{
    // 1. build defaults (volume options, remembered QSettings volume id,
    //    output choices) exactly as today;
    // 2. selectGrowPatchSeedParams(...) — unchanged dialog;
    // 3. persist growpatch_seed/volume_id to QSettings;
    // 4. GrowPatchSeedParams p{...from dialog...};
    //    QString err;
    //    if (!startGrowPatchFromSeed(seedPoint, p, &err))
    //        QMessageBox::warning(_parentWidget, tr("Error"), err);
}
```

Notes binding the implementation:

- The QSettings read/write of `growpatch_seed/volume_id` is interactive-path-only
  (agent calls must not clobber the user's remembered dialog default).
- `startGrowPatchFromSeed` resolves `volumeId` → path via `buildVolumeOptionList()`;
  unknown id → false with `errorMessage = "Unknown volume id: ..."` (bridge maps to
  `-32007`).
- Precondition failures map to distinct sentences so the bridge can classify:
  no vpkg / no volume / runner busy / seed-job active / tool missing / no normal grid /
  output dir creation failed. The bridge inspects the failure *before* calling (for the
  cheap ones: `CState` checks, `_cmdRunner->isRunning()`) and uses `errorMessage` text
  as `data.detail` for the rest.
- The temp params file, `_growPatchSeedJob` bookkeeping, remote-auth configuration
  (`configureCommandRunnerRemoteAuthForVolumePath`), coordinate-identity fixup, and the
  `toolFinished` one-shot connection live in `startGrowPatchFromSeed` — shared verbatim
  by both paths.
- `_cmdRunner->showConsoleOutput()` is called only from the interactive slot; the
  headless path must not pop UI.

RPC `segmentation.grow_patch_from_seed` (§3.12) is a thin adapter over
`startGrowPatchFromSeed`.

---

## 5. MCP tool surface

One MCP server process (Phase 3) exposes the tools below; each maps 1:1 onto an RPC
above unless noted. Tool names are snake_case with a `vc3d_` prefix. All coordinates in
tool params are **volume-space voxels** unless a `space` field says otherwise; the MCP
server passes params through verbatim and returns the RPC `result` as the tool result
(JSON). RPC errors surface as MCP tool errors with `code`/`message`/`data` preserved.

The MCP server takes `--socket <name-or-path>` (matching `--agent-bridge-name`) or
spawns VC3D itself with `--agent-bridge` and parses the stdout handshake line (§1.1).

| MCP tool | → RPC | Description shown to the agent |
|---|---|---|
| `vc3d_ping` | `ping` | Check the VC3D bridge is alive; returns pid, app version, and protocol version. |
| `vc3d_get_state` | `state.get` | Snapshot of VC3D: open volume package, current volume, active segment, viewers (ids/names), editing mode, running job. Call this first. |
| `vc3d_list_segments` | `segments.list` | List segments in the open volume package with loaded/active flags. |
| `vc3d_fetch_segment` | `segments.fetch` | Download ("materialize") an open-data placeholder segment so it can be activated/edited. Sync if already materialized; else a `"catalog"` job (`wait` defaults true). |
| `vc3d_screenshot` | `screenshot.capture` | Capture a PNG of the whole VC3D window or one viewer pane. Returns the PNG as MCP image content when `file_path` is omitted, or a dict with the on-disk path when `file_path` is set (MCP-layer note below). |
| `vc3d_get_cursor_point` | `canvas.get_cursor_volume_point` | Resolve a viewer scene position (or the current cursor) to a 3D volume point + surface normal. |
| `vc3d_click` | `canvas.click` | Synthesize a mouse click in a viewer at a volume-space (or scene-space) position, with button and modifiers (e.g. `{"modifiers": ["shift"]}` to place a point / set focus). |
| `vc3d_shift_click` | `canvas.shift_click` | Shift+click convenience: the canonical place-point / set-focus gesture. |
| `vc3d_center_viewer` | `viewer.center_on_point` | Center a viewer pane on a 3D volume point. |
| `vc3d_zoom_viewer` | `viewer.zoom` | Multiply a viewer's zoom by a factor (>1 zooms in). Returns the new scale. |
| `vc3d_rotate_viewer` | `viewer.rotate` | Rotate the "seg xz"/"seg yz" axis-aligned slice plane (middle-drag equivalent). Relative delta by default. |
| `vc3d_set_axis_aligned_slices` | `viewer.set_axis_aligned_slices` | Enable/disable axis-aligned slice mode (checkbox equivalent) — prerequisite for `viewer.rotate`. |
| `vc3d_set_wrap_annotation_mode` | `wrap_annotation.set_mode` | Enable/disable "Same-wrap annotation mode" (prerequisite for the shift+E commit workflow; seed the preview via `vc3d_shift_click`). |
| `vc3d_commit_wrap_annotation` | `wrap_annotation.commit` | Commit the seeded same-wrap annotation preview into the point collection (the tutorial's shift+E). Requires the mode enabled. |
| `vc3d_undo_wrap_annotation` | `wrap_annotation.undo` | Undo the same-wrap annotation (Ctrl+Z equivalent): clear the preview or undo the last committed collection. |
| `vc3d_enable_editing` | `segmentation.enable_editing` | Turn segmentation editing mode on/off for the active segment. |
| `vc3d_save_segment` | `segmentation.save` | Force the active segment's pending autosave to disk. Idle no-op (`jobId:null`) when nothing is dirty; else an `"autosave"` job (`wait` defaults true). |
| `vc3d_grow_segment` | `segmentation.grow` | Grow the active segmentation surface (method: tracer/corrections/patch_tracer; direction; steps). Async: returns a jobId. |
| `vc3d_grow_patch_from_seed` | `segmentation.grow_patch_from_seed` | Create a brand-new segment by growing a patch from a 3D seed point (headless GrowPatch). Async: returns a jobId and outputDir. |
| `vc3d_commit_points` | `points.commit` | Add annotation points (volume space) to a named collection, optionally with a winding annotation. |
| `vc3d_list_points` | `points.list` | List point collections and their points. |
| `vc3d_open_volume` | `volume.open` | Open a volume package (.volpkg / .volpkg.json / zarr project) and optionally select a volume id. |
| `vc3d_open_catalog_sample` | `catalog.open_sample` | Open an Open Data catalog sample by its manifest sample id. |
| `vc3d_job_status` | `job.status` | Poll a job by id (or the latest job): state, message, console tail. |

Parameter schemas: each tool's `inputSchema` is the JSON-Schema rendering of the
corresponding RPC params block in §3, with the same names, types, defaults, and enums
(e.g. `vc3d_grow_segment.method` is
`{"type":"string","enum":["tracer","corrections","patch_tracer"],"default":"tracer"}`
— `manual_add` dropped per §8.1).
The MCP server performs no semantic validation beyond schema — the bridge is the
authority.

`job.progress` notifications: the MCP server ignores them — it reads them off the socket
and discards them. `wait: true` and progress forwarding are implemented instead by polling
the authoritative `job.status` RPC (§3.17). `wait` is an MCP-server-side convenience param
(not part of the RPC) available on the long-running job-returning tools; when `"wait": true`
it polls `job.status` once a second until the job is terminal and returns the terminal
`job.status` inline. Along the way the server forwards newly-appended `consoleTail` lines to
the MCP client as **best-effort** progress via the FastMCP `Context` (`ctx.report_progress`,
monotonic sequence number) where the client supports it; forwarding is observational only
and never affects job execution or the returned result. `wait` defaults to `false`; when
`true`, the server enforces a 30-minute cap and returns the still-running `jobId` (with
`"waitTimedOut": true`) on timeout, and fails promptly on bridge disconnect (the next poll
raises).

`vc3d_screenshot` return shape (MCP-layer only; the §3.4 RPC contract is untouched): the
`screenshot.capture` RPC still returns `base64` when `filePath` was omitted, else writes the
PNG to disk. On top of that the MCP server adds a presentation layer — when `file_path` is
omitted it decodes the RPC's `base64` and returns the PNG as FastMCP **image content** (an
`Image`) so MCP hosts render it inline; when `file_path` is set it returns the dict carrying
the on-disk `filePath` (no inline base64). This conversion lives entirely in the MCP server.

`vc3d_wait_job` (MCP-only; no underlying RPC): the counterpart to the `wait` param above for
a job the current tool call did **not** itself start. It blocks on an **already-running** job
by id using the identical `job.status` polling loop (once/sec, 30-minute cap, `consoleTail`
forwarded as best-effort progress via the FastMCP `Context`) and returns the terminal
`job.status` record inline, or the last-seen status with `"waitTimedOut": true` on the cap.
Use it to wait on a job started earlier with `wait=false`, or on an externally-initiated job
seen in `state.get`. See §21.3.

---

## 6. Implementation placement (for later phases)

- `apps/VC3D/agent_bridge/AgentBridge.{hpp,cpp}` — QLocalServer, framing, dispatch,
  viewer registry, job tracker. Constructed in `VCAppMain.cpp` after `CWindow` (§1.1);
  `friend class AgentBridge;` added in `CWindow.hpp` beside `RenderBenchReplay`.
- `SegmentationCommandHandler`: refactor per §4 (this is the only behavioral change to
  existing files beyond friend/wrapper additions).
- `MenuActionController`: add public `openOpenDataSampleById(const QString&)` (§3.16).
- MCP server: separate process under `tools/vc3d-mcp/` (Phase 3), stdio MCP
  transport, QLocalSocket (or platform-equivalent) client to the bridge.

---

# Part 2: v2 Command Surface

Status: **binding design** (v2 Stage 0). Everything in Part 1 remains in force; Part 2
only extends it. Where Part 2 amends a Part 1 statement, the amendment is explicit and
Part 2 wins. All grounding references are to real code on branch
`feature/vc3d-agent-bridge` as of this design pass (file:line numbers verified).

Section numbering continues from Part 1 (§7 onward).

---

## 7. v2 scope and compatibility

- All v1 methods keep their exact wire behavior except the two amendments in §8.1/§8.2
  and the additive job-model fields in §8.3 (new fields only; no v1 field is removed or
  retyped).
- New RPC families: `canvas.drag`, `segmentation.manual_add.*`,
  `segmentation.corrections.*`, `catalog.list_samples` / `catalog.describe_sample`,
  `volume.select`, `lasagna.*`, `workspace.switch`, `atlas.*`, `fiber.*`, and the
  Stage 6 backlog surface (`tags.set`, `seeding.*`, `segmentation.push_pull.*`,
  `tracer.run_trace`).
- No new error codes. v2 reuses the §2.5 table verbatim; error-specific `data` fields
  are documented per method.

---

## 8. v2 common conventions

### 8.1 Footgun fix: `segmentation.grow` rejects `"manual_add"`

Amendment to §3.11. The documented `method` enum becomes:

```
"method"?: str = "tracer"      // "tracer" | "corrections" | "patch_tracer"
```

`method: "manual_add"` → error `-32009 UNSUPPORTED` with
`data.detail: "manual_add is not a growth method; use segmentation.manual_add.begin/finish"`.

Rationale: manual-add is an interactive editing *mode*, not a grow invocation; passing
`SegmentationGrowthMethod::ManualAdd` into `CWindow::onGrowSegmentationSurface` from the
bridge would bypass the mode's session state. The C++ enum value
`SegmentationGrowthMethod::ManualAdd` (segmentation/growth/SegmentationGrowth.hpp:23–66)
is **not** removed — the in-app manual-add apply path still uses it internally. Only the
RPC enum (and the corresponding MCP `vc3d_grow_segment` inputSchema enum, §5) drops it.

### 8.2 Footgun fix: `catalog.open_sample` is always non-interactive

Amendment to §3.16. `MenuActionController::openOpenDataSample` shows a **blocking**
replace-project `QMessageBox` (`prompt.exec()`) when a vpkg is already open
(MenuActionController.cpp:518–532, `exec()` at line 527). A bridge handler must never
spin a nested event loop (§1.3), so the bridge never reaches that code path.

Binding contract: the bridge treats an explicit agent `catalog.open_sample` call as
consent to replace the current project. It calls a non-interactive entry point (§10.3,
§14.1) that (a) skips the replace-project prompt entirely and (b) does not create the
segment-download `QProgressDialog` (MenuActionController.cpp:539–546). The existing
public `MenuActionController::openOpenDataSampleById(const QString&)`
(MenuActionController.hpp:48) remains the interactive path; the bridge uses the new
options overload from §14.1 exclusively.

### 8.3 Generalized source-tagged job model

Amendment to §2.4. The single-active-job model becomes **one active job per source**,
with four sources:

| `source`    | Lifecycle authority |
|-------------|---------------------|
| `"tool"`    | `CommandLineToolRunner::toolStarted/toolFinished/consoleOutputReceived` (CommandLineToolRunner.hpp:119–122) |
| `"growth"`  | `CWindow::onSegmentationGrowthStatusChanged(bool)` (CWindow.hpp:274) / `SegmentationGrower::running()` |
| `"lasagna"` | `LasagnaServiceManager` signals: `optimizationStarted`, `optimizationProgress`, `jobStarted`, `jobFinished`, `jobError`, `optimizationFinished`, `optimizationError` (LasagnaServiceManager.hpp:104–121) |
| `"atlas"`   | new `CWindow::atlasSearchProgressChanged` / `atlasSearchFinished` signals (§14.3) |

**JobRecord** (the canonical shape; `job.status` results are exactly this object):

```json
{
  "jobId": "job-<n>",              // monotonically increasing, all sources share one counter
  "source": "tool" | "growth" | "lasagna" | "atlas",
  "kind": str,                     // e.g. "segmentation.grow", "lasagna.optimize", "atlas.fiber_search"
  "label": str,
  "state": "running" | "succeeded" | "failed",
  "message": str,                  // last status/finish message
  "outputPath": str | null,
  "externalId": str | null,        // lasagna service job id when known, else null
  "consoleTail": [str],            // last <=50 lines; [] for sources without console output
  "startedAtMs": int,              // ms since epoch
  "finishedAtMs": int | null
}
```

- Starting a job-producing RPC while a job **of the same source** is active →
  `-32004 JOB_RUNNING` with `data: {"jobId": str, "source": str}`. Jobs of *different*
  sources may run concurrently (matching reality: a lasagna optimization and an external
  tool can and do run simultaneously).
- The bridge also registers **externally initiated** jobs: when a lifecycle-authority
  signal reports a start the bridge did not request (a human clicked a button), the
  bridge creates a JobRecord for it (kind `"<source>.external"`), so `-32004` and
  `state.get` always reflect true app state.
- Retention: the last **8 completed** JobRecords **per source** (amends v1's 8 global).
- `job.status` params become `{"jobId"?: str, "source"?: str}` — with `jobId` omitted,
  returns the most recently started job, filtered to `source` when given. Unknown
  `jobId` → `-32007` (`data.kind:"job"`).
- `job.progress` notifications (§3.18) gain a required `"source"` field in `params`.
- `state.get` (§3.2): the `"job"` field keeps its v1 meaning (most recently started
  active job, else `null`) for compatibility, and a new field
  `"jobs": [JobRecord]` lists **all currently active** jobs (0–4 entries).

### 8.4 Deferred RPC responses

New bridge-core capability for RPCs whose underlying app API replies via a Qt signal
rather than a return value (first users: `lasagna.list_datasets` over
`LasagnaServiceManager::fetchDatasets()` → `datasetsReceived(QJsonArray)`
(LasagnaServiceManager.hpp:91, 127), `lasagna.jobs` over `fetchJobs()` → `jobsUpdated`
(LasagnaServiceManager.hpp:76, 118), `lasagna.ensure_service` external mode over
`serviceStarted`/`serviceError` (LasagnaServiceManager.hpp:104–106), and
`fiber.create_atlas` over `LineAnnotationController::atlasCreated`).

Mechanism (bridge-internal, binding for implementers):

1. The handler invokes the app API, stashes `(QPointer<QLocalSocket>, requestId)` in a
   pending-deferred table, and returns **without** writing a response.
2. A one-shot signal connection (disconnected on first delivery) completes the entry:
   the bridge serializes the result and writes the JSON-RPC response with the stashed
   `id` to the stashed socket (skipped silently if the socket has disconnected).
3. A per-entry `QTimer` enforces the method's documented timeout. On timeout the bridge
   disconnects the one-shot connection and responds
   `-32005 JOB_FAILED` with `data.detail: "timed out after <t> ms waiting for <signal>"`.

Observable contract (what clients may rely on):

- Every deferred method documents a **max latency** (its timeout). Defaults: **10 s**
  unless a method says otherwise. A response (result or `-32005`) is always sent within
  timeout + one event-loop tick.
- Responses to deferred methods may arrive **out of order** relative to later
  non-deferred requests on the same connection (amends the FIFO note in §1.2 for these
  methods only). Clients must correlate by JSON-RPC `id`.
- At most one deferred call per (connection, method) may be in flight; a second call to
  the same deferred method from the same connection while one is pending →
  `-32004` with `data: {"detail": "deferred call already pending", "method": str}`.

---

## 9. v2 method reference — canvas and segmentation editing

### 9.1 `canvas.drag`

Synthesize a full press–move–release drag through the viewer's public mouse slots
`onMousePress` / `onMouseMove` / `onMouseRelease`
(volume_viewers/CChunkedVolumeViewer.hpp:231–233; note `onMouseMove` takes
`Qt::MouseButtons` plural).

- **params:**
  ```json
  {"viewer"?: str,                       // §2.2, default "segmentation"
   "from": Vec3 | {"x": float, "y": float},
   "to":   Vec3 | {"x": float, "y": float},
   "space"?: "volume" | "scene",         // default "volume"; applies to both endpoints
   "button"?: str = "left",              // §2.3, plus "none" (see below)
   "modifiers"?: [str] = [],             // §2.3
   "steps"?: int = 8}                    // interpolated move events, clamped to [1, 256]
  ```
  Coordinate validation reuses §3.6 conventions exactly: with `space:"volume"`, both
  endpoints are converted via `volumeToScene()` and round-trip-verified
  (`sceneToVolume` within 2.0 voxels) — either endpoint failing → `-32003` with
  `data.point` naming the offender (`"from"` / `"to"`).

  Dispatch order: `onMousePress(sceneFrom, button, modifiers)` → for `i = 1..steps`:
  `onMouseMove(lerp(sceneFrom, sceneTo, i/steps), buttonsOf(button), modifiers)` →
  `onMouseRelease(sceneTo, button, modifiers)`. No `onVolumeClicked` is sent (a drag is
  not a click).

  `button: "none"` (canvas.drag only): the press and release are **skipped**; only the
  `steps` move events are dispatched with `Qt::NoButton`. This is the hover-positioning
  primitive (the segmentation module records pointer samples from buttonless moves —
  SegmentationModule_Input.cpp:715–717) used by `segmentation.push_pull.*` (§15.3).
- **result:**
  ```json
  {"dragged": true,
   "from": {"scene": {"x": float, "y": float}, "volumePoint": Vec3 | null},
   "to":   {"scene": {"x": float, "y": float}, "volumePoint": Vec3 | null},
   "steps": int, "button": str, "modifiers": [str]}
  ```
  (`volumePoint` from `sampleSceneVolume` at each endpoint, `null` when off-surface.)
- **errors:** `-32001`, `-32002`, `-32003`, `-32009` (non-chunked viewer), `-32602`
  (bad enum, non-finite coords, steps out of range after clamping is *not* an error —
  clamp silently; steps < 1 or non-integer → `-32602`).

### 9.2 `segmentation.manual_add.begin`

Enter manual-add (hole-fill) mode on the active editing session. Maps to
`SegmentationModule::beginManualAdd()` (segmentation/SegmentationModule.hpp:321) — the
same entry the ManualAddToggle key uses (SegmentationModule_Input.cpp:76–87). The
bridge reaches it through a new **public** wrapper
`bool SegmentationModule::setManualAddModeActive(bool active, bool apply = true)`
(delegating to `beginManualAdd()` / `finishManualAdd(apply)`), since the underlying
methods sit in the module's private section.

- **params:** none (`{}`).
- **result:** `{"active": true}`. Idempotent: if manual-add mode is already active
  (`SegmentationModule::manualAddMode()`, SegmentationModule.hpp:186), returns
  `{"active": true}` without re-entering.
- **errors:** `-32008` (segmentation editing not enabled), `-32007`
  (`data.kind:"session"` — no active edit session, i.e.
  `_editManager->hasSession()` false), `-32004` (`data.source:"growth"` — growth in
  progress), `-32000`, `-32001`.

Constraint placement while the mode is active needs **no new RPC**: it reuses
`canvas.click` / `canvas.shift_click` (§3.6/§3.7). Shift+Left-click adds/replaces a
plane constraint, Shift+Right-click removes the nearest one — routed through
`SegmentationModule::handleManualAddMousePress` (SegmentationModule.hpp:327–331) into
`ManualAddTool::addOrReplacePlaneConstraint` / `removePlaneConstraintNear`
(segmentation/tools/ManualAddTool.hpp:82–84).

### 9.3 `segmentation.manual_add.finish`

Leave manual-add mode. Maps to `SegmentationModule::finishManualAdd(bool apply)`
(segmentation/SegmentationModule.hpp:322) via the §9.2 wrapper.

- **params:** `{"apply"?: bool = true}` — `true` commits the preview (which may hand a
  tracer mask to the growth path via `applyManualAddTracerPreview` /
  `takePendingManualAddTracerMask`, SegmentationModule.hpp:187–188); `false` discards
  (`resetManualAddState`, SegmentationModule.hpp:323).
- **result:** `{"applied": bool}` — the boolean returned by `finishManualAdd`. If the
  apply triggers an in-process growth run, its lifecycle is observable as a
  `source:"growth"` job (§8.3); the RPC itself returns immediately after
  `finishManualAdd` returns.
- **errors:** `-32007` (`data.kind:"manual_add_session"` — manual-add mode not active),
  `-32000`, `-32001`.

### 9.4 `segmentation.manual_add.set_line_mode`

- **params:** `{"mode": str}` — enum mapping to `ManualAddTool::LinePreviewMode`
  (segmentation/tools/ManualAddTool.hpp:15–21):
  `"vertical"=VerticalOnly(0)`, `"horizontal"=HorizontalOnly(1)`, `"cross"=Cross(2)`,
  `"cross_fill"=CrossFill(3)`.
- Backing: a new **public** setter
  `SegmentationWidget::setManualAddLinePreviewMode(ManualAddTool::LinePreviewMode)`
  delegating to a matching new setter on `SegmentationManualAddPanel` (which today only
  exposes `cycleLinePreviewMode()`, segmentation/panels/SegmentationManualAddPanel.hpp:22).
  The panel setter updates its UI control and emits the existing
  `manualAddConfigChanged` wiring (SegmentationWidget.hpp:242), so the module receives
  the new `ManualAddTool::Config` through the same path a human change takes.
- **result:** `{"mode": str}` (the effective mode). Callable whether or not manual-add
  mode is currently active (config persists, matching the panel).
- **errors:** `-32602` (unknown mode string), `-32000`.

### 9.5 `segmentation.manual_add.set_interpolation`

- **params:** `{"mode": str}` — enum mapping to `ManualAddTool::InterpolationMode`
  (segmentation/tools/ManualAddTool.hpp:23–27):
  `"thin_plate_spline"=ThinPlateSpline(0)`,
  `"tracer_restricted_to_fill"=TracerRestrictedToFill(1)`.
- Backing: same pattern as §9.4 — new public
  `SegmentationWidget::setManualAddInterpolationMode(ManualAddTool::InterpolationMode)`
  → panel setter → `manualAddConfigChanged`.
- **result:** `{"mode": str}`.
- **errors:** `-32602`, `-32000`.

### 9.6 `segmentation.manual_add.undo_constraint`

Remove the most recently placed user plane constraint. Maps to
`SegmentationModule::undoManualAddPlaneConstraint()`
(segmentation/SegmentationModule.hpp:326) → `ManualAddTool::removeLastPlaneConstraint`
(segmentation/tools/ManualAddTool.hpp:84), exposed via a new public wrapper
`bool SegmentationModule::undoManualAddConstraint()`.

- **params:** none.
- **result:** `{"undone": bool}` — `false` when there was no user constraint to remove
  (not an error).
- **errors:** `-32007` (`data.kind:"manual_add_session"` — mode not active).

### 9.7 `segmentation.corrections.set_point_mode`

Wraps the G-key correction-point mode: the private flag
`SegmentationModule::_correctionDragKeyActive` (segmentation/SegmentationModule.hpp:407),
set on G press and cleared on G release
(SegmentationModule_Input.cpp:69–75, 379–386). New **public** method
`bool SegmentationModule::setCorrectionPointMode(bool active, QString* errorMessage = nullptr)`
that enforces the same preconditions as the key handler (editing enabled, edit session
present, no growth in progress — SegmentationModule_Input.cpp:70) and sets the flag.

- **params:** `{"active": bool}`
- **result:** `{"active": bool}` (effective state).
- **errors:** `-32008` (editing not enabled), `-32007` (`data.kind:"session"`),
  `-32004` (`data.source:"growth"` — growth in progress), `-32000`, `-32001`.

Unlike the physical key, the mode is **not** auto-cleared on mouse release — it stays
set until `set_point_mode {active:false}`. Agents must switch it off when done, since
while active every plain left-click on the surface commits correction points.

**Verified interaction semantics** (binding; grounded in
SegmentationModule.cpp:1588–1687 and SegmentationModule_Input.cpp:489–505, 798–802):

- Plain Left press over the surface (no Shift/Ctrl/Alt) with the mode active begins a
  correction drag anchored at the grid cell under the cursor
  (`beginCorrectionDrag`, SegmentationModule.cpp:1588). Left release finishes it
  (`finishCorrectionDrag`, called at SegmentationModule_Input.cpp:801).
- **A zero-length drag (plain click, from == to) DOES commit a correction point**:
  `updateCorrectionDrag` only sets `moved = true` when the cursor travels more than
  1.0 voxel from the press point (SegmentationModule.cpp:1609–1613), and
  `finishCorrectionDrag` with `!didMove` falls back to
  `handleCorrectionPointAdded(targetWorld)` — a single, un-anchored correction point,
  with **no** automatic solver run (SegmentationModule.cpp:1633–1638). So
  `canvas.click` (plain left) suffices for single-point corrections.
- A drag longer than 1.0 voxel commits an **anchored** correction (`anchor2d` set to the
  press-point grid cell) and then **immediately triggers the corrections solver**:
  `handleGrowSurfaceRequested(Corrections, All, 0, false)`
  (SegmentationModule.cpp:1656–1677). Use `canvas.drag` for this; the resulting solver
  run appears as a `source:"growth"` job — poll `job.status` before issuing further
  editing RPCs.
- Press positions not over the segmentation surface are silently rejected with a status
  message (SegmentationModule_Input.cpp:497–503); the `canvas.click`/`canvas.drag`
  result cannot distinguish this, so agents should confirm placement via `points.list`
  on the corrections collection when certainty is required.

### 9.8 `state.get` additions (Stage 2)

Amendment to §3.2. The `state.get` result gains four fields reporting manual-add and
correction-point-authoring state (as-built during Stage 2):

```json
{
  "manualAddMode": bool,                 // SegmentationModule::manualAddMode()
  "manualAddLineMode": str | null,       // "vertical"|"horizontal"|"cross"|"cross_fill"
  "manualAddInterpolation": str | null,  // "thin_plate_spline"|"tracer_restricted_to_fill"
  "correctionsPointMode": bool           // SegmentationModule::correctionPointMode()
}
```

`manualAddLineMode` / `manualAddInterpolation` come from `SegmentationWidget::
manualAddConfig()` (the persisted panel config, valid whether or not manual-add mode is
active); they are `null` only when the segmentation widget does not exist. `manualAddMode`
and `correctionsPointMode` are `false` when the segmentation module is absent.

The result also gains an `autosave` object reporting the explicit-save bookkeeping
(SPEC §3.11c), mirrored from `SegmentationModule::autosaveStatus()`:

```json
{
  "autosave": {                 // null when the segmentation module is absent
    "pending": bool,            // a deferred save is queued (edits not yet on disk)
    "saveInProgress": bool,     // a QtConcurrent save is currently running
    "dirtyAfterSave": bool      // more edits arrived while a save was in flight
  }
}
```

**As-built clarification to §9.2:** the documented precondition errors (`-32008`,
`-32007 data.kind:"session"`, `-32004 data.source:"growth"`) are checked in the handler
before entering the mode. `beginManualAdd()` can still return `false` for a residual
reason those checks do not cover — pending unapplied edits, an autosave in progress, or an
unreadable active surface (SegmentationModule.cpp `beginManualAdd`). In that case the
handler returns `-32005 JOB_FAILED` with
`data.detail:"manual-add mode could not start (pending edits, save in progress, or unreadable surface)"`
(no new error code; reuses the §2.5 table).

---

## 10. Remote catalog resource selection

### 10.1 `catalog.list_samples`

- **params:** `{"refresh"?: bool = false}`
- Sourced from the Open Data manifest (default URL
  `vc3d::opendata::kDefaultManifestUrl`, OpenDataManifest.hpp:16–17). The bridge reuses
  the same manifest acquisition/caching path `MenuActionController` uses for
  `openOpenDataSampleById`; `refresh: true` forces a re-fetch. If the fetch is
  asynchronous in the app, the call uses the deferred mechanism (§8.4) with a **30 s**
  timeout.
- **result:**
  ```json
  {"manifestUrl": str,
   "samples": [
     {"id": str, "type": str, "description": str,
      "volumeCount": int, "segmentCount": int, "scanCount": int}
   ]}
  ```
  From `OpenDataManifest::samples` and the `OpenDataSample` counters
  (OpenDataManifest.hpp:118–135).
- **errors:** `-32005` (manifest fetch/parse failed; `data.detail`).

### 10.2 `catalog.describe_sample`

- **params:** `{"sampleId": str, "refresh"?: bool = false}`
- **result:**
  ```json
  {"sampleId": str, "type": str, "description": str,
   "volumes": [
     {"id": str, "scanId": str, "shapeZYX": [int, int, int] | null,
      "pixelSizeUm": float | null, "dataFormat": str}
   ],
   "representations": [
     {"ref": str,                       // "<volumeIndex>:<artifactIndex>", stable within one manifest fetch
      "volumeId": str,                  // sample.volumes[volumeIndex].id
      "artifactType": str,              // raw artifact type string
      "kind": "normal_grids" | "lasagna" | "prediction",
      "url": str | null,                // artifact.resolvedUrl when present
      "targetVolumeId": str | null, "modelId": str | null}
   ],
   "segmentCount": int}
  ```
  `representations` comes from `derivedRepresentations(sample)`
  (OpenDataManifest.cpp:619–651), whose kind classification (normal-grid / lasagna /
  prediction incl. 3D ink detection) is the single authority; `kind` strings map
  `OpenDataRepresentationKind::{NormalGrids, Lasagna, Prediction}`
  (OpenDataManifest.hpp:137–141) to `normal_grids` / `lasagna` / `prediction`. `ref`
  serializes `OpenDataRepresentationRef{volumeIndex, artifactIndex}`
  (OpenDataManifest.hpp:143–147) as `"<volumeIndex>:<artifactIndex>"`.
- **errors:** `-32007` (`data.kind:"sample"`), `-32005`.

### 10.3 `catalog.open_sample` (extended)

Extends §3.16 with optional resource selection.

- **params:**
  ```json
  {"sampleId": str,
   "resources"?: {
     "volumeIds"?: [str],            // subset of the sample's volume ids
     "representationRefs"?: [str],   // "vi:ai" strings from catalog.describe_sample
     "kinds"?: [str]                 // subset of "normal_grids"|"lasagna"|"prediction"
   }}
  ```
  Filter semantics (binding): an **absent** sub-field means "no filter on that axis".
  A volume is attached iff `volumeIds` is absent or contains its id. A derived
  representation is attached iff it passes **all** provided filters (its volume passes
  `volumeIds`; its ref is listed in `representationRefs` if that is present; its kind is
  listed in `kinds` if that is present). `resources` omitted ≡ v1 behavior (attach
  everything). A selection that leaves **zero volumes** → `-32602`
  (`data.param:"resources.volumeIds"`). Any unknown `volumeId` / unparsable or
  out-of-range ref / unknown kind string → `-32007`
  (`data.kind:"resource"`, `data.id` = the offending string), validated **before** any
  download or project mutation begins.
- Implementation contract: a new value type
  ```cpp
  struct vc3d::opendata::OpenDataResourceSelection {
      std::optional<std::vector<std::string>> volumeIds;
      std::optional<std::vector<OpenDataRepresentationRef>> representations;
      std::optional<std::vector<OpenDataRepresentationKind>> kinds;
  };
  ```
  threaded as a new trailing `const OpenDataResourceSelection* selection = nullptr`
  parameter through `createOpenDataSampleProject`
  (OpenDataSampleProject.hpp:33–37, .cpp:353), `attachOpenDataSampleVolumes`
  (OpenDataSampleProject.hpp:39–41, .cpp:474), `attachOpenDataNormalGrids`
  (OpenDataNormalGrids.hpp:68–70, invoked at OpenDataSampleProject.cpp:406), and
  `attachOpenDataLasagna` (OpenDataLasagna.hpp:56–59, invoked at
  OpenDataSampleProject.cpp:411). `nullptr` selection preserves current behavior
  byte-for-byte. The bridge enters via the new non-interactive
  `MenuActionController` overload specified in §14.1.
- **result:**
  ```json
  {"opened": true, "sampleId": str, "vpkgPath": str, "volumeIds": [str],
   "attached": {"volumes": int, "segments": int, "normalGrids": int,
                "lasagnaDatasets": int},
   "messages": [str]}
  ```
  Counters/messages from `OpenDataSampleProjectResult` (OpenDataSampleProject.hpp:15–31).
- **errors:** `-32007` (`data.kind:"sample"` or `"resource"`), `-32005`, `-32602`.

Remote normal grids remain reachable **only** through this catalog streaming-attach
path: `vc::project::validateLocation` rejects remote URIs for every category except
`Volumes` (core/src/VolumePkg.cpp:367–380, "Remote locations are only supported for
volumes"), so there is deliberately **no** generic "attach remote normal grid by URL"
RPC in v2.

### 10.4 `volume.select`

Switch the current volume among the already-attached entries of the open package.

- **params:** `{"volumeId": str}`
- Maps to `CState::setCurrentVolume(std::shared_ptr<Volume>)` (CState.hpp:47) with the
  volume resolved from the open `VolumePkg` by id; the `volumeChanged` signal
  (CState.hpp:89) drives all existing viewer/UI updates exactly as the volume combo
  does. Selecting the already-current id is a no-op success.
- **result:** `{"volumeId": str, "previousVolumeId": str}` (from
  `CState::currentVolumeId()`, CState.hpp:46).
- **errors:** `-32000`, `-32007` (`data.kind:"volume"`, `data.id`).

---

## 10.5 As-built notes (Stage 3)

Clarifications recorded during implementation; these amend §§10.1–10.4 and §14.1.

- **`volume.select` dispatch (§10.4).** Implemented via `CWindow::setVolume(vpkg->
  volume(id))` **followed by** `CWindow::syncVolumeSelectionControls(id)` — the exact
  pair the volume combo's `currentIndexChanged` slot runs (CWindow.cpp:8985–9009).
  `CWindow::setVolume` internally calls `CState::setCurrentVolume`/emits `volumeChanged`
  **and** performs the open-data cross-volume navigation transform, so this is the
  faithful "as the volume combo does" path (a bare `CState::setCurrentVolume` would skip
  the selector-UI reconciliation and the navigation transform). Matches the existing
  `volume.open` volume-switch branch. A resolve/load failure maps to `-32005`
  (`data.detail`).

- **`OpenDataResourceSelection` (§10.3) shared classifier.** A single per-artifact
  authority `vc3d::opendata::classifyDerivedRepresentation(const OpenDataArtifact&)`
  (OpenDataManifest.{hpp,cpp}) now backs both `derivedRepresentations()` (refactored to
  call it) and the attach-time filters, so a representation's `kind`/`ref` identity is
  computed identically at describe time and attach time. `OpenDataResourceSelection`
  gained two helpers: `allowsVolume(volumeId)` and
  `allowsRepresentation(volumeIndex, artifactIndex, kind, volumeId)`.

- **`OpenDataSampleProjectResult` gained `int attachedNormalGrids`** (previously the
  normal-grid count was only embedded in a message string). `catalog.open_sample`'s
  `attached.normalGrids` reads it directly.

- **Selection threading is nullptr-preserving, and the bridge always passes a non-null
  selection.** `catalog.open_sample` with no `resources` param constructs a
  default-constructed `OpenDataResourceSelection` (all axes `nullopt`) whose
  `allowsVolume`/`allowsRepresentation` return `true` for everything — byte-for-byte
  identical to the `nullptr` (attach-everything) path — so the bridge can always take the
  options overload (and thus always receive the `attached`/`messages` result) without
  changing v1 behavior.

- **`MenuActionController` (§14.1) as-built.** The pre-existing single-purpose
  `openOpenDataSampleById(const QString&, bool interactive = true)` is **retained** (the
  catalog window and the simple bridge path still use it). The new overload is
  `openOpenDataSampleById(const QString& sampleId, const OpenDataSampleOpenOptions&
  options, QString* errorMessage = nullptr, vc3d::opendata::OpenDataSampleProjectResult*
  resultOut = nullptr)` — the trailing `resultOut` (beyond the §14.1 sketch) hands the
  bridge the attach counters/messages for the `attached` block. `openOpenDataSample`
  gained matching trailing params `(…, const OpenDataResourceSelection* selection =
  nullptr, QString* errorMessage = nullptr, OpenDataSampleProjectResult* resultOut =
  nullptr)`; the selection is deep-copied before entering the `QtConcurrent` worker.

- **Cached-project reopen caveat (§10.3), verified live.** `createOpenDataSampleProject`
  loads any previously-cached `<remoteCache>/…/<sample>.volpkg.json` **before** attaching,
  so the resource filter gates *new additions* but does not remove volume entries a prior
  (unfiltered) open already persisted to that cache. On a **clean** first open the filtered
  result is exactly the subset (a `volumeIds:[one]`, `kinds:[]` open attaches
  `{volumes:1, normalGrids:0, lasagnaDatasets:0}`); on a **reopen over a fuller cache**,
  `attached.volumes` for that call is 0 while `volumeIds` still lists the cached set. The
  cache-independent invariants a client may rely on: `kinds:[]` never *newly* attaches a
  derived representation, and the filtered `volumeIds` are always a subset of the
  unfiltered `volumeIds`. Lasagna dataset entries **are** stale-pruned on reopen (existing
  `attachOpenDataLasagna` behavior), so a lasagna-excluding filter does drop previously
  cached lasagna entries; volumes/normal-grids have no such pruning by design (avoids
  disturbing virtual rebased-source views and coordinate pairings). A future stage may add
  opt-in volume pruning if agents need exact-subset reopens.

- **Manifest acquisition (§10.1) is the first shipping user of the §8.4 deferred
  mechanism.** When `refresh` is true, or nothing is cached, the bridge fetches
  `fetchOpenDataManifest()` on a `QtConcurrent` worker, replies via `beginDeferred`
  (30 s timeout), and — on the GUI thread in the watcher's `finished` slot — stores the
  result into `CWindow::_openDataManifestCache` (bridge is a `CWindow` friend) so
  subsequent `catalog.*`/`openOpenDataSampleById` calls see it. `catalog.describe_sample`
  reuses the same acquisition helper and applies its `-32007` (unknown sample) inside the
  builder so it surfaces correctly on both the synchronous (cached) and deferred paths.
  `catalog.describe_sample`'s `shapeZYX` is emitted in stored **z, y, x** order.

---

## 11. Lasagna RPCs

All methods below reach the singleton `LasagnaServiceManager::instance()`
(LasagnaServiceManager.hpp:34) and, where noted, the `SegmentationLasagnaPanel`
(reached through `CWindow`, which is already a bridge friend per §1.1; the panel is the
object wired at CWindow.cpp:7712–7713).

### 11.1 `lasagna.service_status`

- **params:** none.
- **result:**
  ```json
  {"running": bool, "external": bool, "host": str, "port": int,
   "lastError": str | null}
  ```
  From `isRunning()` (LasagnaServiceManager.hpp:52), `isExternal()` (:56), `host()`
  (:55), `port()` (:54), `lastError()` (:53) — `lastError` mapped to `null` when the
  QString is empty.
- **errors:** none.

### 11.2 `lasagna.ensure_service`

- **params:** `{"pythonPath"?: str, "host"?: str, "port"?: int}` — `host` and `port`
  must be given together; giving them selects **external** mode.
- External mode: `connectToExternal(host, port)` (LasagnaServiceManager.hpp:47) pings
  `GET /health` asynchronously; the call is **deferred** (§8.4), completed by
  `serviceStarted` (success) or `serviceError(message)` (→ `-32005` with the message as
  `data.detail`). Timeout **15 s**.
- Internal mode: `ensureServiceRunning(pythonPath)` (LasagnaServiceManager.hpp:41)
  returns synchronously; `false` → `-32005` with `lastError()` as `data.detail`.
- **result:** `{"running": true, "external": bool, "host": str, "port": int}`
- **errors:** `-32005`, `-32602` (host without port or vice versa).

### 11.3 `lasagna.list_datasets`

Deferred (§8.4) over `fetchDatasets()` (LasagnaServiceManager.hpp:91), completed by
`datasetsReceived(const QJsonArray&)` (:127). Timeout **10 s**.

- **params:** none.
- **result:** `{"datasets": [...]}` — the service's dataset objects passed through
  verbatim (the bridge does not reshape service JSON).
- **errors:** `-32005` (service not running — checked via `isRunning()` before issuing
  the fetch, `data.detail:"lasagna service is not running"`; or deferred timeout).

### 11.4 `lasagna.start_optimization`

- **params:**
  ```json
  {"mode": str,             // "reoptimize" | "new_model" | "offset" | "atlas"
   "configPath"?: str,      // default: panel's selected config for the mode
   "seed"?: Vec3,           // volume-space; components rounded to int
   "atlasPath"?: str}       // required for mode "atlas" unless the panel already has one
  ```
  Mode mapping to `SegmentationLasagnaPanel::LasagnaMode`
  (segmentation/panels/SegmentationLasagnaPanel.hpp:45):
  `reoptimize=ReOptimize(0)`, `new_model=NewModel(1)`, `offset=Offset(3)`,
  `atlas=Atlas(4)`.
- Implementation contract: a new **public** method on `SegmentationLasagnaPanel`
  ```cpp
  bool startOptimizationHeadless(CState* state,
                                 LasagnaMode mode,
                                 const QString& configPath,      // empty => selectedLasagnaConfigPathForMode(mode)
                                 std::optional<cv::Vec3i> seed,  // nullopt => panel seed / none
                                 const QString& atlasPath,       // empty => panel selection
                                 QString* errorMessage = nullptr);
  ```
  wrapping the existing private `startOptimizationWithOverrides(...)`
  (SegmentationLasagnaPanel.hpp:121–128) — the same engine behind the public
  `startOptimizationAtSeed(...)` (:71–77) — with `statusBar = nullptr` tolerated and
  all failure reporting via `errorMessage` (never a dialog). `atlasPath`, when given,
  is applied via `setSelectedAtlasPath` (:63) before launch. Default `configPath`
  resolution uses `selectedLasagnaConfigPathForMode(mode)` (:78).
- The bridge registers a `source:"lasagna"` job at submission time; the record's
  `externalId` is filled when `jobStarted(jobId)` (LasagnaServiceManager.hpp:119)
  arrives. Progress: `optimizationProgress(stage, step, totalSteps, loss, ...)`
  (:110–112) → `job.progress` `phase:"output"` messages (rate-limited per §3.18);
  terminal: `optimizationFinished(outputDir)` / `jobFinished` → `finished` with
  `success:true` and `outputPath`; `optimizationError` / `jobError` → `finished`
  with `success:false`.
- **result:** `{"jobId": str, "kind": "lasagna.optimize", "source": "lasagna"}`
- **errors:** `-32000`, `-32004` (`data.source:"lasagna"` — the bridge enforces one
  in-flight bridge-submitted optimization even though the service itself queues; use
  `lasagna.jobs` for queue visibility), `-32005` (service not running / submission
  failed; `errorMessage` as `data.detail`), `-32007` (`data.kind:"config"` —
  `configPath` resolves to nothing on disk; or `data.kind:"atlas"` for mode `atlas`
  with no atlas), `-32009` (lasagna panel unavailable), `-32602` (bad mode string /
  malformed seed).

### 11.5 `lasagna.jobs`

Deferred (§8.4) over `fetchJobs()` (LasagnaServiceManager.hpp:76), completed by
`jobsUpdated(const QJsonArray&)` (:118). Timeout **10 s**.

- **params:** none.
- **result:** `{"jobs": [...]}` — service job objects verbatim (including the
  local-upload overlay entries the manager merges in).
- **errors:** `-32005` (service not running / timeout).

### 11.6 `lasagna.cancel`

- **params:** `{"jobId"?: str}` — a bridge job id (`"job-<n>"`, resolved to its
  `externalId`) **or** a raw service job id. Omitted → `stopOptimization()`
  (LasagnaServiceManager.hpp:72) for the active optimization.
- Maps to `cancelJob(const QString&)` (LasagnaServiceManager.hpp:73).
- **result:** `{"cancelRequested": true, "serviceJobId": str | null}`
- **errors:** `-32007` (`data.kind:"job"` — bridge id unknown, or omitted with no
  active lasagna job), `-32005` (service not running).

### 11.7 `lasagna.select_output_segment`

Activate a lasagna output segment by name — the programmatic twin of the
`lasagnaOutputActivated(QString)` signal wiring
(SegmentationLasagnaPanel.hpp:92 → connected at CWindow.cpp:7713).

- **params:** `{"name": str}`
- Maps to `SurfacePanelController::selectSurfaceById(const std::string&)`
  (SurfacePanelController.hpp:120), i.e. the same handler the signal reaches.
- **result:** `{"selected": true, "name": str}`
- **errors:** `-32000`, `-32602` (`name` is empty), `-32007` (`data.kind:"segment"` —
  `selectSurfaceById` returned false).

### 11.8 `lasagna.repeat_last`

- **params:** none.
- Maps to `SegmentationLasagnaPanel::repeatLastLasagnaAction()`
  (SegmentationLasagnaPanel.hpp:69), which relaunches the last-used mode.
- **result:** `{"jobId": str, "kind": "lasagna.optimize", "source": "lasagna"}` — same
  job semantics as §11.4.
- **errors:** `-32004` (`data.source:"lasagna"`), `-32005` (nothing to repeat / launch
  failed; `data.detail`), `-32009` (lasagna panel unavailable), `-32000`.

### 11.9 `workspace.switch`

- **params:** `{"name": str}` — `"main"`, `"lasagna"`, or `"fiber_slice"`.
- Maps to `CWindow::switchToMainWorkspace()` (CWindow.hpp:241, CWindow.cpp:4773) /
  `CWindow::switchToLasagnaWorkspace()` (CWindow.hpp:182, CWindow.cpp:4760) /
  `CWindow::switchToFiberSliceWorkspace()` (CWindow.hpp:184, CWindow.cpp:4786); all
  private, reached via the existing `friend class AgentBridge` (§1.1). The workspace
  tabs are created at CWindow.cpp:2905–2909 (5 tabs total: main, Lasagna, Atlas,
  Fiber Slice, Intersections — only these three are RPC-reachable; Atlas/
  Intersections have no `workspace.switch` target as of this writing).
  **`"main"` is the only documented way back** from `"lasagna"`/`"fiber_slice"` —
  there is no automatic return, and the workspace tab is real Qt UI state that
  persists across app restarts (verified live: a freshly launched VC3D opened
  straight onto a leftover "Fiber Slice" tab from a prior session, silently
  hiding the main-tab viewers — see the screenshot §3.4 note above).
- **result:** `{"workspace": str}`. Any viewers the workspace creates register through
  `ViewerManager` and become targetable per §2.2.
- **errors:** `-32602` (unknown name), `-32000`.

---

## 12. Atlas RPCs

All state lives on `CWindow` (bridge is a friend, §1.1). Search-mode integers:
`ATLAS_SEARCH_MODE_ATLAS_TO_NON_ATLAS = 0`, `ATLAS_SEARCH_MODE_NON_ATLAS_ONLY = 1`
(CWindow.cpp:190–191); phases number `1..ATLAS_SEARCH_PHASE_COUNT` with
`ATLAS_SEARCH_PHASE_COUNT = 5` (CWindow.cpp:192).

### 12.1 `atlas.open`

- **params:** `{"atlasDir": str}` — absolute path, or relative to the volpkg root.
- Maps to `CWindow::displayAtlasFromDirectoryHeadless(const std::filesystem::path&,
  QString* errorMessage)` — a **distinct-name** headless split (as built; see §14.3
  for why a same-name overload is not used). Both it and the interactive
  `displayAtlasFromDirectory` share the dialog-free core
  `CWindow::loadAndDisplayAtlas(...)`; the rebuild `QMessageBox::question` prompt and
  the warning dialog live only in the interactive method.
- **result:** `{"opened": true, "atlasDir": str, "atlasName": str}` — from
  `_currentAtlasDir` / `_currentAtlasName` after the call.
- **errors:** `-32007` (`data.kind:"atlas"` — directory missing), `-32005` (atlas load
  threw; `data.detail` = exception text), `-32000`.

### 12.2 `atlas.status`

- **params:** none.
- **result:**
  ```json
  {"atlasDir": str | null, "atlasName": str | null,
   "search": {"running": bool, "phase": int, "phaseCount": 5,
              "completed": int, "total": int,
              "cancelRequested": bool, "resultCount": int}}
  ```
  From `_currentAtlasDir`/`_currentAtlasName` (CWindow.hpp:354–355),
  `_atlasSearchProgressPhase` (:368–369), `_atlasSearchPhaseCompleted`/`Total`
  (:370–371), `_atlasSearchCancelRequested` (:366), `_atlasSearchResults.size()`
  (:357). `running` reflects the bridge's `source:"atlas"` job state.
- **errors:** none.

### 12.3 `atlas.search_start`

Asynchronous; registers a `source:"atlas"` job.

- **params:**
  ```json
  {"mode"?: str = "atlas_to_non_atlas",  // "atlas_to_non_atlas" | "non_atlas_only"
   "requiredTags"?: [str] = [],
   "excludedTags"?: [str] = [],
   "maxDistance"?: float}                // omit to keep the current spin-box value
  ```
- Implementation contract (as built): a new **public, distinct-name** headless method
  on `CWindow`, decoupled from GUI-widget scraping:
  ```cpp
  struct AtlasFiberSearchParams {
      int searchMode{0};                 // ATLAS_SEARCH_MODE_* (CWindow.cpp)
      QStringList requiredTags;
      QStringList excludedTags;
      std::optional<double> maxDistance; // -> FiberIntersectionBroadPhaseOptions::maxDistance
  };
  bool startAtlasFiberIntersectionSearchHeadless(const AtlasFiberSearchParams& params,
                                                 QString* errorMessage = nullptr);
  ```
  NOT a same-name overload of the zero-arg slot: the slot is used as a
  member-function pointer in a new-style `connect()` (the search dock's Run button),
  which an overload makes ambiguous — same convention as
  `startOptimizationHeadless` / `repeatLastLasagnaActionHeadless` (§14.3).
  The zero-arg slot `startAtlasFiberIntersectionSearch()` scrapes its widgets exactly
  as before — `atlasSearchTypeCombo`, `atlasSearchTagFilterEdit`,
  `atlasSearchExcludeTagFilterEdit`, `atlasSearchMaxDistanceSpin` — into an
  `AtlasFiberSearchParams` and forwards to the headless method. Validation moved into
  the headless method, which reports each failed condition through `errorMessage` and
  returns `false`; the interactive slot shows a single
  `QMessageBox::warning(errorMessage)` (the empty-fiber-storage case stays a silent
  dock refresh, as before). When `maxDistance` is omitted the headless method reads
  the persisted `atlas/search_max_distance` QSettings key — the exact value the spin
  box shows — so "omit to keep the current spin-box value" holds without touching a
  widget. This is the §4 headless-split doctrine applied to atlas search (see §14.3
  for the signals half).
- **result:** `{"jobId": str, "kind": "atlas.fiber_search", "source": "atlas"}`
- **errors:** `-32004` (`data.source:"atlas"`), `-32007` (`data.kind:"atlas"` — mode
  `atlas_to_non_atlas` with no atlas open, or the atlas has no fiber mappings),
  `-32005` (other precondition failures; `errorMessage` as `data.detail`), `-32602`
  (unknown mode / negative maxDistance), `-32000`.

### 12.4 `atlas.search_cancel`

- **params:** none.
- Maps to `CWindow::cancelAtlasFiberIntersectionSearch()` (CWindow.hpp:168), which sets
  `_atlasSearchCancelRequested` / the shared `_atlasSearchCancelFlag`
  (CWindow.hpp:366–367). The job still terminates through `atlasSearchFinished`
  (`success:false`).
- **result:** `{"cancelRequested": true}`
- **errors:** `-32007` (`data.kind:"job"` — no atlas search running).

### 12.5 `atlas.search_results`

Paginated read of `_atlasSearchResults`
(`std::vector<vc::atlas::FiberIntersectionResult>`, CWindow.hpp:357) zipped with
`_atlasSearchSignedWindings` (`std::vector<double>`, CWindow.hpp:358). Result field
names follow `vc::atlas::FiberIntersectionResult`
(core/include/vc/atlas/FiberIntersections.hpp:90–108).

- **params:** `{"offset"?: int = 0, "limit"?: int = 100}` — `limit` clamped to
  `[1, 1000]`.
- **result:**
  ```json
  {"total": int, "offset": int,
   "results": [
     {"index": int,                     // position in the results vector; the id atlas.open_result takes
      "sourceFiberId": str, "targetFiberId": str,   // uint64 as strings
      "candidateDistance": float, "refinedScore": float,
      "windingDistance": float | null,               // null when infinite
      "signedWinding": float | null,                 // from _atlasSearchSignedWindings; null if absent
      "sourcePoint": Vec3, "targetPoint": Vec3,
      "sourceArclength": float, "targetArclength": float,
      "converged": bool, "message": str}
   ]}
  ```
- **errors:** `-32602` (negative offset / bad limit). An empty result set is not an
  error (`total: 0`).

### 12.6 `atlas.open_result`

- **params:** `{"index": int}` — an `index` value as returned by §12.5 (vector order).
- As built: `CWindow::openAtlasSearchResult(int)` already indexes
  `_atlasSearchResults` directly in vector order (the tree items carry the vector
  index via `ATLAS_SEARCH_RESULT_INDEX_ROLE`), so no sorted-index translation exists
  or is needed. The bridge does not call the interactive method (its
  missing-workspace / no-atlas paths are `QMessageBox`es, and
  `LineAnnotationController::showIntersectionInspection` reports failures via
  `showError`'s `QMessageBox`): it re-checks those preconditions itself and calls the
  new dialog-free split
  `LineAnnotationController::showIntersectionInspectionHeadless(result, targetArea,
  atlasDir, QString* errorMessage)` (which shares
  `rebuildIntersectionInspection(QString* errorMessage = nullptr)` with the
  interactive path), plus the same workspace-tab switch the interactive slot does.
- **result:** `{"opened": true, "index": int}`
- **errors:** `-32007` (`data.kind:"result"` — index out of range or no results;
  `data.kind:"atlas"` — no atlas open), `-32005` (workspace unavailable or the
  inspection rebuild failed; `data.detail`), `-32602`.

### 12.7 `atlas.remap`

- **params:** none.
- As built: maps to a new headless split
  `CWindow::startAtlasRemapHeadless(QString* errorMessage, std::function<void(bool,
  const QString&)> onFinished = {})` — the interactive `remapCurrentAtlas()` slot
  forwards to it, supplying its failure `QMessageBox`es (sync via `errorMessage`,
  async via `onFinished`). The split exists because the original method's async
  completion showed a `QMessageBox` on worker failure and re-displayed via the
  *interactive* atlas open (rebuild-prompt risk) — both permanent-hang hazards
  offscreen. The headless completion re-displays via
  `displayAtlasFromDirectoryHeadless` and reports failure to the status bar only.
  Synchronous from the bridge's perspective: the response (`remapped: true`) is sent
  once the remap worker is launched.
- **result:** `{"remapped": true}`
- **errors:** `-32007` (`data.kind:"atlas"` — no atlas open), `-32005` (`data.detail`).

### 12.8 `atlas.optimize_snap_candidates`

- **params:** none.
- As built: maps to a new headless split
  `CWindow::optimizeAtlasSnapCandidatesHeadless(QString* errorMessage,
  std::function<void(const QString&)> onAsyncError = {})`, which queues Laplace
  snap-ranking through the lasagna fit service
  (`LasagnaServiceManager::rankLaplaceSnapPairs`). The interactive
  `optimizeAtlasSnapCandidates()` slot forwards to it and layers its `QMessageBox`es
  on top (sync via `errorMessage`, async via `onAsyncError`); the headless core
  reports async failures to the status bar / stderr only, and its success completion
  re-displays the atlas via the dialog-free `loadAndDisplayAtlas` (previously the
  interactive open with its rebuild-prompt risk).
- **result:** `{"requested": true}` — completion is observable via app status messages;
  v2 deliberately does not model this as a job (Stage 6 may revisit).
- **errors:** `-32007` (`data.kind:"atlas"`), `-32005` (lasagna service not running or
  other precondition failure; `data.detail`).

### 12.9 New `CWindow` signals (binding, backing §12.3 and §8.3)

```cpp
signals:
    /// phase in [1, ATLAS_SEARCH_PHASE_COUNT]; fraction in [0,1] within the phase.
    void atlasSearchProgressChanged(int phase, double fraction);
    /// success=false covers both cancellation and error; resultCount = _atlasSearchResults.size().
    void atlasSearchFinished(bool success, int resultCount);
```

`atlasSearchProgressChanged` is emitted from `updateAtlasSearchProgress` alongside
the existing progress-bar update (phase number via `atlasSearchPhaseNumber(phase)`,
fraction from completed/total). `atlasSearchFinished` is emitted from the
search-watcher completion handler on every terminal path of a launched search:
success (after `populateAtlasSearchResults(...)`), cancellation, and worker failure
(as built, the watcher's `result()` rethrow is caught and treated as a failed search
rather than escaping into the event loop — previously a crash). Pre-launch
validation failures never emit the signal: no search was started (the headless
launcher returns `false` instead, §12.3). The bridge maps the signals to
`job.progress` notifications (`source:"atlas"`; `phase:"output"` with
`message:"phase <p>/5 (<pct>%)"`, then `phase:"finished"` with `success`).
As built, the bridge only tracks **bridge-initiated** searches as jobs (per §12.2's
"running reflects the bridge's job state"); it deliberately does not auto-register a
human-initiated search from progress signals, because `FinishResults` progress also
fires on pure UI re-population (the group-by-fiber checkbox), which would leak a
never-finishing job. `jobIsRunning("atlas")` additionally consults
`CWindow::_atlasSearchCancelFlag` (non-null exactly while a search is in flight) so
`atlas.search_start` still returns `-32004` while a human-initiated search runs.

---

## 13. Line annotation RPCs (`fiber.*`)

All methods reach `LineAnnotationController` (owned by CWindow;
`_lineAnnotationController`).

**Viewer registration (verified):** every pane inside the line-annotation workspaces is
created through `ViewerManager::createViewer(...)` / `createViewerInWidget(...)` with
`ViewerManager::ViewerRole::Annotation` — `LineAnnotationDialog::addPane`
(LineAnnotationDialog.cpp:746–758) and the generated-views builders
(LineAnnotationDialog.cpp:838–842, 1077–1080, 1130–1135, 1200–1204;
LineAnnotationController.cpp:3059–3061, 3869–3871, 4094–4096). `ViewerManager`
registers each created viewer into `_baseViewers` and emits `baseViewerCreated`
(ViewerManager.cpp:235, 260). **Therefore all fiber panes are already reachable via the
`"v<N>"` viewer-id scheme of §2.2 with no new registration code** — `canvas.click` /
`canvas.drag` work on them as-is. Control-point add / delete / branch therefore need
no dedicated RPCs: they reuse `canvas.click` (context menu paths) and `canvas.drag`
on the annotation panes.

**As-built correction (Stage 5b):** the control-point *context menu*
(delete / branch / go-to-linked) is NOT reachable through `canvas.click`. It
is raised only by the physical-event handler
`CVolumeViewerView::mousePressEvent` (Ctrl+Right,
volume_viewers/CVolumeViewerView.cpp:407–414 →
`sendAnnotationContextMenuRequested` → `QMenu::exec()`), which the bridge's
synthesized slot calls (`onMousePress`/`onMouseRelease`/`onVolumeClicked`)
deliberately bypass — a deliberate safety property, since `QMenu::exec()` is
a nested event loop (§1.3). Control-point *placement* and seed gestures on
the annotation panes DO flow through the public slots and work via
`canvas.click`/`canvas.drag`; menu-only operations (delete a generated
control point, create/open a branch) currently have no bridge surface.

**Verified behavior (live agent run, 2026-07-20):** on an annotation pane, a
plain `canvas.click` (no `"shift"` modifier) fires
`generatedControlPointRequested` — this is what adds a control point.
`canvas.shift_click` (or `canvas.click` with `"shift"` in `modifiers`) instead
fires `generatedPredSnapPointRequested`, a "predicted snap point" gesture —
**not** a control-point add (`LineAnnotationDialog.cpp`). Callers placing
control points must use plain `canvas.click`/`vc3d_click`; §3.7's general
"shift+click = place point" framing does not hold on these panes.

Also verified: the annotation panes' `"v<N>"` viewer ids are **not stable
across edits** — `LineAnnotationDialog` rebuilds/reoptimizes its panes after
each control-point change and re-registers them with `ViewerManager` under
new ids. Callers must re-call `state.get` before targeting a pane after any
edit; a stale id fails `-32002`.

### 13.1 `fiber.launch`

Open the line-annotation workspace seeded at a position — the twin of the launch
gesture. Maps to
`LineAnnotationController::launchFromViewerAtPoint(CChunkedVolumeViewer*, const QPointF&, bool replaceOwningAnnotation)`
(LineAnnotationController.hpp:148–150), gated by `canLaunchFromViewer`
(LineAnnotationController.hpp:146).

- **params:**
  ```json
  {"viewer"?: str,                          // §2.2, default "segmentation"
   "position": Vec3 | {"x": float, "y": float},
   "space"?: "volume" | "scene",            // default "volume"; §3.6 validation rules
   "replaceOwning"?: bool = true}
  ```
  **`replaceOwning` defaults to `true`**, which discards the caller's currently
  open/in-progress annotation workspace (verified live: launching fiber 2 with
  the default silently threw away fiber 1's unsaved control points). Pass
  `replaceOwning: false` to keep multiple fiber workspaces open at once (e.g.
  tracing several fibers before a single combined `fiber.save`).
- **result:** `{"launched": true}` — the workspace's new panes appear in
  `state.get.viewers` (registered per the preamble above).
- **errors:** `-32000`, `-32001`, `-32002`, `-32003`, `-32009`
  (`canLaunchFromViewer` false, or non-chunked viewer).

### 13.2 `fiber.list`

- **params:** none.
- **result:** from `fiberSummaries()` / `knownFiberTags()`
  (LineAnnotationController.hpp:179–180); `FiberSummary` shape at
  LineAnnotationController.hpp:64–99. uint64 ids serialize as strings.
  ```json
  {"fibers": [
     {"fiberId": str, "name": str,
      "controlPointCount": int, "linePointCount": int, "lengthVx": float,
      "automaticHvTag": str, "manualHvTag": str, "automaticCertainty": float,
      "tags": [str],
      "spans": [
        {"spanIndex": int, "firstControlIndex": int, "secondControlIndex": int,
         "controlPointCount": int, "linePointCount": int, "lengthVx": float}
      ]}
   ],
   "knownTags": [str]}
  ```
- **errors:** `-32000`.

### 13.3 `fiber.open`

- **params:** `{"fiberId": str}` plus **at most one** of:
  `{"controlPointIndex": int}` | `{"linePointIndex": int}` | `{"span": [int, int]}`.
  More than one selector → `-32602`.
- Maps to `openFiber(uint64_t)` (LineAnnotationController.hpp:151),
  `openFiberAtControlPoint` (:152), `openFiberAtLinePointIndex` (:153),
  `openFiberSpan(fiberId, first, second)` (:154) respectively.
- **result:** `{"opened": true, "fiberId": str}`
- **errors:** `-32007` (`data.kind:"fiber"`), `-32602`, `-32000`.

### 13.4 `fiber.set_follow`

Toggle "current cut follows strip mouse" on the open line-annotation workspace.

- **params:** `{"enabled": bool}`
- Implementation contract: a new tiny **public** wrapper
  `void LineAnnotationDialog::setCutFollowEnabled(bool enabled)` delegating to the
  existing **private** `setCurrentCutFollowsStripMouse(bool)`
  (LineAnnotationDialog.hpp:210; state member `_currentCutFollowsStripMouse`,
  LineAnnotationDialog.hpp:329; keyboard twin `toggleCurrentCutFollowFromKeyboard`,
  LineAnnotationDialog.hpp:226). The bridge targets the most recently opened live
  `LineAnnotationDialog`.
- **result:** `{"enabled": bool}`
- **errors:** `-32007` (`data.kind:"fiber_workspace"` — no line-annotation dialog open).

### 13.5 `fiber.save`

- **params:** none.
- Maps to `saveOpenFibers()` (LineAnnotationController.hpp:174).
- **result:** `{"saved": true}`
- **errors:** `-32000`.

### 13.6 `fiber.delete`

- **params:** `{"fiberIds": [str]}` — ≥ 1 id. All ids validated against
  `fiberSummaries()` first; any unknown id fails the whole call (all-or-nothing).
- Maps to `deleteFibers(std::vector<uint64_t>)` (LineAnnotationController.hpp:156).
- **result:** `{"deleted": [str]}`
- **errors:** `-32007` (`data.kind:"fiber"`, `data.id` = first unknown), `-32602`
  (empty list), `-32000`.

### 13.7 `fiber.set_tag`

- **params:** `{"fiberId": str, "tag": str, "enabled": bool}` — free-form tag string
  (the app's tag vocabulary is open; `fiber.list.knownTags` enumerates ones in use).
- Maps to `setFiberTag(uint64_t, const QString&, bool)`
  (LineAnnotationController.hpp:161).
- **result:** `{"fiberId": str, "tag": str, "enabled": bool}`
- **errors:** `-32007` (`data.kind:"fiber"`), `-32602` (empty tag), `-32000`.

### 13.8 `fiber.create_atlas`

Deferred (§8.4). Maps to `createAtlasFromFiber(uint64_t)`
(LineAnnotationController.hpp:167); completed by the one-shot
`atlasCreated(std::filesystem::path)` signal (LineAnnotationController.hpp:200).
Timeout **600 s**.

- **params:** `{"fiberId": str}`
- **result:** `{"atlasDir": str}`
- **errors:** `-32007` (`data.kind:"fiber"`), `-32005` (timeout / creation failed),
  `-32000`.

### 13.9 `fiber.export`

**Verified:** `exportFibers()` (LineAnnotationController.cpp:2647) is dialog-driven —
it calls `showFiberJsonPathDialog(...)` (invocation at :2655; the dialog helper at
:274–330 uses `QFileDialog::getSaveFileName` etc.) and reports via `QMessageBox`
(:2672–2676). Headless split per the §4 doctrine (see §14.2):

- **params:** `{"path": str, "scale"?: float = 1.0}` — `path` is the output JSON file
  (a `vc3d_fiber_collection` bundle, LineAnnotationController.cpp:2661–2671).
- **result:** `{"exported": int, "path": str}`
- **errors:** `-32000`, `-32005` (no fibers to export / write failed; `data.detail`),
  `-32602` (scale ≤ 0 or non-finite).

### 13.10 `fiber.import`

**Verified:** `importFibers()` (LineAnnotationController.cpp:2500) is likewise
dialog-driven (`showFiberJsonPathDialog` at :2508; success `QMessageBox` at
:2635–2641). Headless split per §14.2. Accepts the same inputs the interactive path
does: a single fiber JSON, a bundle, or a directory of fiber JSONs
(LineAnnotationController.cpp:2537–2604).

- **params:** `{"path": str, "scale"?: float = 1.0}`
- **result:** `{"imported": int, "skipped": int}`
- **errors:** `-32000`, `-32007` (`data.kind:"path"` — path does not exist),
  `-32005` (no valid fibers found / parse failure; `data.detail`), `-32602`.

### 13.11 As-built notes (Stage 5b)

Implementation-time amendments; where these contradict §13.1–13.10, the
as-built notes win.

- **Bridge-lifetime headless error reporting (binding).** Nearly every
  `LineAnnotationController` public method — including all §13 targets —
  reports failures through `showError()` (a blocking `QMessageBox::warning`,
  LineAnnotationController.cpp) or can reach the Lasagna **dataset-picker
  QFileDialog** (`ensureDatasetForSession` /
  `resolveAlignmentMetricsManifestPath` when no dataset is selected), and
  failures also surface from **asynchronous completions** (line-optimization
  results, fiber-save jobs) minutes after an RPC returned, so a per-call
  guard cannot cover them. As built, `AgentBridgeServer`'s constructor calls
  the new `LineAnnotationController::setErrorDialogsSuppressed(true)` once for
  the bridge's lifetime: `showError` logs + records instead of showing a
  dialog, the dataset picker is never opened (the operation fails as if
  cancelled, with the reason recorded), the broken-branch-links repair prompt
  in `loadFibersForCurrentPackage` defaults to the non-destructive "keep
  files unchanged" path, and `confirmLinkedControlPointEdit` refuses (returns
  false) instead of prompting. Handlers convert a recorded message
  (`takeLastSuppressedError()`) into `-32005` with `data.detail`. The bridge
  is opt-in, so plain interactive sessions are unaffected.
- **`fiber.launch` / `fiber.open`** therefore fail `-32005` (not a hang) with
  `data.detail: "No Lasagna dataset is selected for the active volume."` when
  the package has no resolvable Lasagna dataset — the interactive twin would
  block on the picker dialog.
- **`fiber.save`** maps to a new headless split
  `LineAnnotationController::saveOpenFibersHeadless()` (shared core
  `saveOpenFibersCore`), NOT to `saveOpenFibers()` as §13.5 stated: the
  interactive method ends in `waitForFiberSaves()`, which spins a nested
  `QEventLoop` (forbidden, §1.3). The headless variant schedules the same
  saves; they complete asynchronously on the fiber-save watcher.
- **`fiber.create_atlas` is synchronous, not deferred.** §13.8's deferred
  design is dropped: `createAtlasFromFiber` runs entirely on the GUI thread
  and emits `atlasCreated` before returning, so deferral added nothing — and
  both its failure path (`showError`) and its success path (`atlasCreated` →
  the interactive `CWindow::displayAtlasFromDirectory`, which can raise the
  atlas-rebuild `QMessageBox::question`) violate §1.3. As built the RPC calls
  the new `createAtlasFromFiberHeadless(fiberId, &err, &atlasDir)` (shared
  core `createAtlasFromFiberCore`; does **not** emit `atlasCreated`), then
  displays the result via the already-proven
  `CWindow::displayAtlasFromDirectoryHeadless` (§12.1). Result gains additive
  fields: `{"atlasDir": str, "displayed": bool, "displayDetail"?: str}`.
  The RPC blocks until creation finishes (can be slow); clients should use a
  generous timeout.
- **`fiber.set_follow`** is backed by public wrappers
  `LineAnnotationDialog::setCutFollowEnabled(bool)` / `cutFollowEnabled()`
  and `LineAnnotationController::mostRecentLineAnnotationDialog()` (last
  live dialog in pane order).
- **`fiber.delete`** result is `{"deleted": [str]}` as specified; if a
  pre-validated id still fails at the filesystem level the call returns
  `-32005` with `data.deleted` listing the partial set and `data.detail`
  carrying the recorded reason.
- **`fiber.list`** rows omit the alignment-metrics block (async/pending by
  nature); ids are runtime ids reassigned on each reload
  (`loadFibersForCurrentPackage` numbers fibers 1..N in file order), so ids
  from an earlier `fiber.list` are stale after any import/delete/reload.
- `fiber.export` / `fiber.import` reject `scale <= 0` with `-32602` (the
  interactive spin box allows negative scales; the RPC surface does not).

---

## 14. Interactive/headless splits introduced in v2

Same doctrine as §4: shared logic moves to a non-interactive method that reports via
`QString* errorMessage`; dialogs and `QMessageBox`es remain exclusively in the
interactive slot; no logic is duplicated.

### 14.1 `MenuActionController` — non-interactive catalog open

New public overload:

```cpp
struct OpenDataSampleOpenOptions {
    vc3d::opendata::OpenDataResourceSelection selection;  // §10.3; default = attach all
    bool interactive{true};   // bridge always passes false
};
bool MenuActionController::openOpenDataSampleById(const QString& sampleId,
                                                  const OpenDataSampleOpenOptions& options,
                                                  QString* errorMessage = nullptr);
```

With `interactive == false` the flow skips the replace-project `QMessageBox`
(MenuActionController.cpp:518–532, `prompt.exec()` at :527) — an explicit agent call is
consent to replace — and does not construct the segment-download `QProgressDialog`
(MenuActionController.cpp:539–546). The existing single-arg
`openOpenDataSampleById(const QString&)` (MenuActionController.hpp:48) becomes a thin
forwarder with `interactive = true` and empty selection.

### 14.2 `LineAnnotationController` — fiber export/import

```cpp
bool exportFibersToPath(const std::filesystem::path& path, double scale,
                        QString* errorMessage = nullptr, int* exportedCount = nullptr);
bool importFibersFromPath(const std::filesystem::path& path, double scale,
                          QString* errorMessage = nullptr,
                          int* importedCount = nullptr, int* skippedCount = nullptr);
```

`exportFibers()` / `importFibers()` (LineAnnotationController.cpp:2647 / :2500) become
"dialog (`showFiberJsonPathDialog`, :274–330) → call the headless method →
`QMessageBox` the outcome". The JSON serialization/parsing bodies
(:2513–2634, :2660–2671) move verbatim into the headless methods, including
`loadFibersForCurrentPackage()` on successful import (:2634).

As built (Stage 5b), `LineAnnotationController` gained three further headless
splits beyond the two above, all distinct names per the same doctrine:
`createAtlasFromFiberHeadless` (+ shared core `createAtlasFromFiberCore`),
`saveOpenFibersHeadless` (+ shared core `saveOpenFibersCore`; the interactive
`saveOpenFibers` keeps its `waitForFiberSaves()` nested event loop), and the
bridge-lifetime `setErrorDialogsSuppressed(bool)` /
`takeLastSuppressedError()` headless error-reporting valve plus
`mostRecentLineAnnotationDialog()`; `LineAnnotationDialog` gained the public
`setCutFollowEnabled(bool)` / `cutFollowEnabled()` wrappers. See §13.11.

### 14.3 `CWindow` — atlas headless splits and signals

As specified in §12.3 (headless launcher + widget-scraping refactor of the zero-arg
slot) and §12.9 (the two new signals). As built, all atlas headless splits use
**distinct method names** (`...Headless`), never same-name overloads: several of the
interactive originals are used as member-function pointers in new-style `connect()`
calls (`startAtlasFiberIntersectionSearch` on the search dock's Run button,
`displayAtlasFromDirectory` on `LineAnnotationController::atlasCreated`), and adding
a same-name overload makes those `connect()` calls ambiguous and breaks the build.
This matches every prior headless split in the codebase
(`startOptimizationHeadless`, `repeatLastLasagnaActionHeadless`,
`activateSurfaceById` vs `selectSurfaceById`). Full as-built list:
`displayAtlasFromDirectoryHeadless` (+ shared core `loadAndDisplayAtlas`, §12.1),
`startAtlasFiberIntersectionSearchHeadless` (§12.3), `startAtlasRemapHeadless`
(§12.7), `optimizeAtlasSnapCandidatesHeadless` (§12.8), and — on
`LineAnnotationController` — `showIntersectionInspectionHeadless` (+
`rebuildIntersectionInspection(QString* errorMessage = nullptr)`, §12.6).

### 14.4 `SegmentationCommandHandler` — headless Run Trace

Current reality: `onGrowSegmentFromSegment(const std::string& segmentId)`
(SegmentationCommandHandler.cpp:2101–2182) mixes preconditions (surface+runner check,
remote-volume rejection at :2105–2112, `trace_params.json` existence at :2131–2134),
the modal `TraceParamsDialog` (:2136–2145; window title "Run Trace Parameters",
ToolDialogs.cpp:344), params-JSON merge (:2147–2168), and launch
(`setTraceParams` + `execute(CommandLineToolRunner::Tool::GrowSegFromSegment)`,
:2170–2180).

Binding refactor:

```cpp
struct RunTraceParams {
    QJsonObject paramOverrides;   // merged over trace_params.json (same merge as :2156-2157)
    int ompThreads{-1};           // -1 => runner default
    QString tgtDir;               // empty => <volpkg>/traces (created if missing, :2117-2128)
};
bool startRunTrace(const std::string& segmentId, const RunTraceParams& params,
                   QString* errorMessage = nullptr);
```

`startRunTrace` owns steps 1 (preconditions), 3 (merge + write
`trace_params_ui.json`), 4 (launch); the slot keeps the dialog and
`_cmdRunner->showConsoleOutput()` (:2178, interactive-only). RPC surface in §15.4.

**Amendment (as-built, Stage 6):** the shipped signature adds a fourth out-param so
the RPC can report the resolved target dir without re-deriving it:
`bool startRunTrace(const std::string& segmentId, const RunTraceParams& params,
QString* errorMessage = nullptr, QString* resolvedOutputDir = nullptr)`. The slot
becomes "dialog → build `RunTraceParams{paramOverrides = dlg.makeParamsJson(),
ompThreads = dlg.ompThreads(), tgtDir = dlg.tgtDir()}` → `startRunTrace` →
`QMessageBox` on failure → `showConsoleOutput()`". `startRunTrace` does **not** call
`requireSurfaceAndRunner` (which pops `QMessageBox`es) — it re-implements the same
checks inline reporting through `errorMessage` with distinct sentences the bridge
matches: `"Invalid segment"` → `-32007` `data.kind:"segment"`, `"trace_params.json
not found"` → `-32007` `data.kind:"file"`, `"remote"` → `-32009`, `"Command line
tools not available"` → `-32006`, `"already running"` → `-32004` `data.source:"tool"`,
else `-32005` `data.detail`. `outputDir` accepts an absolute path or one relative to
the volpkg root (created if missing).

---

## 15. Stage 6 backlog surface (lighter-weight, still binding)

### 15.1 `tags.set`

- **params:** `{"segmentId": str, "tag": str, "enabled": bool}` — `tag` enum:
  `"approved" | "defective" | "reviewed" | "inspect"`, mapping 1:1 to
  `SurfacePanelController::Tag {Approved, Defective, Reviewed, Inspect}`
  (SurfacePanelController.hpp:81–86). **Note (verified):** there is **no** `"revisit"`
  tag in the app — the Tag enum and the tag checkbox UI
  (SurfacePanelController.hpp:74–79) have exactly the four values above; `"revisit"` →
  `-32602`.
- Dispatch: `SurfacePanelController::selectSurfaceById(segmentId)`
  (SurfacePanelController.hpp:120) — `setTagChecked` operates on the current selection,
  so the call **leaves `segmentId` selected** (documented side effect) — then
  `setTagChecked(Tag, enabled)` (SurfacePanelController.hpp:125). `toggleTag`
  (SurfacePanelController.hpp:124) is deliberately not exposed (non-idempotent).
- **result:** `{"segmentId": str, "tag": str, "enabled": bool}`
- **errors:** `-32000`, `-32007` (`data.kind:"segment"`), `-32602`.

### 15.2 `seeding.*`

All map to `SeedingWidget` (apps/VC3D/SeedingWidget.hpp). The action entry points are
**private slots** today, so each RPC below requires a small public wrapper on
`SeedingWidget` (added in the public section, one-line delegations; names below are
binding):

| RPC | params | wrapper → private target | result |
|---|---|---|---|
| `seeding.set_winding_annotation_mode` | `{"active": bool}` | (already public slot) `setRelWindingAnnotationMode(bool)` (SeedingWidget.hpp:64) | `{"active": bool}` |
| `seeding.analyze_paths` | `{}` | `runAnalyzePaths()` → `analyzePaths()` (:98) | superseded, see below † |
| `seeding.preview_rays` | `{}` | `previewRaysHeadless()` | `{"requested": true}` |
| `seeding.cast_rays` | `{}` | `castRaysHeadless()` | `{"requested": true}` |
| `seeding.run` | `{}` | `runSeeding()` → `onRunSegmentationClicked()` (:74) | superseded, see below † |
| `seeding.expand` | `{}` | `runExpandSeeds()` → `onExpandSeedsClicked()` (:75) | superseded, see below † |
| `seeding.reset_points` | `{}` | `runResetPoints()` → `onResetPointsClicked()` (:76) | `{"reset": true}` |

† `run`/`expand`/`analyze_paths` were deferred at this point in the doc's history (see the
Stage 6 amendment immediately below), then actually implemented later (see the
"batch-seeding follow-up" amendment further down) with real result shapes that
supersede the placeholder ones in this table — jump to that amendment for the
binding params/result/error contract, this row is history only.

Errors for all: `-32000`, `-32001`. When a wrapper launches work through
`CommandLineToolRunner`, the standard `source:"tool"` job wiring (§8.3) picks it up
automatically (including the implicit externally-initiated record) — the RPCs
themselves stay fire-and-forget. Seed/path placement reuses `canvas.click` /
`canvas.drag` (the widget's `onMousePress/Move/Release` slots, SeedingWidget.hpp:61–63,
are already fed by the real viewer wiring).

**Amendment (as-built, Stage 6):** initially, three of the seven listed actions
(`run`, `expand`, `analyze_paths`) were **deferred** because each ended in a nested
`QApplication::processEvents` loop that blocked the event loop until child processes
finished (`run`/`expand`) or repainted per-path (`analyze_paths`) — a §1.3 violation.
The exposed subset was `set_winding_annotation_mode`, `preview_rays`, `cast_rays`
(async `QtConcurrent`, safe), and `reset_points`.

`preview_rays` / `cast_rays` use dialog-free entry points that return precondition
errors to the bridge. Their interactive slots call the same entry points and show the
returned error when needed, so enabling the bridge does not change human-facing
dialogs. Result shapes: `preview_rays`/`cast_rays` return `{"requested": true}`;
`reset_points` returns `{"reset": true}`;
`set_winding_annotation_mode` returns `{"active": bool}`.

**Amendment (as-built, batch-seeding follow-up):** the three deferred actions are now
**exposed**, the blocking loops removed:

- **The refactor (SeedingWidget.hpp/.cpp).** `onRunSegmentationClicked` /
  `onExpandSeedsClicked` no longer spin `while (jobsRunning) processEvents`. The
  self-referencing local `std::function` launchers and the by-reference-captured
  stack locals (`completedJobs`, `nextPointIndex`/`nextIterationIndex`, `totalPoints`/
  `expansionIterations`, the batch config) were promoted to member state (`_batchKind`,
  `_batchTotal`, `_batchCompleted`, `_batchNextIndex`, `_batchOmpThreads`,
  `_batchPoints`, `_batchVolumePath`, `_batchPathsDir`, `_batchConfigJson`,
  `_batchWorkingDir`) and to member methods (`startSegmentationProcessForPoint`,
  `startExpansionProcessForIteration`, `handleBatchProcessFinished`,
  `finalizeSeedingBatch`). Only one batch (run OR expand) is active at a time, gated by
  the pre-existing `jobsRunning` flag. Each interactive slot now delegates to a
  distinctly-named non-blocking headless twin (`runSegmentationHeadless(QString*)` /
  `runExpandSeedsHeadless(QString*)`) that runs the same validation + launch and
  reports precondition failures through the out-param instead of a `QMessageBox`. The
  batch drains through the `QProcess` `finished` callbacks and resolves via
  `finalizeSeedingBatch`. `analyzePaths` dropped its per-path
  `QApplication::processEvents()` (pure synchronous compute now) and gained
  `runAnalyzePathsHeadless(QString*, int* pathsAnalyzed, int* peaksFound)`. The Cancel
  button's `onCancelClicked` slot delegates to a headless twin
  `cancelSeedingBatchHeadless()` (bounded `terminate()` + `waitForFinished(1000)` →
  `kill()` per child).

- **New `SeedingWidget` signals (binding).**
  `void seedingBatchProgressChanged(const QString& kind, int completed, int total)`
  (emitted on each child completion; `kind` is `"run"` | `"expand"`) and
  `void seedingBatchFinished(const QString& kind, bool success, bool canceled, int completed, int total, const QString& message)`
  (emitted exactly once at completion, execution failure, or cancel). The bridge connects
  these in `subscribeJobSignals()` and mirrors them onto a `source:"seeding"` job,
  exactly as the atlas search wires `atlasSearchProgressChanged`/`atlasSearchFinished`.
  Introspection getters: `seedingBatchActive()` (run/expand only — excludes a neural
  trace, which shares `jobsRunning`), `seedingBatchKind()`, `seedingBatchTotal()`.

- **Seeding batch outcome (SPEC §1 correctness).** A batch **succeeds only if every
  child process starts, exits normally, and returns exit code 0**. The widget aggregates
  per-child outcomes: `finished(exitCode, exitStatus)` and
  `errorOccurred(FailedToStart)` are both wired, de-duplicated through a
  `QSet<QProcess*>` so each child contributes to completion exactly once, and abnormal
  exit / nonzero exit / failure-to-start all count as failures. An individual failure
  does **not** abort the batch — the remaining children still run — but it is counted.
  Merged child output is drained continuously (`readyReadStandardOutput`) so a QProcess
  buffer never grows unbounded; a bounded diagnostic tail is retained for failure
  reporting. The priority wrappers `nice`/`ionice` are resolved with
  `QStandardPaths::findExecutable` (macOS has no `ionice`, and neither is guaranteed);
  when a wrapper is absent the child launches `vc_grow_seg_from_seed` directly, and
  `FailedToStart` is handled on every launch path so a missing tool fails the batch
  instead of stranding it. The terminal `success` is
  `!cancelRequested && failureCount == 0`; `finalizeSeedingBatch()` is idempotent.
  `message` carries meaningful terminal text: success `"Seeding <kind> finished: N/N"`,
  failure `"Seeding <kind> failed: X of N child processes failed; <diagnostic>"`,
  cancel `"Seeding <kind> canceled after X/N"`. The bridge's
  `handleSeedingBatchFinished` maps **success → job `succeeded`**, **execution failure →
  job `failed`** (with the failure detail), and **cancellation → job `failed`** (with the
  explicit cancel message); `canceled` distinguishes the two failure paths.

- **Job model.** `seeding.run` and `seeding.expand` are both `source:"seeding"`
  (§8.3). They are already mutually exclusive via `jobsRunning`; mapping both to one
  source expresses that through the job model too — a second run/expand while one is
  active returns `-32004` (`data.source:"seeding"`). Only bridge-initiated batches
  become tracked jobs (a human-clicked Run/Expand is not auto-registered), mirroring
  the atlas-search precedent (the widget's `seedingBatchActive()` is the lifecycle
  authority used by `jobIsRunning("seeding")`, so a human batch still reads as busy for
  the `-32004` guard). `job.progress` label is `"<kind> <completed>/<total>"`.

  | RPC | params | result | errors |
  |---|---|---|---|
  | `seeding.run` | `{}` | `{"jobId", "kind": "seeding.run", "source": "seeding", "points", "total"}` | `-32000`, `-32001`, `-32004` (`data.source:"seeding"`), `-32005` (no source collection / no points / launch failure, `data.detail`), `-32006` (`vc_grow_seg_from_seed` not found), `-32007` (`data.kind:"file"` — `seed.json`/paths dir missing), `-32010` |
  | `seeding.expand` | `{}` | `{"jobId", "kind": "seeding.expand", "source": "seeding", "iterations", "total"}` | `-32000`, `-32001`, `-32004`, `-32005`, `-32006`, `-32007` (`data.kind:"file"` — `expand.json`/paths dir missing), `-32010` |
  | `seeding.cancel` | `{}` | `{"cancelRequested": true}` | `-32007` (`data.kind:"job"` — no batch running), `-32010` |
  | `seeding.analyze_paths` | `{}` | `{"analyzed": true, "paths", "peaks"}` (synchronous, **not** a job) | `-32000`, `-32001`, `-32007` (`data.kind:"path"` — no drawn paths), `-32010` |

  `seeding.cancel` is synchronous: the bounded teardown sets the cancel flag and emits
  `seedingBatchFinished(..., canceled=true, ...)` before the RPC returns, so the job's
  terminal `job.progress` notification (state `failed`, with the explicit cancel
  message) has already gone out. A batch that runs to completion with one or more failed
  children likewise terminates as `failed` (message names the failure count and a
  bounded diagnostic tail); only an all-clean batch terminates as `succeeded`.
  Populating the source collection for `seeding.run` reuses `points.commit`
  (a committed collection becomes the combo's current selection) or the widget's own
  Cast Rays flow; `analyze_paths` requires paths drawn via Draw-mode `canvas.drag`.

**Amendment (as-built, issue #1188): remote-aware volume argument.**
`runSegmentationHeadless`/`runExpandSeedsHeadless` (and therefore the interactive
`onRunSegmentationClicked`/`onExpandSeedsClicked` slots that delegate to them) used to
pass `currentVolume->path()` verbatim as the `vc_grow_seg_from_seed` volume positional
argument. For a fully remote/streaming-only volume (no local zarr mirror — e.g. an Open
Data catalog sample attached via `catalog.open_sample`), `Volume::path()` is empty, so
the child was spawned with an **empty** volume argument and failed instantly. Both
functions now resolve the argument via a file-local `commandPathForVolume()` helper
(mirroring the one in `SegmentationCommandHandler.cpp` that already backs
`segmentation.grow_patch_from_seed`): a remote volume contributes its
`Volume::remoteLocator()` (the portable S3/HTTP zarr URL), a local volume its
`path()`. The latched `SeedingWidget::_batchVolumePath` member is now a `QString`
(the resolved command path, which may be a URL) rather than a `std::filesystem::path`.
If resolution yields an empty string (no local mirror **and** no remote locator), the
headless function fails its precondition with a distinct message
("Could not resolve a volume path for segmentation/expansion …"); through
`launchSeedingBatch` this maps to the already-documented **`-32005`**
(`data.detail` carries the message) — it does not add a new error code. This does
**not** change the `paths`/`seed.json`/`expand.json` config-file resolution, which still
derives from `Volume::path()` and so remains local-only for the seeding widget (a
separate, lower-priority concern than `startGrowPatchFromSeed`'s dynamic
`normalGridPaths()` resolution — see the issue's secondary note).

### 15.3 `segmentation.push_pull.*`

Real classes (verified): `SegmentationPushPullTool`
(segmentation/tools/SegmentationPushPullTool.hpp:26) driven through public
`SegmentationModule` API — `setAlphaPushPullConfig(const AlphaPushPullConfig&)`
(segmentation/SegmentationModule.hpp:92). Config fields: `AlphaPushPullConfig
{start, stop, step, low, high, blurRadius, computeScale, perVertexLimit, perVertex}`
(segmentation/SegmentationPushPullConfig.hpp:3–14).

**Amendment (as-built, Stage 6):** `startPushPull(int, std::optional<bool>)` and
`stopAllPushPull()` are **private** on `SegmentationModule` (they sit near line 408/410,
not the 387/389 the design draft cited), and a private `stopPushPull(int)` overload
already exists. So the bridge drives them through two new **public wrappers with
distinct names** (never same-name overloads — that broke a build in an earlier stage
and would collide with `stopPushPull(int)` here):
`bool SegmentationModule::startPushPullMode(int direction, std::optional<bool>
alphaOverride)` and `void SegmentationModule::stopPushPullAll()`, each a one-line
delegation. The current config for the read-modify-write in `set_config` is read from
`SegmentationWidget::alphaPushPullConfig()` (SegmentationWidget.hpp:46; the panel holds
it even before an edit session exists); `SegmentationModule::setAlphaPushPullConfig`
sanitizes and updates both the tool and the panel UI.

- `segmentation.push_pull.set_config` — **params:** any subset of
  `{"start": float, "stop": float, "step": float, "low": float, "high": float,
  "blurRadius": int, "computeScale": int, "perVertexLimit": float, "perVertex": bool}`
  (read-modify-write over the current config; sanitized via
  `SegmentationPushPullTool::sanitizeConfig`,
  segmentation/tools/SegmentationPushPullTool.hpp:46). **result:** the full effective
  config object. **errors:** `-32602`.
- `segmentation.push_pull.start` — **params:**
  `{"direction": "push" | "pull", "alpha"?: bool}`; mapping `push=+1`, `pull=-1` into
  `startPushPullMode(direction, alphaOverride)`. **result:** `{"active": bool}` (the
  bool returned by the tool — `false` when there is no valid hover target, which is
  **not** an error). **errors:** `-32008` (editing not enabled), `-32007`
  (`data.kind:"session"` — no active edit session), `-32602`.
- `segmentation.push_pull.stop` — **params:** none → `stopPushPullAll()`.
  **result:** `{"stopped": true}`. **errors:** none beyond `-32000`.

Recommended agent pattern (documented, binding on the docs): push/pull acts at the
module's last recorded pointer position, so position the cursor first with a
buttonless hover drag — `canvas.drag` with `button:"none"` (§9.1) ending on the target
vertex — then `start`, wait, `stop`.

### 15.4 `tracer.run_trace`

Thin adapter over `SegmentationCommandHandler::startRunTrace` (§14.4). Asynchronous;
`source:"tool"` job.

- **params:**
  ```json
  {"segmentId": str,
   "paramOverrides"?: obj = {},   // merged over <volpkg>/trace_params.json
   "ompThreads"?: int,
   "outputDir"?: str}             // default <volpkg>/traces
  ```
- **result:** `{"jobId": str, "kind": "tracer.run_trace", "source": "tool",
  "outputDir": str}`
- **errors:** `-32000`, `-32004` (`data.source:"tool"`), `-32006`
  (`vc_grow_seg_from_segments` unavailable), `-32007` (`data.kind:"segment"`, or
  `data.kind:"file"` for missing `trace_params.json`,
  SegmentationCommandHandler.cpp:2131–2134), `-32009` (remote current volume — the
  tool accepts only local volumes, SegmentationCommandHandler.cpp:2105–2112),
  `-32005` (`data.detail`).

---

## 16. MCP tool surface additions (v2)

Same conventions as §5 (snake_case, `vc3d_` prefix, params/result passthrough, RPC
errors preserved). The v1 table stands; `vc3d_grow_segment`'s `method` enum drops
`"manual_add"` per §8.1. New tools, 1:1 with the RPCs above:

| MCP tool | → RPC |
|---|---|
| `vc3d_drag` | `canvas.drag` |
| `vc3d_manual_add_begin` | `segmentation.manual_add.begin` |
| `vc3d_manual_add_finish` | `segmentation.manual_add.finish` |
| `vc3d_manual_add_set_line_mode` | `segmentation.manual_add.set_line_mode` |
| `vc3d_manual_add_set_interpolation` | `segmentation.manual_add.set_interpolation` |
| `vc3d_manual_add_undo_constraint` | `segmentation.manual_add.undo_constraint` |
| `vc3d_corrections_set_point_mode` | `segmentation.corrections.set_point_mode` |
| `vc3d_list_catalog_samples` | `catalog.list_samples` |
| `vc3d_describe_catalog_sample` | `catalog.describe_sample` |
| `vc3d_select_volume` | `volume.select` |
| `vc3d_lasagna_service_status` | `lasagna.service_status` |
| `vc3d_lasagna_ensure_service` | `lasagna.ensure_service` |
| `vc3d_lasagna_list_datasets` | `lasagna.list_datasets` |
| `vc3d_lasagna_start_optimization` | `lasagna.start_optimization` |
| `vc3d_lasagna_jobs` | `lasagna.jobs` |
| `vc3d_lasagna_cancel` | `lasagna.cancel` |
| `vc3d_lasagna_select_output` | `lasagna.select_output_segment` |
| `vc3d_lasagna_repeat_last` | `lasagna.repeat_last` |
| `vc3d_switch_workspace` | `workspace.switch` |
| `vc3d_atlas_open` | `atlas.open` |
| `vc3d_atlas_status` | `atlas.status` |
| `vc3d_atlas_search_start` | `atlas.search_start` |
| `vc3d_atlas_search_cancel` | `atlas.search_cancel` |
| `vc3d_atlas_search_results` | `atlas.search_results` |
| `vc3d_atlas_open_result` | `atlas.open_result` |
| `vc3d_atlas_remap` | `atlas.remap` |
| `vc3d_atlas_optimize_snap_candidates` | `atlas.optimize_snap_candidates` |
| `vc3d_fiber_launch` | `fiber.launch` |
| `vc3d_fiber_list` | `fiber.list` |
| `vc3d_fiber_open` | `fiber.open` |
| `vc3d_fiber_set_follow` | `fiber.set_follow` |
| `vc3d_fiber_save` | `fiber.save` |
| `vc3d_fiber_delete` | `fiber.delete` |
| `vc3d_fiber_set_tag` | `fiber.set_tag` |
| `vc3d_fiber_create_atlas` | `fiber.create_atlas` |
| `vc3d_fiber_export` | `fiber.export` |
| `vc3d_fiber_import` | `fiber.import` |
| `vc3d_set_segment_tag` | `tags.set` |
| `vc3d_seeding` (one tool per `seeding.*` RPC, same suffix) | `seeding.*` |
| `vc3d_push_pull_set_config` / `_start` / `_stop` | `segmentation.push_pull.*` |
| `vc3d_run_trace` | `tracer.run_trace` |

`vc3d_job_status` gains the optional `source` param (§8.3). The MCP server's `wait`
convenience (§5) applies to every job-returning v2 tool with the same 30-minute cap.

Tool additions in **Parts 3–7** are documented in each Part's own MCP-tool-surface
subsection rather than folded back into this v2 table: `vc3d_activate_segment` (§17.5),
`vc3d_open_catalog_sample`'s job-returning change (§18.7), `vc3d_render_tifxyz` (§19.3),
the three `vc3d_flatten_*` (§20.5), and the Part 7 autonomy tools — `vc3d_cancel_job`
(§21.3), `vc3d_wait_job` (MCP-only, §21.3), `vc3d_list_volumes` (§22.3),
`vc3d_delete_segment` / `vc3d_rename_segment` (§23.4), and `vc3d_get_render_settings` /
`vc3d_set_render_settings` (§24.3).

---

# Part 3: Segment Activation

Status: **binding design**. Everything in Parts 1–2 remains in force; Part 3 only
extends it. Section numbering continues (§17). All grounding references are to real
code on branch `feature/vc3d-agent-bridge` as of this design pass.

---

## 17. `segments.activate` — programmatic surface activation

### 17.1 The gap

Real-data testing confirmed there is **no** RPC that can make a segmentation surface
the current editing target. `CState::activeSurfaceId()` is populated only via
`CWindow::onSurfaceActivated` (CWindow.cpp:9586), reached exclusively through
`SurfacePanelController::surfaceActivated` (emitted from
`handleTreeSelectionChanged`, SurfacePanelController.cpp:1130), which fires only on a
real `QTreeWidget::itemSelectionChanged` signal — a live human click in the segment
list (connection at SurfacePanelController.cpp:258–259). The one programmatic entry
point, `selectSurfaceById()` (SurfacePanelController.cpp:1667), deliberately wraps the
tree mutation in a `QSignalBlocker` (:1690), so it highlights without activating.
Consequently `segmentation.enable_editing` fails with `-32007` on any headless session
(AgentBridgeServer.cpp ~1044), which gates off `segmentation.grow`,
`segmentation.manual_add.*`, `segmentation.corrections.*`, and the §15.3 push-pull
surface.

### 17.2 Design: first-class activation on `SurfacePanelController` (option a)

**Chosen approach:** a new public method that performs the selection **and** explicitly
emits `surfaceActivated`, so activation becomes a callable operation rather than a
signal-only side effect of a click.

This is not a new pattern — it is the promotion of an existing internal one:
`activateMaterializedSurface` (SurfacePanelController.cpp:897–915) already activates
programmatically via `selectSurfaceById(id)` + `emit surfaceActivated(...)` after an
open-data fetch completes. The new method generalizes it; the materialization path is
refactored to delegate to it (no duplicated logic).

**Rejected alternative (option b — unblock `selectSurfaceById`):** removing or
parameterizing the `QSignalBlocker` would route activation through the live
`itemSelectionChanged` path, which is wrong three ways:

1. `clearSelection()` + `setSelected(true)` fire `itemSelectionChanged` **twice**; the
   intermediate empty-selection pass through `handleTreeSelectionChanged`
   (SurfacePanelController.cpp:1091–1101) emits `surfaceSelectionCleared` and resets
   the tag UI mid-flight.
2. `handleTreeSelectionChanged` may enter `startOpenDataMaterialization`
   (SurfacePanelController.cpp:1108), which shows a modal `QProgressDialog`
   (:849–859) — forbidden inside a bridge handler (§1.3).
3. The blocker protects every existing reactive caller of `selectSurfaceById`
   (lasagna output sync via `CWindow::selectLasagnaOutputSegment` → CWindow.cpp:4403,
   §15.1 `tags.set` dispatch, surface-reload resync) from redundant reload/signal
   loops; a mode flag on that method invites misuse at those call sites.

**New public method** (declared next to `selectSurfaceById`,
SurfacePanelController.hpp:120):

```cpp
/// Programmatically activate a segment: select it in the tree (signals blocked,
/// via selectSurfaceById) and then emit surfaceActivated exactly as a live click
/// would, reaching CWindow::onSurfaceActivated (CState::setActiveSurface, the
/// "segmentation" slot swap, SegmentationModule::onActiveSegmentChanged, corr-points
/// and atlas result loading, focus-POI/orientation updates). Never shows UI.
/// Returns false with a reason in *errorMessage when the id is unknown, the surface
/// cannot be loaded, the segment is an unmaterialized open-data placeholder, or the
/// selection is locked while growth runs.
bool activateSurfaceById(const std::string& surfaceId,
                         QString* errorMessage = nullptr);
```

Binding implementation contract (SurfacePanelController.cpp):

1. `_selectionLocked` (set during growth, SurfacePanelController.hpp:128/:231) →
   return `false`, `errorMessage = "surface selection is locked while growth runs"`.
2. Resolve `surface = getSurfaceById(surfaceId)` (:2813) and locate the tree item by
   `SURFACE_ID_COLUMN` user-role data (same walk as `selectSurfaceById`, :1675–1682).
   No tree item → `false`, `errorMessage = "unknown segment: <id>"`. Tree item present
   but `surface == nullptr` → `false`, `errorMessage = "segment <id> could not be loaded"`.
3. `surface` is an open-data placeholder
   (`vc3d::opendata::isOpenDataSegmentPlaceholder(surface->path)`, the same check
   `startOpenDataMaterialization` performs at :831–832) → `false`,
   `errorMessage = "segment <id> is an open-data placeholder; fetch it first"`.
   The headless path must **not** start the interactive materialization flow
   (modal progress dialog).
4. `selectSurfaceById(surfaceId)` — **unchanged**, blocker and all. Blocking is
   *correct* here: the explicit emit in step 6 replaces the tree signal, so the
   blocker prevents double activation, not activation.
5. Sync the named CState entry exactly as `handleTreeSelectionChanged` does
   (:1113–1118): if `_state && !_state->surface(surfaceId)` →
   `_state->setSurface(surfaceId, surface, true, false)`. This is what lets
   `onSurfaceActivated` resolve multi-folder display ids kept only in CState
   (CWindow.cpp:9591–9597).
6. `emit surfaceActivated(QString::fromStdString(surfaceId), surface.get());`
   — delivered synchronously (direct connection, CWindow.cpp:8384–8385) to
   `CWindow::onSurfaceActivated`, identical to a click.
7. `applyFilters()` (mirroring the `surfaceJustLoaded` tail of
   `handleTreeSelectionChanged`, :1132–1134); return `true`.

Refactor: `activateMaterializedSurface` (:897) replaces its
`selectSurfaceById` + `emit surfaceActivated` body with a call to
`activateSurfaceById` (its surface is freshly materialized, so the placeholder guard
in step 3 correctly passes).

No changes to `selectSurfaceById`, `handleTreeSelectionChanged`, or any `CWindow`
activation handler.

### 17.3 RPC `segments.activate`

- **params:** `{"segmentId": str}` — a segment id as returned by `segments.list`
  (§3.3), including folder-qualified display ids (e.g. `"paths/foo"`).
- Dispatch: the bridge calls `_window->_surfacePanel->activateSurfaceById(...)`
  (bridge is a `CWindow` friend, §1.1). After a `true` return, the handler verifies
  `CState::activeSurfaceId() == segmentId`: `onSurfaceActivated` clears the active
  surface when the surface throws while loading (CWindow.cpp:9614–9625), and that
  post-condition failure maps to `-32005` (see errors). Activating the **already
  active** id is a no-op success (no re-emit, no side effects) — mirroring §10.4
  `volume.select` and matching the tree, where re-clicking the selected row fires no
  `itemSelectionChanged`.
- **result:**
  ```json
  {"activated": true,
   "segment": {"id": str, "path": str, "loaded": true, "active": true},
   "previousSegmentId": str | null,
   "alreadyActive": bool}
  ```
  `segment` uses the §3.3 entry shape (`path` from the loaded surface; empty string
  when unresolvable). `previousSegmentId` is the pre-call
  `CState::activeSurfaceId()`, `null` when none.
- **errors:**
  - `-32000 NO_VOLPKG`.
  - `-32602` — missing/empty `segmentId` (`data.param:"segmentId"`).
  - `-32007` — `data.kind:"segment"`, `data.id` — the id names no segment (no tree
    item / not in the package).
  - `-32005` — `data.detail` — the segment exists but cannot become active:
    unmaterialized open-data placeholder, surface failed to load, or the
    post-activation check found the active surface cleared by the
    `onSurfaceActivated` exception path.
  - `-32004` — `data.source:"growth"` — surface selection is locked while growth
    runs (`SurfacePanelController::setSelectionLocked(true)`).

**Documented side effect (binding on docs):** activating a *different* segment
disables segmentation editing, exactly as a human click does
(CWindow.cpp:9607–9612). Agents must therefore call `segments.activate` **before**
`segmentation.enable_editing`, and re-enable editing after switching segments.
A `source:"growth"` job in flight also implies the selection lock above — poll
`job.status` first.

### 17.4 Interaction with existing surface state RPCs

- `segments.list` (§3.3): the `active` field already computes
  `CState::activeSurfaceId() == id` (AgentBridgeServer.cpp:756) and therefore
  reflects `segments.activate` correctly with **no change** — activation now flows
  through `CState::setActiveSurface` like a click.
- `state.get` (§3.2): `activeSurface` likewise reflects the new id with no change.
- `lasagna.select_output_segment` (§11.7) is **unchanged and remains
  selection-only** (visual highlight via `selectSurfaceById`, no activation). Agents
  that want a lasagna output as the editing target follow it with
  `segments.activate`.
- §15.1 `tags.set` is unchanged (its documented side effect is *selection*, not
  activation).

### 17.5 MCP tool surface addition

| MCP tool | → RPC | Description shown to the agent |
|---|---|---|
| `vc3d_activate_segment` | `segments.activate` | Make a segment the active editing target (the programmatic equivalent of clicking it in the segment list). Required before `vc3d_enable_editing` / `vc3d_grow_segment` and after any segment switch. |

Same conventions as §5 (params/result passthrough, RPC errors preserved).

`vc3d_activate_segment` additionally takes an MCP-server-side `auto_fetch` flag
(default **true**, not part of the underlying RPC): when the target is an
unfetched open-data placeholder, the tool catches the `-32005`
`"…placeholder; fetch it first"` refusal, runs `vc3d_fetch_segment`
(§3.19, blocking) and then retries `segments.activate` once — mirroring a GUI
double-click, which fetches on click. The result then carries `fetched: true`.
Pass `auto_fetch=false` to surface the raw placeholder error without
downloading. Fetching a segment without activating it is `vc3d_fetch_segment`.

---

# Part 4: Asynchronous catalog open (SIGSEGV fix)

Status: **binding design**. Everything in Parts 1–3 remains in force; Part 4 amends
§3.16/§10.3 (the `catalog.open_sample` wire contract), §8.3 (job sources), and §14.1
(the `MenuActionController` entry point the bridge uses). Section numbering continues
(§18). All grounding references are to real code on branch `feature/vc3d-agent-bridge`
as of this design pass.

---

## 18. `catalog.open_sample` becomes a job; the open-data prefill completion becomes cancel-safe

### 18.1 The defect (confirmed, reproducible SIGSEGV)

Real-data testing reproduced a process-killing crash (EXC_BAD_ACCESS at address
`0x28`, byte-identical stacks across three runs) whenever `catalog.open_sample` is
called a **second** time in one VC3D process. Two cooperating defects, both in
`apps/VC3D/MenuActionController.cpp`:

1. **`openOpenDataSample` spins a nested event loop on the GUI thread.** The open
   task runs on a `QtConcurrent` worker, but the method blocks on it with a stack
   `QFutureWatcher<OpenDataOpenTaskResult>` + `QEventLoop` and
   `loop.exec(QEventLoop::ExcludeUserInputEvents)` (MenuActionController.cpp:675–701).
   Because the bridge's `handleCatalogOpenSample` (AgentBridgeServer.cpp:1800) reaches
   this method via `openOpenDataSampleById(sampleId, options, ...)`, every bridge
   catalog open violates §1.3's absolute rule: *handlers must never spin a nested
   event loop*. The nested loop pumps the whole Qt event queue mid-handler —
   delivering unrelated queued signals, and even other bridge requests, re-entrantly.

2. **The volume-prefill completion lambda is not cancel-safe.** A successful open
   ends by launching a background chunk-prefill: `startOpenDataVolumePrefill`
   (MenuActionController.cpp:782) creates a heap
   `QFutureWatcher<OpenDataVolumePrefillResult>` whose `finished` handler
   (MenuActionController.cpp:822–905) **begins with `watcher->result()`**
   (:828). Every open first calls `cancelOpenDataVolumePrefills()`
   (:578, definition :915–930), which calls `watcher->cancel()` on any still-running
   prefill watcher. `QFuture::result()` on a canceled future is undefined behavior —
   once canceled, the result is never stored, and the result-store lookup dereferences
   a null entry (the observed null-plus-small-offset `0x28` fault). The lambda's
   `try/catch` does not help: this is a segfault, not a C++ exception.

Crash sequence: open #1 succeeds and leaves prefill watcher A running → open #2
cancels watcher A (defect 2 armed) and closes the volume → open #2's nested
`loop.exec()` (defect 1) pumps the event queue and delivers watcher A's `finished`
signal → the lambda calls `result()` on the canceled future → SIGSEGV. A single open
in a fresh process never cancels a live watcher, which is why the first call always
works.

Note the two defects are separable: even without the nested loop, watcher A's
`finished` would eventually be delivered by the *main* event loop after open #2's
handler returned, and crash the same way — so fixing only the event loop would leave
a latent time-bomb, and fixing only the lambda would leave the §1.3 violation (plus
the state-interleaving hazard: open #2 runs entirely *inside* open #1's `loop.exec()`
frame, after which open #1 resumes and clobbers open #2's `setVpkg`/UI state). Both
fixes below are binding.

### 18.2 Fix A (all paths): cancel-safe prefill completion

`MenuActionController::startOpenDataVolumePrefill` — the `finished` lambda
(MenuActionController.cpp:822–905) must check cancellation **before** touching the
result:

- If `watcher->isCanceled()`, do **not** call `watcher->result()`. Synthesize
  `OpenDataVolumePrefillResult{status = Cancelled, volumeId = volumeId.toStdString(),
  message = "cancelled before completion"}` and fall through to the existing
  bookkeeping (erase from `_openDataPrefillWatchers` / `_openDataPrefillCancelFlags`,
  reset `_openDataPrefillCancelFlag` if it matches, `watcher->deleteLater()`) and the
  existing `Status::Cancelled` logging branch (:889–896).
- Otherwise the current `try { result = watcher->result(); } catch (...)` body runs
  unchanged.

`cancelOpenDataVolumePrefills()` itself is unchanged (the cancel flags already stop
the worker cooperatively; no disconnect is needed once the handler is cancel-safe —
the handler owns its own bookkeeping regardless of delivery timing). This fix alone
removes the SIGSEGV for **every** path, including the human interactive one (open
sample A, then open sample B from the catalog window while A's prefill still runs).

### 18.3 Fix B (bridge path): a genuinely asynchronous open core in `MenuActionController`

**Chosen approach: option (a)** — convert the bridge's open flow to the existing
job model, eliminating the nested event loop from the bridge call path entirely.
Rationale over option (b) (cancel/disconnect hardening only): (b) leaves
`loop.exec()` inside a bridge handler, which §1.3 forbids absolutely and which
remains a live re-entrancy hazard (any queued signal or a second bridge request is
dispatched mid-handler); a real catalog open is a multi-second-to-multi-minute
network operation, exactly the shape §2.4/§8.3 jobs exist for
(`segmentation.grow_patch_from_seed` is the precedent); and the §8.4 deferred-response
mechanism is the wrong tool here because it requires a documented max latency, while
a large sample open has no sane fixed timeout. The heavy work already runs on a
`QtConcurrent` worker — the nested loop exists *only* to fake a synchronous return,
so making the API asynchronous removes code rather than adding scaffolding.

New public surface on `MenuActionController` (MenuActionController.hpp, next to the
existing `openOpenDataSampleById` overloads):

```cpp
// Terminal outcome of an asynchronous Open Data sample open.
struct OpenDataSampleOpenOutcome {
    bool success{false};
    QString error;                                       // set iff !success
    vc3d::opendata::OpenDataSampleProjectResult result;  // attach counters/messages
};

// Asynchronous, always non-interactive open of a catalog sample (the §1.3-safe
// twin of openOpenDataSampleById(sampleId, options, ...)). Resolves `sampleId`
// against the cached Open Data manifest, then starts the open without spinning
// any nested event loop and returns immediately.
//
// Returns false (with *errorMessage) on synchronous precondition failure:
// no window/state, empty id, manifest unavailable, unknown sample, or an open
// already in flight. Returns true when the background task started; exactly one
// GUI-thread invocation of `onFinished` follows (after the project has been
// attached and the UI refreshed, or the open failed). `onProgress`, when set,
// is invoked on the GUI thread with the same OpenDataSampleDownloadProgress
// stream the interactive QProgressDialog consumes (OpenDataSegmentCache.hpp:24).
// `options.interactive` is ignored; this entry point is non-interactive by
// construction (SPEC §8.2).
bool startOpenDataSampleOpen(
    const QString& sampleId,
    const OpenDataSampleOpenOptions& options,
    std::function<void(const OpenDataSampleOpenOutcome&)> onFinished,
    std::function<void(const vc3d::opendata::OpenDataSampleDownloadProgress&)> onProgress = {},
    QString* errorMessage = nullptr);

// True while any sample open (interactive or bridge-started) is in flight.
bool openDataSampleOpenInFlight() const;
```

Binding implementation contract (MenuActionController.cpp) — a refactor of
`openOpenDataSample`, no logic duplicated:

1. The body of today's `openOpenDataSample` splits into:
   - **`beginOpenDataSampleOpenTask(sample, interactive, selection, onFinished,
     onProgress)`** (new private) — everything from `cancelOpenDataVolumePrefills()`
     + `_window->CloseVolume()` (:578–579) through the `QtConcurrent::run` launch
     (:681–697), except the watcher/loop: the `QFutureWatcher<OpenDataOpenTaskResult>`
     becomes **heap-allocated, parented to `this`** (the `OpenDataOpenTaskResult`
     struct stays .cpp-local, :85), with its `finished` slot doing
     `deleteLater()` + clearing the in-flight flag, then calling the epilogue and
     finally `onFinished`. The interactive `QProgressDialog` block (:581–597) runs
     only when `interactive`; the existing `progressCallback` additionally forwards
     to `onProgress` (marshalled to the GUI thread with the same
     `QMetaObject::invokeMethod(..., Qt::QueuedConnection)` pattern the dialog update
     uses, guarded by a `QPointer`-equivalent liveness check on `this`).
     Sets `_openDataSampleOpenInFlight = true` (new `bool` member; the header does
     not gain the .cpp-local task type).
   - **`finishOpenDataSampleOpen(OpenDataOpenTaskResult task, bool interactive,
     OpenDataSampleOpenOutcome* outcomeOut)`** (new private) — everything today
     after `watcher.result()` (:702–779): progress-dialog teardown, error handling
     (message boxes gated on `interactive` exactly as now), `setVpkg`,
     `updateRecentVolpkgList`, `refreshCurrentVolumePackageUi`, `UpdateView`,
     `startOpenDataVolumePrefill`, status-bar message.
2. `startOpenDataSampleOpen` = precondition checks (window/state, manifest via
   `_window->cachedOpenDataManifest()`, `findSample`, in-flight guard — each failure
   returns `false` with a distinct `*errorMessage` sentence) →
   `beginOpenDataSampleOpenTask(sample, /*interactive=*/false, &options.selection,
   onFinished, onProgress)` → `return true`. **No `QEventLoop` anywhere on this
   path.** Completion is signaled solely by the watcher's `finished` → epilogue →
   `onFinished` chain on the GUI thread.
3. The existing private **`openOpenDataSample(...)` keeps its synchronous signature
   and its nested loop, but only for the interactive path** (catalog-window
   double-click, whose `setOpenSampleHandler` contract needs a bool "opened"): its
   body becomes replace-project prompt (unchanged, interactive-only) → in-flight
   guard (a second interactive open while one runs returns `false` instead of
   interleaving) → `beginOpenDataSampleOpenTask(...)` with an `onFinished` that
   stores the outcome and quits a local `QEventLoop` →
   `loop.exec(QEventLoop::ExcludeUserInputEvents)` → return. This nested loop is
   **never reachable from the bridge** (see §18.4) and is §1.3-acceptable in a
   human-driven path; with Fix A and the in-flight guard it can no longer observe a
   stale prefill watcher or a second overlapping open. The two public
   `openOpenDataSampleById` overloads keep their signatures; the bridge simply stops
   calling them (they remain for the catalog window and for symmetry with §14.1).
4. Shutdown note: the heap watcher is parented to the controller, so its connection
   dies with it; a `QtConcurrent` open task cannot be interrupted and simply
   discards its result if the app exits mid-open (same as today's stack watcher,
   minus the block on exit).

### 18.4 Bridge: job source `"catalog"` and the amended `catalog.open_sample` contract

Amendment to §8.3: the source table gains a fifth row —

| `source`    | Lifecycle authority |
|-------------|---------------------|
| `"catalog"` | `MenuActionController::startOpenDataSampleOpen`'s `onFinished` callback (bridge-provided); in-flight truth via `openDataSampleOpenInFlight()` |

Amendment to §3.16/§10.3: **`catalog.open_sample` is now asynchronous.** Params,
param validation, and the §10.3 `resources` semantics are byte-for-byte unchanged
(AgentBridgeServer.cpp:1800–1955 keeps its sampleId/manifest/selection validation,
all performed synchronously before any mutation). What changes is dispatch and the
result:

- After validation, `handleCatalogOpenSample`:
  1. `requireSourceIdle("catalog")` (→ `-32004`, `data.source:"catalog"`), and
     additionally checks `mc->openDataSampleOpenInFlight()` to catch a
     *human-initiated* interactive open the bridge didn't start (→ `-32004`,
     `data.source:"catalog"`,
     `data.detail:"an interactive Open Data open is in progress"`). No external
     JobRecord is registered for human opens (there is no start signal to hook);
     the in-flight check is the authority.
  2. Calls `mc->startOpenDataSampleOpen(sampleId, options, onFinished, onProgress)`
     — `mc` is `_window->_menuController.get()` as today. A synchronous `false` →
     `-32005` with `*errorMessage` as `data.detail` (**no** job record is created,
     matching §2.5's "synchronous launch failure" meaning).
  3. On `true`: `jobId = beginJob("catalog", "catalog.open_sample",
     "open sample <id>", /*broadcastStart=*/true)` and returns immediately.
- **result:** `{"jobId": str, "kind": "catalog.open_sample", "source": "catalog",
  "sampleId": str}` — same shape family as §3.12. The old synchronous result body
  moves to the job record (below).
- **errors:** unchanged set (`-32602`, `-32007` `data.kind:"sample"|"resource"`,
  `-32005`, `-32010`) plus `-32004` (`data.source:"catalog"`).
- The `onProgress` callback maps to `job.progress` `phase:"output"` notifications
  with the same human-readable label text the interactive dialog shows
  (downloading/transforming/resolving-volumes counts), rate-limited to ≤10/s per
  §3.18.
- The `onFinished` callback resolves the active catalog job (single-threaded GUI
  ⇒ race-free lookup of `_activeJobs["catalog"]`; a new private
  `void completeCatalogOpenJob(const OpenDataSampleOpenOutcome&)` keeps it out of
  the lambda), builds the **v1/§10.3 result body** —
  `{"opened": true, "sampleId", "vpkgPath", "volumeIds", "attached", "messages"}`,
  computed exactly as AgentBridgeServer.cpp:1970–1995 does today (post-`setVpkg`
  state) — and calls `finishJob("catalog", outcome.success, message, vpkgPath)`.
  On failure, `message` is `outcome.error` and the result body is null.

**JobRecord addition (additive, amends §8.3):** `JobRecord` gains
`QJsonObject resultJson` (empty ⇒ `null` on the wire). `job.status` responses and
the terminal `job.progress` notification gain a `"result": obj | null` member —
for `source:"catalog"` jobs it carries the body above; for all other sources it is
`null` (existing sources are unchanged; the field is purely additive, per §7's
compatibility rule).

Agent recipe (documented): `catalog.open_sample` → `jobId` → poll
`job.status {jobId}` until `state != "running"` → read `result` for
`vpkgPath`/`volumeIds`/`attached`. `state.get` during the open reports
`vpkg: null` (the old project is closed synchronously in the handler) plus the
running catalog job in `jobs`.

### 18.5 What stays interactive-only

The nested `QEventLoop` survives in exactly one place: the private synchronous
`openOpenDataSample` used by the catalog window's double-click handler
(MenuActionController.cpp:469–471). It is unreachable from any bridge handler —
binding rule: **no bridge code path may call `openOpenDataSample` or either
`openOpenDataSampleById` overload; `startOpenDataSampleOpen` is the only supported
bridge entry** (extends §8.2). A bridge request arriving while that interactive loop
spins is handled re-entrantly (pre-existing Qt reality); `catalog.open_sample`
specifically is protected by the §18.4 in-flight check, and read-only RPCs observe
consistent state because the handler closes the old project before the loop starts.

### 18.6 Test-harness bindings (`apps/VC3D/agent_bridge/test/`)

1. **Timeout ⇒ liveness check (fixes the finding that masked this crash).** The
   existing catalog steps classify a client-side `TimeoutError` as a non-fatal
   "deferred" note (e.g. the unfiltered-regression step,
   bridge_client_test.py ~"DEFERRED (full remote open exceeded the offscreen
   timeout; not a failure)"). That is only valid when VC3D is still alive — a
   SIGSEGV also presents as a client timeout. Binding: the catalog suites
   (`run_catalog_resource_selection_smoke`,
   `run_catalog_strict_subset_and_volume_select`) take the `Vc3dProcess` handle
   (vc3d_process.py; `is_running()` :77, `exit_code()` :80, `tail_log()` :73) as a
   parameter, and **every** `except TimeoutError` branch that would record a
   non-failure first checks `proc.exit_code()`:
   - `None` (alive) → the existing deferred/non-fatal classification is allowed;
   - not `None` → record a **failure**: `"VC3D died during <step>: exit=<code>"`
     plus the last stdout lines from `tail_log()`, and abort the suite (the socket
     is gone). The same rule applies to any future suite: a timeout may never be
     recorded as non-fatal without a liveness check.
2. **Job-flow update.** Catalog open steps switch to the §18.4 contract: assert the
   RPC returns `{jobId, source:"catalog"}` promptly (≤10 s — the RPC itself no
   longer blocks on the network), then poll `job.status` until terminal within the
   step's overall deadline (liveness-checked on expiry per rule 1), then assert the
   old invariants (`opened`, `attached`, subset rules) against `result`.
3. **Crash regression test.** A new step performs **two** consecutive
   `catalog.open_sample` calls (filtered then unfiltered, the reproducer) in one
   process, waits for both jobs' terminal states, and asserts
   `proc.exit_code() is None` afterwards — this is the direct regression proof for
   §18.1, and must also pass with the second call issued while the first job is
   still `running` (asserting a clean `-32004` rather than a crash).

### 18.7 MCP surface note

`vc3d_open_catalog_sample` (§5) becomes job-returning like `vc3d_grow_patch_from_seed`;
the MCP server's `wait: true` convenience (§5, 30-minute cap) applies and, when used,
returns the terminal `job.status` record (including `result`) inline. No tool
rename.

---

# Part 5: Rendering

Status: **binding design** (Stage 7a). Everything in Parts 1–4 remains in force; Part 5
only extends it. All grounding references are to real code on branch
`feature/vc3d-agent-bridge`. Section numbering continues from Part 4 (§17/§18 are taken;
this is §19).

---

## 19. Rendering RPC (`render.tifxyz`)

Headless twin of the "Render" context-menu action (`SegmentationCommandHandler::
onRenderSegment`, SegmentationCommandHandler.cpp), which today pops a blocking
`RenderParamsDialog` (`dlg.exec()`) before launching `vc_render_tifxyz` via
`CommandLineToolRunner::execute(Tool::RenderTifXYZ)`. The bridge never reaches that
dialog: it drives a new non-interactive launcher that builds the runner arguments
directly. Asynchronous; `source:"tool"` job (§8.3).

### 19.1 Interactive/headless split: `startRenderSegment`

Same doctrine as §4/§14.4: shared launch logic moves to a non-interactive method that
reports via `QString* errorMessage`; the `RenderParamsDialog` and all `QMessageBox`es
stay in the interactive slot. Because the dialog collects a large **superset** of
options — crop, affine (+invert), scale-segmentation, rotate, flip, include-tifs, and
an in-render ABF++ flatten (`--flatten[-iterations/-downsample]`) — that the reduced
headless surface deliberately does not expose, the headless method is a **separate
self-contained launcher** rather than a routing of the slot through it (the slot keeps
its full advanced-option setter sequence). This mirrors how `startRunTrace`
re-implements `requireSurfaceAndRunner`'s checks inline rather than calling it.

New public types/method on `SegmentationCommandHandler`:

```cpp
struct RenderSegmentParams {
    QString volumeId;             // vpkg volume id; empty => current volume
    CommandLineToolRunner::RenderOutputFormat outputFormat{
        CommandLineToolRunner::RenderOutputFormat::TifStack};  // tif_stack | zarr
    QString outputDir;            // absolute or relative to volpkg root; empty =>
                                  // the segment folder (dialog default <segment>/layers)
    float   scale{1.0f};          // pixels per level-g voxel; > 0
    int     groupIdx{0};          // OME-Zarr group index; >= 0
    int     numSlices{1};         // number of slices; >= 1
    bool    hasVoxelSize{false};  // true => override voxel size with voxelSizeUm
    double  voxelSizeUm{0.0};     // physical voxel size (micrometers), > 0
};

/// Headless Render launch (vc_render_tifxyz). Performs ALL preconditions,
/// volume/output resolution, and launch that onRenderSegment performs today, but
/// reports failures through `errorMessage` (never via QMessageBox) and never opens
/// the RenderParamsDialog. Returns true when the tool process was launched. On
/// success `resolvedOutputDir` receives the output artifact path.
bool startRenderSegment(const std::string& segmentId,
                        const RenderSegmentParams& params,
                        QString* errorMessage = nullptr,
                        QString* resolvedOutputDir = nullptr);
```

The output-format choice is threaded through a new
`CommandLineToolRunner::RenderOutputFormat { TifStack, Zarr }` enum plus
`setRenderOutputFormat(RenderOutputFormat)`. `buildArguments(Tool::RenderTifXYZ)` —
which previously **always** emitted `--tif-output` — now emits `--zarr-output` for
`Zarr` and `--tif-output` for `TifStack`. The default is `TifStack` (preserving the
existing GUI behavior); `onRenderSegment` **re-asserts** `TifStack` before launch so a
prior headless Zarr render cannot leak its selection into a subsequent interactive
render (the runner is a shared long-lived object).

`startRenderSegment` re-implements `onRenderSegment`'s preconditions and `execute()`'s
pre-launch checks inline (distinct sentences the bridge matches), so `execute()` never
reaches one of its own blocking `QMessageBox` paths on a headless run:

- no vpkg → `"No volume package or volume loaded"`; no current volume →
  `"No volume loaded"`.
- unknown segment → `"Invalid segment or segment not loaded: <id>"`.
- runner missing → `"Command line tools not available"`; runner busy →
  `"A command line tool is already running"`.
- **tool binary missing** → `"vc_render_tifxyz not found or not executable: <path>"`
  (checked with the same `applicationDirPath()`-relative path `toolName()` builds,
  because `execute()` would otherwise pop a `QMessageBox` **and** show the console
  dialog on a missing binary — a §1.3 hang for an unattended run).
- unknown `volumeId` → `"Unknown volume id: <id>"`.
- output-dir creation failure → `"Failed to create output directory: ..."`.

Volume resolution: `volumeId` empty → current volume; otherwise validated against
`VolumePkg::volumeIDs()` and resolved via the file's `commandPathForVolume` helper.
Render (unlike Run Trace) accepts **remote** volumes, so they are not rejected; remote
URL + auth are configured through the existing
`configureCommandRunnerRemoteAuthForVolumePath(volumePath)` helper. Voxel size is either
the explicit `voxelSizeUm` override or derived from the resolved volume exactly as the
interactive path does (`voxelSize()` + `baseScaleLevel()>0 || hasExplicitVoxelSizeOverride()`).
Output artifact path: `<baseDir>/layers` for a TIFF stack, `<baseDir>/surface.zarr` for
Zarr, where `baseDir` is `outputDir` (absolute or volpkg-relative) or the segment folder
by default; `baseDir` is created if missing. Advanced options are reset to defaults
(`setRenderAdvanced(0,0,0,0,"",false,1,0,-1)`, `setIncludeTifs(false)`,
`setFlattenOptions(false,10,1)`, `setOmpThreads(-1)`) so a prior interactive render's
dialog settings cannot leak into the reduced-surface headless render.

### 19.2 RPC `render.tifxyz`

Thin adapter over `startRenderSegment`. Asynchronous; `source:"tool"` job. Reuses the
exact `setSuppressCompletionDialogs(true)`-before-launch pattern proven for
`segmentation.grow_patch_from_seed` / `tracer.run_trace` (cleared on the synchronous
failure path; auto-cleared on `toolFinished`).

- **params:**
  ```json
  {"segmentId": str,                          // required
   "outputFormat": "zarr" | "tif_stack",      // required; the headline capability
   "volumeId"?: str,                          // default: current volume id
   "outputDir"?: str,                         // absolute or relative to volpkg root;
                                              // default: the segment folder
   "scale"?: float = 1.0,                      // > 0
   "groupIdx"?: int = 0,                       // OME-Zarr group index; >= 0
   "numSlices"?: int = 1,                      // >= 1
   "voxelSize"?: float}                        // micrometers; omit to use volume metadata
  ```
- **result:** `{"jobId": str, "kind": "render.tifxyz", "source": "tool",
  "outputDir": str, "outputFormat": str, "volumeId": str}` — `outputDir` is the output
  artifact path (the layers dir or the `.zarr` store). Completion is a
  `job.progress` (`phase:"finished"`, `source:"tool"`) notification and is observable
  via `job.status`.
- **errors:** `-32000` (no vpkg), `-32001` (no current volume),
  `-32004` (`data.source:"tool"` — a tool job is already running),
  `-32005` (`data.detail` — output-dir create / generic launch failure),
  `-32006` (`vc_render_tifxyz` unavailable),
  `-32007` (`data.kind:"segment"` unknown segment, or `data.kind:"volume"` unknown
  `volumeId`),
  `-32009` (segmentation command handler unavailable),
  `-32602` (missing `segmentId`; bad `outputFormat`; non-positive/non-finite `scale`;
  negative `groupIdx`; `numSlices` < 1; non-positive/non-finite `voxelSize`).

Deliberately **not** exposed on this core surface (future work): `--accum`/`--accum-type`,
`--composite-*`, alpha / Beer-Lambert params, `--iso-cutoff`, `--slice-step`,
`--pyramid`, `--resume`, `--num-parts`/`--part-id`, crop, affine, scale-segmentation,
rotate, flip, include-tifs, and the in-render `--flatten` ABF++ option.

### 19.3 MCP tool

| MCP tool | → RPC |
|---|---|
| `vc3d_render_tifxyz` | `render.tifxyz` |

Snake_case params 1:1 with §19.2 (`output_format`, `volume_id`, `output_dir`,
`group_idx`, `num_slices`, `voxel_size`), plus the standard MCP-server-side
`wait: bool = false` convenience (§5; folds the terminal `job.status` inline on
completion, 30-minute cap).

---

# Part 6: Flattening

Status: **binding design** (Stage 7b). Everything in Parts 1–5 remains in force; Part 6
only extends it. All grounding references are to real code on branch
`feature/vc3d-agent-bridge`. Section numbering continues from Part 5 (§19 is taken;
this is §20).

---

## 20. Flattening RPCs (`flatten.slim` / `flatten.abf` / `flatten.straighten`)

Headless twins of the three flattening context-menu actions
(`SegmentationCommandHandler::onSlimFlatten` / `onABFFlatten` / `onStraighten`,
SegmentationCommandHandler.cpp), each of which today pops a blocking params dialog
(`SlimFlattenDialog` / `ABFFlattenDialog` / `StraightenDialog`, `dlg.exec()`) and then
constructs a bespoke `QObject` job class (`SlimJob` / `ABFJob` / `StraightenJob`) with
its **own** `QProgressDialog` and several terminal `QMessageBox` calls. The bridge never
reaches any of these dialogs. All three are **asynchronous** and share a **new job
source** `"flatten"` (§8.3): they run concurrently with `tool`/`growth`/`lasagna`/
`atlas`/`catalog` jobs but only one flatten may run at a time (`-32004` otherwise).

### 20.1 Job source `"flatten"` and the suppression capability

Amendment to §8.3: `"flatten"` is added as a sixth job source. Its lifecycle authority is
**two new signals on `SegmentationCommandHandler`**, emitted by all three job classes from
**both** the interactive slots and the headless launchers:

```cpp
void flattenJobStarted(QString kind, QString label);   // kind: "flatten.slim" | ...
void flattenJobFinished(bool success, QString message, QString outputPath);
```

The bridge wires these in `subscribeJobSignals()` to `handleFlattenStarted` /
`handleFlattenFinished`, exactly mirroring the `tool`/`growth` external-job pattern: a
human-initiated flatten registers as a `"flatten.<type>"` external `JobRecord` (so
`-32004` and `state.get` reflect true app state); a bridge-initiated flatten adopts the
concrete kind/label into the record the RPC already created. `flattenJobFinished` with
`success=false` and an **empty** message denotes a user cancel (the bridge surfaces it as
`"Flatten cancelled"`).

Unlike rendering (which already had `CommandLineToolRunner::setSuppressCompletionDialogs`),
the three flatten job classes had **zero** suppression mechanism. Each gained a
constructor `bool suppressDialogs` (defaulting to `false`, so the interactive
`new SlimJob(...)` / `new ABFJob(...)` / `new StraightenJob(...)` calls are unchanged) and
a `emitFinishedOnce_()` helper. When `suppressDialogs` is true:

- the `QProgressDialog` is created but its `minimumDuration` is set to
  `INT_MAX`, so its auto-show timer never fires — the dialog stays invisible while every
  `setValue`/`setLabelText` call remains a safe no-op on a live object (no class ever
  calls `show()`/`exec()` on it);
- every `QMessageBox` call site (static **and** `open()`-based) is gated: the message text
  is captured and delivered via `flattenJobFinished` instead of shown, and the cleanup /
  `deleteLater()` that the box's `finished` handler would have performed runs inline.

The headless launchers (`startSlimFlatten` / `startAbfFlatten` / `startStraighten`)
construct the job with `suppressDialogs=true`. They pre-resolve every required executable
**before** constructing the job, so the job's synchronous `showImmediateToolNotFound_`
(a blocking `QMessageBox`) path is never reached on a headless run. All three report
failures through `QString* errorMessage` with distinct sentences the bridge classifies,
and never open a params dialog.

### 20.2 `flatten.slim`

The production-recommended method. Headless default `keepPercent` is **100**
(full-resolution SLIM, no decimation) per current internal R&D guidance — not the
dialog's 1.5% session default. Backed by `SegmentationCommandHandler::startSlimFlatten`,
which pre-resolves `flatboi`, `vc_tifxyz2obj`, `vc_obj2tifxyz` (and `vc_obj_uv_lift` when
`keepPercent < 100`), then constructs a suppressed `SlimJob`.

- **params:**
  ```json
  {"segmentId": str,                  // required
   "iterations"?: int = 50,           // flatboi iterations, >= 1
   "tolerance"?: float = 0.0,         // 0 => no --tol (run all iterations); >= 0
   "energyType"?: str = "symmetric_dirichlet",  // or "conformal"
   "keepPercent"?: float = 100.0,     // in (0, 100]; 100 => full-res (no decimation)
   "inpaintHoles"?: bool = false,
   "outputDir"?: str}                 // absolute or relative to volpkg root;
                                      // default <segment>_flatboi
  ```
- **result:** `{"jobId": str, "kind": "flatten.slim", "source": "flatten",
  "outputDir": str}` — `outputDir` is the flattened tifxyz directory. Completion is a
  `job.progress` (`phase:"finished"`, `source:"flatten"`) notification, observable via
  `job.status`.
- **errors:** `-32000` (no vpkg), `-32001` (no current volume),
  `-32004` (`data.source:"flatten"` — a flatten job is already running),
  `-32006` (`flatboi` / `vc_tifxyz2obj` / `vc_obj2tifxyz` / `vc_obj_uv_lift`
  unavailable — `data.detail`),
  `-32007` (`data.kind:"segment"` — unknown segment),
  `-32009` (segmentation command handler unavailable),
  `-32005` (`data.detail` — generic launch failure),
  `-32602` (missing `segmentId`; `iterations` < 1; negative/non-finite `tolerance`; bad
  `energyType`; `keepPercent` outside (0, 100]; non-bool `inpaintHoles`).

### 20.3 `flatten.abf`

Simplest: `ABFJob` runs **in-process** via `QtConcurrent::run` calling
`vc::abfFlattenToNewSurface` directly (no subprocess). Backed by `startAbfFlatten`.

- **params:**
  ```json
  {"segmentId": str,                  // required
   "iterations"?: int = 10,           // ABF++ iterations, >= 1
   "downsampleFactor"?: int = 1}      // >= 1
  ```
- **result:** `{"jobId": str, "kind": "flatten.abf", "source": "flatten",
  "outputDir": str}` — `outputDir` is `<segment>_abf`. On success the surface panel is
  reloaded (same as the interactive path), so the new segment is observable via
  `segments.list`.
- **errors:** `-32000` (no vpkg), `-32004` (`data.source:"flatten"`),
  `-32007` (`data.kind:"segment"`), `-32009`, `-32005` (`data.detail`),
  `-32602` (missing `segmentId`; `iterations` < 1; `downsampleFactor` < 1).
  (No `-32001`/`-32006`: ABF++ needs no current volume and no external tool.)

### 20.4 `flatten.straighten`

`StraightenJob` runs a single `vc_straighten` subprocess (tifxyz → tifxyz, no OBJ
round-trip). Backed by `startStraighten`, which resolves `vc_straighten` and rejects an
existing output directory up front (the tool refuses to overwrite), then builds the CLI
args exactly as `StraightenDialog::toArgs()` does.

- **params:**
  ```json
  {"segmentId": str,                  // required
   "unbend"?: bool = true,            // --unbend stage
   "unbendSmoothCols"?: float = 300,  // --unbend-smooth-cols (only when unbend)
   "overlapPasses"?: int = 2,         // --overlap-pairs (always emitted), >= 0
   "orthogonalize"?: bool = true,     // --orthogonalize
   "trim"?: bool = true,              // --trim stage
   "trimMaxEdge"?: float = 100,       // --trim-max-edge (only when trim)
   "outputDir"?: str}                 // absolute or relative to volpkg root;
                                      // default <segment>_straightened
  ```
- **result:** `{"jobId": str, "kind": "flatten.straighten", "source": "flatten",
  "outputDir": str}` — `outputDir` is the straightened tifxyz directory. On success the
  surface panel is reloaded, so the new segment is observable via `segments.list`.
- **errors:** `-32000` (no vpkg), `-32001` (no current volume),
  `-32004` (`data.source:"flatten"`),
  `-32006` (`vc_straighten` unavailable — `data.detail`),
  `-32007` (`data.kind:"segment"`), `-32009`,
  `-32005` (`data.detail` — output directory already exists / generic launch failure),
  `-32602` (missing `segmentId`; non-finite/negative `unbendSmoothCols`; `overlapPasses`
  < 0; non-finite/negative `trimMaxEdge`; non-bool `unbend`/`orthogonalize`/`trim`).

### 20.5 MCP tools

| MCP tool | → RPC |
|---|---|
| `vc3d_flatten_slim` | `flatten.slim` |
| `vc3d_flatten_abf` | `flatten.abf` |
| `vc3d_flatten_straighten` | `flatten.straighten` |

Snake_case params 1:1 with §20.2–§20.4 (`energy_type`, `keep_percent`, `inpaint_holes`,
`output_dir`, `downsample_factor`, `unbend_smooth_cols`, `overlap_passes`,
`trim_max_edge`), plus the standard MCP-server-side `wait: bool = false` convenience
(§5; folds the terminal `job.status` inline on completion, 30-minute cap).

---

# Part 7: Agent autonomy — job cancel, volume discovery, segment delete/rename, viewer render settings

Status: **binding design** (Stage 8). Everything in Parts 1–6 remains in force; Part 7
only extends it, plus one additive amendment to §3.2 `state.get` (§22.2). **No new error
codes** — the §2.5 table stands verbatim; error-specific `data` fields are documented per
method. Section numbering continues from Part 6 (§20 is taken; this is §21). All grounding
references are to real code on branch `feature/vc3d-agent-bridge`.

---

## 21. Generic job cancellation (`job.cancel`)

Parts 1–2 grew a source-tagged job model (§2.4, §8.3) but offered only **domain-specific**
cancels: `lasagna.cancel` (§11.6), `seeding.cancel` (§15.2), and `atlas.search_cancel`
(§12). There was no single, source-agnostic "stop this job" verb an agent could reach for
after starting any cancellable job. `job.cancel` is that verb: it resolves a JobRecord
(§8.3), reads its `source`, and dispatches to whatever cancel authority that source has —
reusing the existing domain cancels underneath rather than adding a new mechanism.

### 21.1 `job.cancel`

- **params:** `{"jobId"?: str, "source"?: str}` — **at least one** is required; prefer
  `jobId` (unambiguous). The job is resolved via `jobById(jobId)` when `jobId` is given,
  else `activeJobId(source)` (the active job of that source).
- **result:** `{"cancelRequested": true, "jobId": str, "source": str, "kind": str}` —
  echoing the resolved record. Cancellation is **request-only** (like §11.6/§15.2): the
  terminal state still arrives asynchronously via `job.progress` / `job.status`. A
  successful call means the stop was dispatched, not that the job has already ended.
- **errors:**
  - `-32602` (`data.param:"jobId"`) — neither `jobId` nor `source` was given.
  - `-32007` (`data.kind:"job"`, plus `data.id` when `jobId` was given or `data.source`
    when only `source` was) — no such job, or no active job of that source.
  - `-32010` (`data.kind:"job"`, `data.reason:"not cancellable"`, `data.source`) — the
    job's source has no cancel authority (§21.2).
  - `-32005` — an authority-not-running edge (the underlying cancel authority reports
    nothing to cancel).

Dispatch is on `job->source`:

| `source`    | Cancel authority invoked |
|-------------|--------------------------|
| `"atlas"`   | `cancelAtlasFiberIntersectionSearch()` (the §12 search) |
| `"seeding"` | `cancelSeedingBatchHeadless()` (§15.2) |
| `"lasagna"` | `stopOptimization()` (no `externalId`) / `cancelJob(externalId)` (§11.6) |
| `"tool"`    | `CommandLineToolRunner::cancel()` on the child `QProcess` — covers the `tracer.run_trace` (§15.4) and `render.tifxyz` (§19.2) tool jobs |

### 21.2 Cancellable-vs-not: a real limitation

Not every job source can be stopped once started. This is a hard property of the
underlying app APIs, not a bridge choice, and agents must plan around it:

| `source`     | Cancellable? | Why |
|--------------|:---:|-----|
| `"tool"`     | **yes** | child `QProcess` — `CommandLineToolRunner::cancel()` |
| `"atlas"`    | **yes** | `cancelAtlasFiberIntersectionSearch()` |
| `"seeding"`  | **yes** | `cancelSeedingBatchHeadless()` (bounded teardown, §15.2) |
| `"lasagna"`  | **yes** | `stopOptimization()` / `cancelJob(externalId)` (§11.6) |
| `"growth"`   | **no**  | `segmentation.grow` runs on a `QtConcurrent` future with no cancel API |
| `"flatten"`  | **no**  | the §20 flatten jobs own self-scheduled ephemeral objects with no exposed cancel handle |
| `"catalog"`  | **no**  | no cancel handle on the async open core (§18) |
| `"autosave"` | **no**  | short-lived explicit save (§3.11c); no cancel handle |

For a **non-cancellable** source, `job.cancel` returns `-32010`
`data.reason:"not cancellable"` rather than silently succeeding — so an agent can tell
"I asked it to stop and it will" from "this job cannot be stopped; wait it out." The four
cancellable sources reuse exactly the domain cancels of §11.6 / §15.2 / §12; `job.cancel`
is the uniform front door on top of the §8.3 source-tagged model, not a new capability.

### 21.3 MCP tool surface

| MCP tool | → RPC | Description shown to the agent |
|---|---|---|
| `vc3d_cancel_job` | `job.cancel` | Request cancellation of a running job by `jobId` (or by `source`). Only `tool`/`atlas`/`seeding`/`lasagna` jobs are cancellable; `growth`/`flatten`/`catalog`/`autosave` return an error. Request-only — poll `vc3d_job_status` for the terminal state. |

Same conventions as §5 (params/result passthrough, RPC errors preserved).

**MCP-only `vc3d_wait_job` (no RPC).** Alongside the `wait: true` convenience param (§5) —
which blocks on a job the *same* tool call just started — the MCP server exposes a
standalone `vc3d_wait_job` tool that blocks on an **already-running** job by id. It has no
RPC of its own: it wraps the identical `job.status` polling loop (once/sec, 30-minute cap,
`consoleTail` forwarded as best-effort progress via the FastMCP `Context`) and returns the
terminal `job.status` record inline, or the last-seen status with `"waitTimedOut": true` on
the cap. Use it to wait on a job started earlier with `wait=false`, or on an
externally-initiated job seen in `state.get`. Documented alongside the `wait` convention in
§5, not as a 1:1 RPC tool.

---

## 22. Volume discovery (`volume.list`) and the `state.get` amendment

### 22.1 `volume.list`

Enumerate the volumes in the open volume package — the vpkg-level discovery path that does
**not** depend on a current volume being selected.

- **params:** none.
- **result:** `{"volumeIds": [str], "currentVolumeId": str | null}` — `volumeIds` from
  `state->vpkg()->volumeIDs()`; `currentVolumeId` is the currently-selected volume id, or
  `null` when a vpkg is open but no volume is current.
- **errors:** `-32000 NO_VOLPKG`.

Deliberately **omitted**: a richer per-volume `[{id, path, voxelSize}]` array. Reading
`voxelSize` forces a volume's metadata to load, and for a vpkg backed by remote/lazy volumes
that would force-load **every** volume (potentially remote) just to answer a listing — the
exact cost `volume.list` exists to avoid. Ids alone are enough to then `volume.select`
(§10.4) one and read its `voxelSize` from `state.get`.

### 22.2 `state.get` amendment: `volume.volumeIds` (additive)

Amendment to §3.2. The `volume` block gains a `"volumeIds": [str]` member (from
`state->vpkg()->volumeIDs()`):

```json
"volume": null | {"id": str, "path": str, "voxelSize": float, "volumeIds": [str]}
```

This is an **additive v2-era field** (per §7's compatibility rule: new field only, nothing
removed or retyped).

**Deliberate placement decision:** `volumeIds` lives **nested inside** `volume`, not at the
top level. Consequently it is **absent whenever `volume` is `null`** — i.e. when a vpkg is
open but no current volume is selected. That is intentional: the current-volume-independent
way to enumerate volumes is `volume.list` (§22.1), which works regardless of current-volume
state and returns `currentVolumeId` alongside. `state.get`'s nested copy is a convenience
for the common "a volume is already current" case, so an agent that already called
`state.get` need not make a second round-trip.

### 22.3 MCP tool surface

| MCP tool | → RPC | Description shown to the agent |
|---|---|---|
| `vc3d_list_volumes` | `volume.list` | List the volume ids in the open volume package and which one is current, without force-loading any (possibly remote) volume. |

Same conventions as §5 (params/result passthrough, RPC errors preserved).

---

## 23. Segment lifecycle (`segments.delete` / `segments.rename`)

Two mutating segment operations a human performs from the segment-list context menu. Both
are **synchronous** (no job), and both required an interactive/headless split (§23.3)
because the GUI slots embed modal dialogs.

### 23.1 `segments.delete`

Delete a segment from disk. **Irreversible** (on-disk removal), so it is confirm-gated.

- **params:** `{"segmentId": str, "confirm": bool}` — `confirm` **must** be `true`.
- **result:** `{"deleted": [str]}` — the deleted segment id(s).
- **errors:**
  - `-32000 NO_VOLPKG`.
  - `-32602` (`data.param:"confirm"`, `data.reason:"destructive; pass confirm=true"`) —
    `confirm` was not `true` (on-disk deletion is irreversible).
  - `-32004` (`data.detail:"cannot delete while editing"`) — segmentation editing is
    enabled; disable it first.
  - `-32007` (`data.kind:"segment"`, `data.id`) — unknown segment.
  - `-32010` (`data.detail`) — core deletion failure.

Deleting the **active** segment is allowed: the core clears the active slot **before**
removing the on-disk data, so there is no use-after-free on the just-deleted surface.

### 23.2 `segments.rename`

Rename a segment (its id and on-disk folder).

- **params:** `{"segmentId": str, "newName": str}`.
- **result:** `{"oldId": str, "newId": str}`.
- **errors:**
  - `-32000 NO_VOLPKG`.
  - `-32602` (`data.param:"newName"`) — `newName` fails the id regex `^[a-zA-Z0-9_-]+$`,
    or equals the current name (unchanged).
  - `-32004` — segmentation editing is enabled; disable it first.
  - `-32007` (`data.kind:"segment"`, `data.id`) — unknown source segment.
  - `-32010` (`data.kind:"segment"`, `data.id:<newName>`,
    `data.reason:"target name already exists"`) — a segment named `newName` already exists.

### 23.3 Interactive/headless split

Same doctrine as §4/§14/§19.1: shared logic moves to a **dialog-free core**; the modal
dialog stays in the interactive slot; no logic is duplicated.

- **Delete.** The GUI slot `SurfacePanelController::handleDeleteSegments` contained a
  `QMessageBox::question` confirmation. The dialog-free core
  `deleteSegmentsHeadless(QStringList, QString* err)` was extracted; the slot now runs
  "dialog → core", and the bridge calls the core directly (the RPC's `confirm=true` gate
  stands in for the human's dialog "Yes").
- **Rename.** The GUI slot `SegmentationCommandHandler::onRenameSurface` contained a
  `QInputDialog::getText`. The dialog-free core
  `renameSurfaceHeadless(oldId, newId, QString* err)` was extracted the same way.

`renameSurfaceHeadless` reports failures through short, **untranslated** machine-read
sentinel strings — never `tr()`-wrapped user prose — so that **both** the GUI slot and the
bridge can match on them to choose the user-facing message / the JSON-RPC error code. The
sentinels and their bridge mappings:

| sentinel (`err`)          | bridge error |
|---------------------------|--------------|
| `"name exists"`           | `-32010` (`data.reason:"target name already exists"`) |
| `"invalid name"`          | `-32602` (`data.param:"newName"`) |
| `"name unchanged"`        | `-32602` (`data.param:"newName"`) |
| `"editing in progress"`   | `-32004` |
| `"segment not found"`     | `-32007` (`data.kind:"segment"`) |

### 23.4 MCP tool surface

| MCP tool | → RPC | Description shown to the agent |
|---|---|---|
| `vc3d_delete_segment` | `segments.delete` | Delete a segment from disk. **Irreversible** — requires `confirm=true`. Fails while segmentation editing is enabled; deleting the active segment is allowed. |
| `vc3d_rename_segment` | `segments.rename` | Rename a segment (id + folder). `new_name` must match `^[a-zA-Z0-9_-]+$`, differ from the current name, and not collide. Fails while editing is enabled. |

Same conventions as §5 (params/result passthrough, RPC errors preserved).

---

## 24. Viewer render settings (`viewer.get_render_settings` / `viewer.set_render_settings`)

Read and write the shared viewer render/overlay settings — the intersection-line, overlay,
and surface-normal / direction-hint toggles a human sets from the viewer controls.

### 24.1 `viewer.get_render_settings` / `viewer.set_render_settings`

Both return the **full** settings object with these exact keys:

```json
{"intersectionOpacity": float,          // 0..1
 "intersectionThickness": float,         // >= 0
 "overlayOpacity": float,                // 0..1
 "intersectionMaxSurfaces": int,         // >= 0
 "volumeWindow": {"low": float, "high": float},  // each 0..255; high forced > low
 "samplingStride": int,                  // >= 1 (floored)
 "zScrollSensitivity": float,            // 0.1..100
 "segmentationCursorMirroring": bool,
 "planeIntersectionLinesVisible": bool,
 "showSurfaceNormals": bool,
 "showDirectionHints": bool,
 "surfaceOverlayEnabled": bool,
 "normalArrowLengthScale": float,        // 0.1..2.0
 "normalMaxArrows": int,                 // 4..100
 "highlightedSurfaceIds": [str]}
```

**`viewer.get_render_settings`**
- **params:** none.
- **result:** the settings object above.
- **errors:** none.

**`viewer.set_render_settings`**
- **params:** any **subset** of the keys above (all optional; unknown keys are ignored).
  Opacities clamp to `0..1`; `intersectionThickness` / `intersectionMaxSurfaces` clamp to
  `>= 0`; `volumeWindow.low`/`volumeWindow.high` clamp to `0..255` each (`high` is forced to
  stay `> low`, minimum gap `1.0`, same rule as `viewer.set_overlay`'s `window`);
  `samplingStride` floors to `>= 1`; `zScrollSensitivity` clamps to `0.1..100`;
  `segmentationCursorMirroring` accepts any bool; `normalArrowLengthScale` clamps to
  `0.1..2.0` (the GUI slider's range); `normalMaxArrows` clamps to `4..100` (the GUI
  slider's range); `highlightedSurfaceIds` **replaces** the list wholesale. A key present
  with a wrong-typed value → `-32602` (`data.param` names the offender).
- **result:** the full settings object **after** applying the change (same shape as get).
- **errors:** `-32602` (`data.param` — a provided key had the wrong type).

Grounding: `intersectionOpacity`/`intersectionThickness`/`overlayOpacity`/
`intersectionMaxSurfaces`/`volumeWindow`/`samplingStride`/`zScrollSensitivity` are
`ViewerManager` setters/getters; `segmentationCursorMirroring` is a `CWindow` member
(`_mirrorCursorToSegmentation`) with its own QSettings key; the remaining toggle/highlight
fields (including `normalArrowLengthScale`/`normalMaxArrows`) are per-viewer
`CChunkedVolumeViewer` toggles broadcast to every live viewer via `forEachBaseViewer`.

### 24.2 Known limitation: ViewerManager-backed vs live-viewer-only fields

The field groups have **different persistence**, and the echoed result can disagree
with the request when no viewer is open — a real limitation surfaced in review:

- **ViewerManager/window-backed (persist without a viewer):** `intersectionOpacity`,
  `intersectionThickness`, `overlayOpacity`, `intersectionMaxSurfaces`, `volumeWindow`,
  `samplingStride`, `zScrollSensitivity`, `segmentationCursorMirroring`. Setting these
  updates `ViewerManager` (or, for `segmentationCursorMirroring`, `CWindow`) state and
  QSettings immediately; the echoed result reflects the set value even with **zero**
  viewers instantiated.
- **Live-viewer-only (no-op without a viewer):** `planeIntersectionLinesVisible`,
  `showSurfaceNormals`, `showDirectionHints`, `surfaceOverlayEnabled`,
  `highlightedSurfaceIds`. These apply by broadcasting to live `CChunkedVolumeViewer`s via
  `forEachBaseViewer`. With **zero** viewers instantiated the broadcast reaches nothing, so
  the set is a **no-op** and the echoed result reports the **defaults** for these fields
  rather than the values just requested. Once at least one viewer exists they behave
  normally.
- **Exception — `normalArrowLengthScale` / `normalMaxArrows`:** these two are per-viewer
  `CChunkedVolumeViewer` toggles like the group above, but the setter **unconditionally**
  persists them to the same QSettings keys the "ViewerNormalVisualizationPanel" sliders
  use (`normalArrowLengthScale` stored as an int percent, `normalArrowLengthScale * 100`)
  *before* broadcasting, and the getter falls back to those same QSettings keys whenever no
  chunked viewer is live. So — unlike the four toggle/highlight fields above — their echoed
  value matches the request even with **zero** viewers instantiated; only the live-viewer
  broadcast itself is a no-op in that case.

Agents that need the five live-viewer-only fields (`planeIntersectionLinesVisible`,
`showSurfaceNormals`, `showDirectionHints`, `surfaceOverlayEnabled`,
`highlightedSurfaceIds`) to stick should ensure a viewer is open (the default workspace has
them, §2.2) before setting them, and should treat those fields as viewer-scoped rather than
global.

### 24.3 MCP tool surface

| MCP tool | → RPC | Description shown to the agent |
|---|---|---|
| `vc3d_get_render_settings` | `viewer.get_render_settings` | Read the shared viewer render/overlay settings (intersection lines, overlay opacity, surface normals, direction hints, highlighted surfaces). |
| `vc3d_set_render_settings` | `viewer.set_render_settings` | Set any subset of the viewer render/overlay settings; returns the full settings after applying. The five toggle/highlight fields are no-ops when no viewer is open (§24.2). |

Same conventions as §5 (snake_case params 1:1 with §24.1, params/result passthrough, RPC
errors preserved).

---

## 25. Viewer overlay + intersects

Round-2 viewer controls: the overlay-volume settings the GUI's overlay combo box +
sliders drive (`viewer.get_overlay` / `viewer.set_overlay` / `viewer.list_overlay_volumes`,
all backed by `ViewerManager`'s `_overlay*` state), and the per-viewer intersection-line
surface set (`viewer.set_intersects`, mirroring `SurfacePanelController`'s per-viewer
`setIntersects()` calls).

### 25.1 `viewer.get_overlay`

- **params:** none.
- **result:**
  ```json
  {"volumeId": str,           // "" when no overlay volume is set
   "colormap": str,           // "" is the "no explicit choice" sentinel
   "opacity": float,          // 0..1
   "threshold": float,        // alias of windowLow (see 25.2)
   "windowLow": float,        // 0..255
   "windowHigh": float,       // 0..255, forced > windowLow
   "maxDisplayedResolution": int,   // 0..5
   "composite": {"enabled": bool, "method": "max"|"mean"|"min",
                 "layersFront": int, "layersBehind": int}}  // layers 0..64
  ```
- **errors:** `-32010` (`data.detail` — viewer manager unavailable).

### 25.2 `viewer.set_overlay`

- **params:** any **subset** of:
  ```json
  {"volumeId"?: str,          // "" or omitting-with-clear:true means "no overlay"
   "clear"?: bool,            // true clears the overlay volume; equivalent to volumeId:""
   "colormap"?: str,          // one of the ids in the table below, or "" to clear
   "opacity"?: float,         // clamps 0..1
   "threshold"?: float,       // clamps >= 0 (see note below)
   "window"?: {"low": float, "high": float},  // each clamps 0..255; high forced > low
   "maxDisplayedResolution"?: int,   // clamps 0..5
   "composite"?: {"enabled"?: bool, "method"?: "max"|"mean"|"min",
                  "layersFront"?: int, "layersBehind"?: int}}  // layers clamp 0..64
  ```
  Absent fields leave the current setting untouched. `composite` is **merged** over the
  current composite settings (an agent may send just `{"enabled": false}`), not replaced
  wholesale.
- **result:** the full overlay settings object (same shape as `viewer.get_overlay`),
  reflecting what actually stuck (see the coordinate-space note below).
- **errors:**
  - `-32010` (`data.detail`) — viewer manager unavailable.
  - `-32000` — no volume package loaded (only reachable when a non-empty `volumeId` is
    supplied, since resolving it requires the vpkg).
  - `-32007` (`data.kind:"volume"`, `data.id`) — `volumeId` does not name a known volume.
  - `-32602` (`data.param:"colormap"`) — `colormap` is not one of the known ids (below).
  - `-32602` (`data.param:"composite.method"`) — `composite.method` is not
    `"max"`/`"mean"`/`"min"`.
  - `-32602` (`data.param`) — a wrong-typed/malformed field (`window` missing `low`/`high`,
    a non-object `composite`, etc).

**Request atomicity.** `viewer.set_overlay` parses and validates **every** supplied field
first — including resolving `volumeId` to a `Volume` and checking `colormap` /
`composite.method` against their enums — before applying **any** of them. A malformed or
unknown field anywhere in the request rejects the whole call and mutates nothing; the
overlay is never left half-updated by a request that fails partway through.

**Colormap validation.** Unlike the renderer (`vc::resolve()`, which silently falls back to
`"fire"` on an unrecognized id), `viewer.set_overlay` validates `colormap` against the same
table the overlay panel's combo box is populated from (`vc::specs()`,
core/src/render/Colormaps.cpp) and rejects an unknown id with `-32602` rather than silently
accepting a typo. The empty string `""` is always valid (the "no explicit choice yet"
sentinel `VolumeOverlayController::resetToDefaults()` uses). Valid non-empty ids:
`"fire"`, `"viridis"`, `"magma"`, `"red"`, `"green"`, `"blue"`, `"cyan"`, `"magenta"`,
`"glasbey_black0"`.

**Clear semantics.** `clear: true` and an explicit empty `"volumeId": ""` are equivalent —
both null the overlay volume (`ViewerManager::setOverlayVolume(nullptr, "")`). Omitting
`volumeId` entirely (and not passing `clear`) leaves the current overlay volume untouched.

**`threshold` is an alias, not an independent field.** `ViewerManager::setOverlayThreshold`
is implemented as `setOverlayWindow(max(threshold, 0), currentWindowHigh)` — i.e. setting
`threshold` actually moves `windowLow` (clamped `>= 0`) while leaving `windowHigh`
untouched, going through the same `high > low` gap enforcement as `window`. The echoed
`threshold` in the result always equals the resulting `windowLow`. Setting `threshold` and
`window` in the same request is not additive — whichever is applied last in handler order
(`window` after `threshold`) wins for `windowLow`.

**Coordinate-space re-validation.** `ViewerManager::setOverlayVolume` independently
re-validates the requested volume's coordinate-space tag against the base (current) volume
and **silently nulls** the selection on a mismatch (logged, not surfaced as an RPC error).
`viewer.set_overlay`'s `-32007` only catches an unresolvable `volumeId`; a resolvable but
coordinate-space-incompatible one is accepted by the RPC and then quietly dropped by
`ViewerManager`. Always check the echoed `volumeId` in the result to confirm a set actually
stuck — `viewer.list_overlay_volumes` (25.3) intentionally does not pre-filter by
compatibility, so an agent can pick a volume, attempt the set, and diagnose the rejection
from the echo.

### 25.3 `viewer.list_overlay_volumes`

- **params:** none.
- **result:** `{"volumes": [{"id": str, "current": bool}], "overlayVolumeId": str}` —
  every `VolumePkg` volume id, `current` marking the base (current) volume; `overlayVolumeId`
  is `""` when no overlay is set. Not filtered by coordinate-space compatibility (25.2).
- **errors:** `-32010` (viewer manager unavailable), `-32000` (no volume package loaded).

### 25.4 `viewer.set_intersects`

Set which surfaces' intersection lines a viewer draws — the RPC twin of
`SurfacePanelController`'s per-viewer `setIntersects()` calls.

- **params:**
  ```json
  {"surfaceIds": [str],       // required; surface/slot ids to draw intersections for
   "viewer"?: str}            // viewer id or surface-slot name (§2.2); omit/null => broadcast
  ```
- **result:** `{"surfaceIds": [str], "appliedToViewers": [str]}` — `surfaceIds` is the
  resulting set (see the union note below); `appliedToViewers` lists the registry viewer
  id(s) (§2.2) the set was applied to.
- **errors:**
  - `-32010` (viewer manager unavailable).
  - `-32602` (`data.param:"surfaceIds"`) — `surfaceIds` is missing, not an array, or
    contains a non-string.
  - `-32002` — `viewer` does not resolve to exactly one live viewer (§2.2 targeting rules).
  - `-32009` (`data.detail:"the segmentation viewer does not draw intersections against
    itself"`) — `viewer` resolves to the `"segmentation"` surface slot.

**Always unions `"segmentation"`.** The applied set always includes `"segmentation"` even
if the caller's `surfaceIds` omits it — mirroring `SurfacePanelController::applyFilters`,
which always seeds the drawn set with `"segmentation"`. The echoed `surfaceIds` reflects
this union, not the raw request array.

**Broadcast when `viewer` is omitted.** With no `viewer` (or an explicit JSON `null`), the
call applies to **every** base viewer except the one bound to the `"segmentation"` surface
slot (the GUI's own no-filter default). `appliedToViewers` then lists every viewer it
touched.

**`-32009` when the target resolves to the segmentation viewer.** Explicitly targeting a
viewer whose `surfName()` is `"segmentation"` is rejected outright (that viewer never draws
intersections against itself) rather than silently no-op'd.

### 25.5 MCP tool surface

| MCP tool | → RPC | Description shown to the agent |
|---|---|---|
| `vc3d_get_overlay` | `viewer.get_overlay` | Read the current overlay-volume settings. |
| `vc3d_set_overlay` | `viewer.set_overlay` | Update the overlay-volume settings (volume, colormap, opacity, threshold, window, resolution cap, composite); any subset. |
| `vc3d_list_overlay_volumes` | `viewer.list_overlay_volumes` | List every volume id in the open package, for picking an overlay volume. |
| `vc3d_set_intersects` | `viewer.set_intersects` | Set which surfaces' intersection lines a viewer draws. |

Same conventions as §5 (snake_case params 1:1 with 25.1–25.4, params/result passthrough, RPC
errors preserved).

---

## 26. Point-collection editing (`points.*`)

Full CRUD + attribute editing over the live `VCCollection` point-annotation store
(`CState::pointCollection()`), extending the read/create-only `points.commit` /
`points.list` pair (§3.13–3.14). Every handler in this family first resolves the live
store (`-32000` if no volume package is loaded; `-32010` if the store itself is
unexpectedly null) and, where a collection is named, resolves its id (see below).

**Identifier convention.** Every collection-scoped RPC accepts **either** `"collection"`
(name, string) **or** `"collectionId"` (numeric id) to identify the target collection.
When both are present, `collectionId` wins. An unresolvable name or id →
`-32007` with `data.kind:"collection"`, `data.id` (the offending name or, for an unknown
id, its string form). A point-scoped RPC identifies its target with a single required
`"pointId"` (numeric); an unknown point id → `-32007` with `data.kind:"point"`,
`data.id` (the numeric id, as a JSON number).

**Safe-integer id limit.** Point and collection ids travel as JSON numbers (IEEE-754
doubles), which round-trip integers exactly only up to `2^53 - 1`
(`9007199254740991`, JSON's safe-integer bound). Any `pointId` / `collectionId` param is
validated as a positive integer `<= 2^53 - 1`; a non-integer, non-positive, or
too-large value → `-32602` rather than silently corrupting via a double-to-`uint64_t`
round-trip.

**Winding null↔NaN convention.** `winding_annotation` is a C++ `float`; on the wire,
"unset" is JSON `null` and any other value is a finite number. Reading a point
(`points.list`, and the results below) reports `null` when the annotation is NaN.
Writing a point (`points.update_point`) accepts an explicit JSON `null` to **clear** the
annotation (sets it back to NaN); omitting `"winding"` entirely leaves it untouched.
A non-finite-as-float number (e.g. `1e300`, which would silently narrow to `+/-inf` and
be indistinguishable from "unset") is rejected with `-32602` rather than accepted.

**Winding-fill `mode` enum.** `"none" | "incremental" | "decremental" | "constant"`
(`PointCollections::WindingFillMode`); an unrecognized string → `-32602`
(`data.param:"mode"`, `data.value`). `"constant"` is the only mode that reads the optional
`"constant"?: float` parameter (default `0.0` when omitted).

### 26.1 Collection lifecycle

**`points.add_collection`**
- **params:** `{"name"?: str}` — omit for an auto-generated unique name (mirrors the GUI's
  "New collection" button, `VCCollection::generateNewCollectionName()`).
- **result:** `{"collectionId": int, "name": str}`.
- **errors:** `-32000`, `-32010`.

**`points.rename_collection`**
- **params:** `{"collection"?: str, "collectionId"?: int, "newName": str}`.
- **result:** `{"collectionId": int, "name": str}`.
- **errors:** `-32000`, `-32010`, `-32007` (unknown source collection),
  `-32602` (`data.param:"newName"`) — `newName` is empty (an empty name would make the
  collection unreachable by name via the `collection` identifier).

**`points.clear_collection`**
- **params:** `{"collection"?: str, "collectionId"?: int}`.
- **result:** `{"cleared": true}`.
- **errors:** `-32000`, `-32010`, `-32007` (unknown collection).

**`points.clear_all`**
- **params:** none.
- **result:** `{"cleared": true}`.
- **errors:** `-32000`, `-32010`.

### 26.2 Point mutation

**`points.update_point`**
- **params:**
  ```json
  {"pointId": int,             // required
   "position"?: Vec3,          // volume space; omit to leave unchanged
   "winding"?: float | null}   // null clears; omit leaves unchanged
  ```
- **result:** `{"id": int, "position": Vec3, "winding": float | null}` — the point's
  state after applying the change.
- **errors:** `-32000`, `-32010`, `-32007` (`data.kind:"point"` — unknown `pointId`),
  `-32602` (missing/non-finite `pointId`; non-finite `position`; out-of-float-range
  `winding`).

**`points.remove_point`**
- **params:** `{"pointId": int}`.
- **result:** `{"removed": true}`.
- **errors:** `-32000`, `-32010`, `-32007` (`data.kind:"point"` — unknown `pointId`).

### 26.3 Collection attributes

**`points.set_collection_color`**
- **params:** `{"collection"?: str, "collectionId"?: int, "color": [float, float, float]}`
  — an `[r, g, b]` array (not an `{x,y,z}` object).
- **result:** `{"collectionId": int, "color": [float, float, float]}`.
- **errors:** `-32000`, `-32010`, `-32007`, `-32602` (`color` not a 3-element finite
  array).

**`points.set_collection_metadata`**
- **params:** `{"collection"?: str, "collectionId"?: int, "absoluteWindingNumber": bool}`.
- **result:** `{"collectionId": int, "absoluteWindingNumber": bool}`.
- **errors:** `-32000`, `-32010`, `-32007`, `-32602` (missing/non-bool
  `absoluteWindingNumber`).

**`points.set_collection_tag`**
- **params:** `{"collection"?: str, "collectionId"?: int, "key": str, "value": str}`.
- **result:** `{"ok": true}`.
- **errors:** `-32000`, `-32010`, `-32007`, `-32602` (missing `key`/`value`).

**`points.remove_collection_tag`**
- **params:** `{"collection"?: str, "collectionId"?: int, "key": str}`.
- **result:** `{"ok": true}`.
- **errors:** `-32000`, `-32010`, `-32007`, `-32602` (missing `key`).

**`points.set_windings_linked`**
- **params:** `{"collection"?: str, "collectionId"?: int, "linkedCollectionIds": [int]}`
  — ids of other collections whose winding numbers stay linked to this one.
- **result:** `{"collectionId": int, "linkedCollectionIds": [int]}` — echoes the array as
  applied (ids are **not** individually existence-checked against `getAllCollections()`).
- **errors:** `-32000`, `-32010`, `-32007` (unknown target collection),
  `-32602` (`linkedCollectionIds` not an array of safe-integer ids).

### 26.4 Winding fills

**`points.auto_fill_windings`**
- **params:** `{"collection"?: str, "collectionId"?: int, "mode": str, "constant"?: float}`
  — applies the fill immediately across the collection's existing points.
- **result:** `{"ok": true}`.
- **errors:** `-32000`, `-32010`, `-32007`, `-32602` (bad `mode`; out-of-float-range
  `constant`).

**`points.set_auto_fill_mode`**
- **params:** same shape as `auto_fill_windings`, but only records the mode/constant for
  **future** points added to the collection — it does not touch existing points.
- **result:** `{"ok": true}`.
- **errors:** same as `auto_fill_windings`.

**`points.reset_windings`**
- **params:** none. Resets winding numbers **across all collections**.
- **result:** `{"ok": true}`.
- **errors:** `-32000`, `-32010`.

**`points.apply_anchor_offset`**
- **params:** `{"offsetX": float, "offsetY": float}` — applies a 2D grid offset to every
  collection's anchor (used when remapping annotations after a surface-growth shift).
- **result:** `{"ok": true}`.
- **errors:** `-32000`, `-32010`, `-32602` (non-finite offset).

### 26.5 Persistence (whole-collection JSON + per-segment correction paths)

**`points.save_json` / `points.load_json`**
- **params:** `{"path": str}` — absolute or relative file path.
- **result:** `{"saved": bool}` / `{"loaded": bool}` — the underlying
  `VCCollection::saveToJSON` / `loadFromJSON` success flag; a write/parse failure is
  reported **in-band** (`false`), not as an RPC error.
- **errors:** `-32000`, `-32010`, `-32602` (missing/empty `path`).

**`points.save_segment_path` / `points.load_segment_path`**
- **params:** `{"segmentPath": str}` — a segment directory (2D-anchored correction points
  live alongside the segment's own data, not in a standalone JSON file).
- **result:** `{"saved": bool}` / `{"loaded": bool}` — same in-band-failure convention as
  `save_json`/`load_json`.
- **errors:** `-32000`, `-32010`, `-32602` (missing/empty `segmentPath`).

### 26.6 MCP tool surface

| MCP tool | → RPC |
|---|---|
| `vc3d_add_point_collection` | `points.add_collection` |
| `vc3d_rename_point_collection` | `points.rename_collection` |
| `vc3d_clear_point_collection` | `points.clear_collection` |
| `vc3d_clear_all_points` | `points.clear_all` |
| `vc3d_update_point` | `points.update_point` |
| `vc3d_remove_point` | `points.remove_point` |
| `vc3d_set_point_collection_color` | `points.set_collection_color` |
| `vc3d_set_point_collection_metadata` | `points.set_collection_metadata` |
| `vc3d_set_point_collection_tag` | `points.set_collection_tag` |
| `vc3d_remove_point_collection_tag` | `points.remove_collection_tag` |
| `vc3d_set_point_windings_linked` | `points.set_windings_linked` |
| `vc3d_auto_fill_windings` | `points.auto_fill_windings` |
| `vc3d_set_auto_fill_mode` | `points.set_auto_fill_mode` |
| `vc3d_reset_windings` | `points.reset_windings` |
| `vc3d_apply_anchor_offset` | `points.apply_anchor_offset` |
| `vc3d_save_points_json` / `vc3d_load_points_json` | `points.save_json` / `points.load_json` |
| `vc3d_save_points_segment_path` / `vc3d_load_points_segment_path` | `points.save_segment_path` / `points.load_segment_path` |

Same conventions as §5 (params/result passthrough, RPC errors preserved). Param names are
1:1 with the RPC (this family does not snake_case-rename `collectionId`/`pointId`/etc.).

---

## 27. Review-state segment listing (`segments.review`)

Programmatic twin of the surface panel's review-tag filter checkboxes
(`SurfacePanelController.cpp` ~2727–2782): lists segments together with their
`approved`/`defective`/`reviewed`/`inspect`/`partial_review` tag state and a summarized
`reviewState`, with the same optional filtering the panel's checkboxes apply.

- **params:**
  ```json
  {"onlyLoaded"?: bool = false,   // restrict to currently-loaded surfaces
   "filter"?: {                  // all optional bools; ANDed together
     "unreviewed"?: bool,        // keep only segments WITHOUT "reviewed"
     "approved"?: bool,          // keep only segments WITH "approved"
     "defective"?: bool,         // keep only segments WITH "defective"
     "hideDefective"?: bool,     // keep only segments WITHOUT "defective"
     "reviewed"?: bool,          // keep only segments WITH "reviewed"
     "inspect"?: bool,           // keep only segments WITH "inspect"
     "partialReview"?: bool}}    // keep only segments WITH "partial_review"
  ```
  A `filter` key that is absent or `false` contributes nothing — same as an unchecked
  checkbox — rather than inverting the predicate.
- **result:**
  ```json
  {"segments": [
    {"id": str, "path": str, "loaded": bool, "active": bool,
     "tags": {"approved": bool, "defective": bool, "reviewed": bool,
              "inspect": bool, "partial_review": bool},
     "reviewState": str}],       // one-line precedence summary, see below
   "total": int,                 // count of the onlyLoaded-scoped candidate set
   "returned": int}              // count after `filter` is additionally applied
  ```
- **errors:** `-32000` (no volume package loaded).

**`reviewState` precedence.** `"defective"` (wins even over `"approved"` — a segment
flagged defective needs attention regardless of a stale approval) > `"approved"` >
`"reviewed"` > `"partial_review"` > `"inspect"` > `"unreviewed"` (none of the tags set).
This mirrors the filter's `hasPartialReview = partial_review || reviewed` grouping while
still distinguishing the two states individually.

**Meta source (freshness).** For each segment, tags are read from, in order: (1) a live
`QuadSurface` — `CState::surface(id)` (attached to a viewer) or, failing that,
`VolumePkg::getSurface(id)` (loaded earlier this session) — because tag-checkbox edits in
the GUI mutate that object's `meta` in place and flush to disk immediately; this is the
freshest possible source for a segment touched this session. (2) Failing that, a **fresh
read of `meta.json` straight off disk** — deliberately **not**
`Segmentation::metadata()`, which is parsed once at project-open time and never updated
afterwards, so it would go stale the moment a segment is loaded, tag-edited, and unloaded
again within the same session. The `meta.json` re-read is a small-file parse (not a
TIFF/point-data load), so this stays cheap enough that `onlyLoaded:false` remains a safe
default. If the file is transiently unreadable, the handler falls back to the stale
cached `Segmentation::metadata()` rather than dropping the segment from the listing.

**`loaded` matches `segments.list`'s convention.** Both endpoints compute `"loaded"` the
same way — set membership against `CState::surfaceNames()` — so they agree on what
"loaded" means for a given segment id.

### 27.1 MCP tool surface

| MCP tool | → RPC | Description shown to the agent |
|---|---|---|
| `vc3d_review_segments` | `segments.review` | List segments with review-tag state and optional server-side filtering (the programmatic equivalent of the surface panel's review filter checkboxes). |

Snake_case params 1:1 with the RPC above (`only_loaded`; `filter`'s nested keys are passed
through unrenamed).

---

## 28. Per-segment mesh operations (`segment.*`)

Six operations extending the per-segment mesh-editing surface (§23's segment lifecycle,
Workstream 4): two synchronous dialog-free ops (`crop_bounds`, `recalc_area`), two
asynchronous external-tool ops sharing the `source:"tool"` job slot (§8.3) like
`render.tifxyz` (`reoptimize`, `refine_alpha_comp`), and two asynchronous in-process ops
resolved via the §8.4 deferred-response mechanism (`generate_mask`, `append_mask`).

### 28.1 `segment.crop_bounds` (synchronous)

Headless twin of the "Crop to valid region" context-menu action
(`SegmentationCommandHandler::cropSurfaceToValidRegion`): crops the surface grid to its
tightest valid bounds, writes it in place, and refreshes metrics — synchronously, before
the RPC returns.

- **params:** `{"segmentId": str}`.
- **result:** `{"cropped": true, "segmentId": str}` — the op does not distinguish
  "cropped" from "already at tightest bounds"; both report `cropped: true`.
- **errors:** `-32000` (no vpkg), `-32001` (no current volume),
  `-32007` (`data.kind:"segment"` — unknown segment),
  `-32004` (`data.detail:"a mask render is in progress"`) — **rejected** while a
  `generate_mask`/`append_mask` render is running, because crop rewrites the same
  in-memory `QuadSurface` (points/channels/meta) a background mask render is mutating,
  `-32009` (`data.detail`) — segmentation command handler unavailable,
  `-32005` (`data.detail`) — internal crop failure (missing coordinate grid,
  channel-size mismatch, save error).

### 28.2 `segment.recalc_area` (synchronous)

Pure computation, no UI — the headless twin of the surface panel's area recalculation.

- **params:** `{"segmentIds": [str]}` — non-empty array.
- **result:**
  ```json
  {"results": [
    {"segmentId": str, "areaVx2": float, "areaCm2": float,
     "success": bool, "errorReason": str | null}]}
  ```
  A bad/unknown id is reported **in-band** (`success:false`, `errorReason` set) rather
  than failing the whole call.
- **errors:** `-32000` (no vpkg), `-32001` (no current volume),
  `-32602` (`segmentIds` missing/not-an-array/empty/non-string-element).

### 28.3 `segment.reoptimize` (asynchronous, `source:"tool"`)

Resume-opt local reoptimization (`vc_grow_seg_from_segments` via
`SegmentationCommandHandler::startResumeLocalGrowPatch`) — the headless twin of the
"Resume-opt Local (GrowPatch)" context-menu action. No dialog is ever shown.

- **params:**
  ```json
  {"segmentId": str,               // required
   "volumeId"?: str,                // default: current volume
   "ompThreads"?: int = 1,          // >= 0 (0 => runner default)
   "paramOverrides"?: {...}}        // merged over the fixed resume-local tracer params
  ```
- **result:** `{"jobId": str, "kind": "segment.reoptimize", "source": "tool",
  "outputDir": str, "volumeId": str}`. Completion is a `job.progress`
  (`phase:"finished"`, `source:"tool"`) notification, observable via `job.status`.
- **errors:** `-32000`, `-32001`, `-32007` (unknown `segmentId` or `volumeId`),
  `-32602` (`ompThreads < 0`; `paramOverrides` not an object),
  `-32004` (`data.source:"tool"`) — a `"tool"`-source job is already running/active,
  `-32006` — `vc_grow_seg_from_segments` not found or not executable,
  `-32009` — segmentation command handler unavailable,
  `-32005` (`data.detail`) — generic launch failure.

### 28.4 `segment.refine_alpha_comp` (asynchronous, `source:"tool"`)

Alpha-composite refinement (`vc_objrefine` via
`SegmentationCommandHandler::startAlphaCompRefine`) — the headless twin of the
"Alpha-comp refine" context-menu action. No dialog is ever shown. **Rejects remote
volumes** (alpha-comp refine only supports local volumes).

- **params:** (all optional; defaults mirror `AlphaCompRefineDialog`'s session defaults)
  ```json
  {"segmentId": str,                // required
   "refine"?: bool = true,
   "start"?: float = -6.0, "stop"?: float = 30.0, "step"?: float = 2.0,
   "low"?: int = 26, "high"?: int = 255,
   "borderOff"?: float = 1.0, "radius"?: int = 3,
   "genVertexColor"?: bool = false, "overwrite"?: bool = true,
   "readerScale"?: float = 0.5, "scaleGroup"?: str = "1",
   "ompThreads"?: int,               // omit for the runner default
   "outputDir"?: str}                // default: <segment>_refined
  ```
- **result:** `{"jobId": str, "kind": "segment.refine_alpha_comp", "source": "tool",
  "outputDir": str, "segmentId": str}`. Completion is a `job.progress`
  (`phase:"finished"`, `source:"tool"`) notification, observable via `job.status`.
- **errors:** `-32000`, `-32001`, `-32007` (unknown `segmentId`),
  `-32009` (`data.detail`) — **remote volume rejected**, or segmentation command handler
  unavailable,
  `-32004` (`data.source:"tool"`) — a `"tool"`-source job is already running,
  `-32006` — `vc_objrefine` not found or not executable,
  `-32005` (`data.detail`) — generic launch failure.

### 28.5 `segment.generate_mask` / `segment.append_mask` (asynchronous, deferred, §8.4)

Render a segment's binary mask (`generate_mask`) or append a volume-image layer to an
existing/new mask (`append_mask`) — headless twins of the "Edit mask" / "Append mask"
context-menu actions. The underlying render runs on a `QtConcurrent` worker with no
bridge-visible completion signal, so both RPCs resolve via the **deferred-response**
mechanism (§8.4) rather than a job: the bridge holds the reply open and writes it from the
worker's finished callback. **Timeout: 120 s** (`beginDeferred(120000, ...)`); the MCP
tools set a **130 s client-side timeout** to stay safely past the server cap.

- **params:** `{"segmentId": str}` (both).
- **result:**
  - `generate_mask`, mask already exists (`mask.tif` present — **not** regenerated,
    matching the GUI's "Edit mask" behavior): synchronously
    `{"generated": false, "alreadyExists": true, "maskPath": str, "segmentId": str}`.
  - `generate_mask`, fresh render / `append_mask` (always renders): on completion
    `{"generated": true, "appended": bool, "maskPath": str, "segmentId": str,
    "message": str}`.
- **errors:**
  - `-32000` — no volume package loaded.
  - `-32007` (`data.kind:"segment"`) — unknown segment.
  - `-32001` — no current volume (`append_mask` **only**: appending renders a
    volume-image layer, so a current volume is required; `generate_mask`'s binary-mask
    path does not need one).
  - `-32004` (`data.detail:"a mask render is already in progress"`) — another
    `generate_mask`/`append_mask` render is already running (at most one at a time,
    guarded by `_maskRenderInProgress`; this is the same flag `segment.crop_bounds`
    checks in the other direction).
  - `-32005` (`data.detail`) — the render failed (reported through the worker's
    completion callback, or a synchronous launch/setup failure caught and converted to
    the same deferred error).

### 28.6 Deliberately not implemented

`SegmentationCommandHandler::onExportWidthChunks` (the "Export width-chunks (40k px)"
context-menu action) has **no** bridge RPC. It was intentionally left off this surface;
there is no `segment.export_width_chunks` (or similar) method.

### 28.7 MCP tool surface

| MCP tool | → RPC | Description shown to the agent |
|---|---|---|
| `vc3d_crop_segment_bounds` | `segment.crop_bounds` | Crop a segment's surface grid to its tightest valid bounds. Synchronous. |
| `vc3d_recalc_segment_area` | `segment.recalc_area` | Recompute surface area for one or more segments. Synchronous. |
| `vc3d_reoptimize_segment` | `segment.reoptimize` | Resume-opt local reoptimization of a segment. Asynchronous (`source:"tool"`); supports `wait: bool = false` (30-minute cap). |
| `vc3d_refine_segment_alpha_comp` | `segment.refine_alpha_comp` | Alpha-composite refinement of a segment. Asynchronous (`source:"tool"`); rejects remote volumes; supports `wait: bool = false` (30-minute cap). |
| `vc3d_generate_segment_mask` | `segment.generate_mask` | Render a segment's binary mask. Blocks until the render completes (130 s client timeout); no job to poll. |
| `vc3d_append_segment_mask` | `segment.append_mask` | Append a volume-image layer to a segment's mask. Blocks until the render completes (130 s client timeout); requires a current volume. |

Same conventions as §5 (snake_case params 1:1, `wait` is an MCP-server-side convenience
folding the terminal `job.status` inline per §5/§19.3, RPC errors preserved).
