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
  If `QLocalServer::listen(name)` fails (stale socket), call
  `QLocalServer::removeServer(name)` once and retry; if it still fails, print an error to
  stderr and exit with code 2 (a test that asked for a bridge must not silently run without
  one).

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
- **result:** `{"pong": true, "pid": int, "version": str}` — `version` from
  `QApplication::applicationVersion()`.
- **errors:** none.

### 3.2 `state.get`

Snapshot of app state. Never errors on missing volume — reports what exists.

- **params:** none.
- **result:**
  ```json
  {
    "vpkg":   null | {"path": str},
    "volume": null | {"id": str, "path": str, "voxelSize": float},
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
- **errors:** `-32002` (bad viewer target), `-32005` (file write failed, reuse with
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

- **params:** `{"viewer"?: str, "factor": float}` — `factor` > 0; >1 zooms in.
- Maps to `VolumeViewerBase::adjustZoomByFactor(float)` (VolumeViewerBase.hpp:85).
- **result:** `{"scale": float}` — post-zoom `getCurrentScale()`.
- **errors:** `-32002`, `-32602` (factor ≤ 0 or non-finite).

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
  {"method"?: str = "tracer",     // "tracer"|"corrections"|"patch_tracer"|"manual_add"
   "direction"?: str = "all",     // "all"|"up"|"down"|"left"|"right"|"fill"
   "steps": int,                  // >= 1
   "inpaintOnly"?: bool = false}
  ```
  Enum mapping (segmentation/growth/SegmentationGrowth.hpp:23–66):
  `tracer=Tracer(0)`, `corrections=Corrections(1)`, `patch_tracer=PatchTracer(3)`,
  `manual_add=ManualAdd(4)`; directions map in declared order `All..Fill`.
- **result:** `{"jobId": str, "kind": "segmentation.grow"}` — completion is signaled by a
  `job.progress` notification with `phase:"finished"` when
  `onSegmentationGrowthStatusChanged(false)` fires.
- **errors:** `-32000`, `-32001`, `-32004`, `-32007` (no active segmentation surface),
  `-32008` (editing not enabled), `-32602` (bad enum string / steps < 1).

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
| `vc3d_ping` | `ping` | Check the VC3D bridge is alive; returns pid and app version. |
| `vc3d_get_state` | `state.get` | Snapshot of VC3D: open volume package, current volume, active segment, viewers (ids/names), editing mode, running job. Call this first. |
| `vc3d_list_segments` | `segments.list` | List segments in the open volume package with loaded/active flags. |
| `vc3d_screenshot` | `screenshot.capture` | Capture a PNG of the whole VC3D window or one viewer pane. Returns base64 or writes to filePath. |
| `vc3d_get_cursor_point` | `canvas.get_cursor_volume_point` | Resolve a viewer scene position (or the current cursor) to a 3D volume point + surface normal. |
| `vc3d_click` | `canvas.click` | Synthesize a mouse click in a viewer at a volume-space (or scene-space) position, with button and modifiers (e.g. `{"modifiers": ["shift"]}` to place a point / set focus). |
| `vc3d_shift_click` | `canvas.shift_click` | Shift+click convenience: the canonical place-point / set-focus gesture. |
| `vc3d_center_viewer` | `viewer.center_on_point` | Center a viewer pane on a 3D volume point. |
| `vc3d_zoom_viewer` | `viewer.zoom` | Multiply a viewer's zoom by a factor (>1 zooms in). Returns the new scale. |
| `vc3d_enable_editing` | `segmentation.enable_editing` | Turn segmentation editing mode on/off for the active segment. |
| `vc3d_grow_segment` | `segmentation.grow` | Grow the active segmentation surface (method: tracer/corrections/patch_tracer/manual_add; direction; steps). Async: returns a jobId. |
| `vc3d_grow_patch_from_seed` | `segmentation.grow_patch_from_seed` | Create a brand-new segment by growing a patch from a 3D seed point (headless GrowPatch). Async: returns a jobId and outputDir. |
| `vc3d_commit_points` | `points.commit` | Add annotation points (volume space) to a named collection, optionally with a winding annotation. |
| `vc3d_list_points` | `points.list` | List point collections and their points. |
| `vc3d_open_volume` | `volume.open` | Open a volume package (.volpkg / .volpkg.json / zarr project) and optionally select a volume id. |
| `vc3d_open_catalog_sample` | `catalog.open_sample` | Open an Open Data catalog sample by its manifest sample id. |
| `vc3d_job_status` | `job.status` | Poll a job by id (or the latest job): state, message, console tail. |

Parameter schemas: each tool's `inputSchema` is the JSON-Schema rendering of the
corresponding RPC params block in §3, with the same names, types, defaults, and enums
(e.g. `vc3d_grow_segment.method` is
`{"type":"string","enum":["tracer","corrections","patch_tracer","manual_add"],"default":"tracer"}`).
The MCP server performs no semantic validation beyond schema — the bridge is the
authority.

`job.progress` notifications: the MCP server listens for them and (a) folds them into
`vc3d_job_status` responses (console tail, state) and (b) where the MCP client supports
progress notifications, forwards `phase:"output"` messages as tool progress for the
long-running `vc3d_grow_*` calls when the caller opted into `"wait": true` — an
MCP-server-side convenience param (not part of the RPC) that blocks the tool call until
the job's `finished` notification and returns the terminal status inline.
`wait` defaults to `false`; when `true`, the MCP server enforces a 30-minute cap and
returns the still-running `jobId` on timeout.

---

## 6. Implementation placement (for later phases)

- `apps/VC3D/agent_bridge/AgentBridge.{hpp,cpp}` — QLocalServer, framing, dispatch,
  viewer registry, job tracker. Constructed in `VCAppMain.cpp` after `CWindow` (§1.1);
  `friend class AgentBridge;` added in `CWindow.hpp` beside `RenderBenchReplay`.
- `SegmentationCommandHandler`: refactor per §4 (this is the only behavioral change to
  existing files beyond friend/wrapper additions).
- `MenuActionController`: add public `openOpenDataSampleById(const QString&)` (§3.16).
- MCP server: separate process under `apps/VC3D/agent_bridge/mcp/` (Phase 3), stdio MCP
  transport, QLocalSocket (or platform-equivalent) client to the bridge.
