# Segmentation Editing Architecture

## Purpose and Scope

The Segmentation Editor lets annotators reshape a generated `QuadSurface`
patch interactively. It layers a Qt-based interaction module over the core
surface utilities so that users can sample the existing grid into draggable
handles, tweak local geometry, fill missing areas, and persist the result back
to disk. This document captures the current implementation in detail (late
2024 state) so new contributors can confidently extend or debug the tool.

The primary runtime is the VC3D application; references below use source paths
relative to `apps/VC3D` unless otherwise noted.


## High-Level Data Flow

1. **Session start** – `SegmentationModule::beginEditingSession` receives the
   active `QuadSurface` from the segmentation tool. The module forwards the
   surface into `SegmentationEditManager::beginSession`, which clones the raw
   points into a preview surface (`SegmentationEditManager.cpp:127-149`).
2. **Handle generation** – The edit manager down-samples the grid and materialises
   a handle for each valid cell along that stride while persisting previously
   added manual handles (`SegmentationEditManager.cpp:728-768`).
3. **User interaction loop** – The module mediates input from each
   `CVolumeViewer`, calls into the edit manager for coordinate updates, and pushes
   preview changes back to the shared `CSurfaceCollection` for live rendering
   (`SegmentationModule.cpp:524-807`).
4. **Overlay feedback** – `SegmentationOverlayController` queries the edit
   manager for handle positions and renders them as a viewer overlay with hover
   and selection highlights (`overlays/SegmentationOverlayController.cpp:33-125`).
5. **Preview persistence** – On “Apply”, the preview grid overwrites the base
   surface and calls `QuadSurface::saveOverwrite` to commit to disk
   (`SegmentationModule.cpp:216-236`). “Reset” re-syncs the preview from the
   original clone (`SegmentationEditManager.cpp:192-205`).


## Component Overview

| Component | Responsibility |
| --- | --- |
| `SegmentationWidget` (`SegmentationWidget.cpp`) | Qt sidebar with controls for enabling editing, adjusting down-sample/radius/sigma, configuring hole filling behaviour, and triggering Apply/Reset.
| `SegmentationModule` (`SegmentationModule.cpp/.hpp`) | Orchestrates the session lifecycle, binds viewers, processes input, and drives the overlay/state machines.
| `SegmentationEditManager` (`SegmentationEditManager.cpp/.hpp`) | Core data model – clones surfaces, manages handle lists, applies deformations, fills holes, and tracks dirty state.
| `SegmentationOverlayController` (`overlays/SegmentationOverlayController.cpp`) | Draws handle markers inside slice viewers and provides transient radius indicators.
| `QuadSurface` (core layer, `core/include/vc/core/util/Surface.hpp`) | Stores the dense `cv::Mat_<cv::Vec3f>` grid that represents the segmentation surface in world space.

### Session Lifecycle

1. **Begin** – `beginEditingSession` configures down-sample/radius/sigma, swaps the
   overlay to editing mode, and flags pending changes (`SegmentationModule.cpp:404-426`).
2. **End** – `endEditingSession` discards the preview surface, restores the base
   surface to the viewer, clears overlays, resets input state, and exits point-add
   mode (`SegmentationModule.cpp:431-445`).
3. **Apply/Reset** – Both operations go through `SegmentationEditManager`, either
   copying preview → base (Apply) or restoring preview ← baseline clone (Reset)
   (`SegmentationModule.cpp:216-242`, `205-214`).


## Handle Model and Sampling

### Automatic vs Manual Handles

- **Automatic handles** represent a regular lattice of control points derived by
  stepping through the preview grid with the current down-sample value. These
  are recreated whenever down-sampling changes or the preview is reset
  (`SegmentationEditManager.cpp:728-768`).
- **Manual handles** are the points the user adds or modifies. They are stored in
  the `_handles` vector with `Handle::isManual = true`, survive re-sampling, and
  can be removed with Ctrl-click or the Delete key (`SegmentationEditManager.hpp:38-47`,
  `SegmentationModule.cpp:540-555`).

All handles cache both their original (`originalWorld`) and current (`currentWorld`)
positions so deltas can be re-applied whenever radius or sigma settings change
(`SegmentationEditManager.cpp:206-234`, `884-936`).

### Down-sample Control

The widget exposes a spin-box (default 12) that determines the grid stride
(`SegmentationWidget.cpp:43-114`). Changing it recomputes the full handle set and
updates the overlay, while the active preview surface remains untouched except
for the regenerated handles (`SegmentationModule.cpp:167-192`).

Slice viewers now mirror the VCCollection behaviour: a dedicated “Slice
Visibility” section lets you tune how far from the plane handles remain visible
and whether they fade out or disappear altogether (`SegmentationWidget.cpp:132-226`,
`SegmentationOverlayController.cpp`).

Handle hover highlighting also mirrors VCCollection. A configurable
“Highlight distance” limits which handle the cursor can latch onto, keeping the
UI from constantly snapping to faraway points (`SegmentationWidget.cpp`,
`SegmentationModule.cpp:750`).


## Interaction Model

### Mouse

- **Left-click (default mode)** – Ensures a grid cell handle exists under the
  cursor (creating or reusing one on demand) and immediately begins the drag
  reposition (`SegmentationModule.cpp:880-1065`,
  `SegmentationEditManager.cpp:348-458`).
- **Left-click (point-add mode)** – When Shift is held (or toggled), the module
  calls `addHandleAtWorld` with `allowCreate=true`, enabling new handle creation
  inside gaps (`SegmentationModule.cpp:563-607`). Point-add mode forces the
  “Add” cursor (`SegmentationModule.cpp:283-325`, `618-640`, `700-739`).
- **Ctrl + Left-click** – Deletes the closest manual handle if one falls within
  the radius tolerance (`SegmentationModule.cpp:540-555`).
- **Left Drag** – Updates the handle’s `currentWorld` position continuously and
  re-runs the influence falloff to update nearby grid cells (`SegmentationModule.cpp:556-607`).
- **Mouse Wheel (with focus in viewer)** – Adjusts the brush radius; Ctrl doubles
  the step. The module updates the overlay, the preview surface, and shows an onscreen
  label (`SegmentationModule.cpp:716-779`).

### Keyboard Shortcuts

- **Shift (hold/tap)** – Toggles point-add mode (`SegmentationModule.cpp:327-336`).
- **R** – Focuses the current handle in viewer space by setting a POI on the surface
  and aligning the overlay selection (`SegmentationModule.cpp:337-369`).
- **Delete / Backspace** – Removes the active/hovered manual handle (`SegmentationModule.cpp:370-409`).
- Global segmentation shortcuts such as evenly-spacing points (Y/Z/V) are exposed
  through the keybinding dialog but handled elsewhere (`MenuActionController.cpp:395-449`).


## Adding and Filling Points

### Barycentric Projection

When a user adds a handle (normal or point-add mode), the edit manager finds the
closest valid triangle in the neighbourhood, computes the barycentric coordinates
of the click, and projects the point onto that face. This works for both interior
and extrapolated hits (`SegmentationEditManager.cpp:520-569`). The barycentric UVs
identify the target grid cell to populate (`SegmentationEditManager.cpp:576-603`).

### Hole Detection & Relaxation

If the chosen grid cell was previously invalid or point-add mode created a new
slot, the manager performs a local flood fill around the seed using the current
"Hole search radius" (default 6) to collect the contiguous hole region
(`SegmentationEditManager.cpp:407-425`, `607-675`). It then iteratively smooths
that patch by averaging each unfixed cell with neighbouring cells and nearby
valid grid samples for the configured number of iterations (default 25) while
keeping the seed pinned to the barycentric projection
(`SegmentationEditManager.cpp:676-748`). This produces a stable local mesh
fragment rather than a lone point floating away from the existing quad surface.
If you prefer to avoid the automatic solve, uncheck "Fill invalid regions" in
the widget and the editor will skip the flood-fill/relaxation pass, requiring an
existing grid cell instead (`SegmentationWidget.cpp:176-207`,
`SegmentationEditManager.cpp:361-520`).


## Handle Influence and Preview Updates

Dragging a handle computes the displacement vector between the original and
current positions. The manager applies the selected falloff profile to nearby
grid cells: the legacy Chebyshev stencil, the geodesic/circular walk, or the
new Row/Column sweep that only nudges neighbours along the active row/column
(`SegmentationEditManager.cpp:910-1085`). Cells with the sentinel value
`(-1,-1,-1)` are skipped so you never accidentally revive intentionally blank
regions. After each pass the preview grid keeps the handle cell in sync
exactly.

Whenever radius or sigma changes, `reapplyAllHandles` replays the stored deltas
over a fresh copy of the original grid to avoid cumulative drift (`SegmentationEditManager.cpp:938-955`).


## Overlay and Viewer Integration

- The overlay controller acquires handle data via `SegmentationEditManager::handles`
  and styles them based on active/hover/keyboard focus states
  (`overlays/SegmentationOverlayController.cpp:62-123`).
- During addition or drag the module sets `_hover`, `_active`, or `_keyboard`
  handles and triggers targeted overlay refreshes to avoid expensive full redraws.
- Handle markers can be culled to a cursor-centred world radius by toggling the
  widget's "Show all handles" option; the overlay enforces the configured
  `handleDisplayDistance` whenever that mode is active.
- Radius feedback is rendered as a transient overlay with text showing the number
  of grid steps represented by the current radius (`SegmentationModule.cpp:759-807`).


## UI and Settings Persistence

`SegmentationWidget` persists user preferences (down-sample, radius, sigma, hole
search radius, hole smoothing iterations, the fill-invalid-regions toggle,
handle visibility mode, handle display distance) into `VC.ini` by calling
`writeSetting` each time a control changes
(`SegmentationWidget.cpp`). The "Hole Filling" and "Handle Display" groups mirror
the parameters the editor and overlay consume for gap repair and visibility
culling. When the widget is constructed it restores those values so sessions
begin with familiar defaults. Pending-change state toggles the Apply button
enabling and the status label (`SegmentationWidget.cpp:304-351`).


## Applying and Saving Changes

- **Apply** – Copies the preview grid back to the base surface, invalidates caches,
  and calls `QuadSurface::saveOverwrite()` inside a try/catch, surfacing errors to
  the status bar (`SegmentationModule.cpp:216-237`). The widget and overlay are
  reset afterwards.
- **Reset** – Restores the preview from the original clone, rebuilds handles, and
  clears manual state (`SegmentationModule.cpp:205-214`,
  `SegmentationEditManager.cpp:192-205`).
- **Stop** – Emits a `stopToolsRequested` signal so upstream UI can exit the
  segmentation workflow gracefully (`SegmentationWidget.cpp:150-163`,
  `SegmentationModule.cpp:215-221`).


## Extending the Editor

Useful extension points:

- **Additional interaction modes** – Introduce new cursor states or gestures by
  extending `SegmentationModule::handleMouse*` and `_pointAddMode`. Keep the overlay
  in sync via `SegmentationOverlayController` helpers.
- **Custom influence kernels** – Modify `applyHandleInfluence` to introduce
  alternate falloff curves or anisotropic behaviour. Remember to adjust
  `_radius`/`_sigma` semantics accordingly.
- **Advanced hole filling** – Adjust the search radius or iterations via
  `SegmentationEditManager::setHoleSearchRadius` /
  `SegmentationEditManager::setHoleSmoothIterations`, or replace the relaxation
  phase entirely. All hooks live inside the edit manager.
- **Per-handle metadata** – Extend `SegmentationEditManager::Handle` if you need to
  annotate handles with provenance flags or custom weights. Ensure
  `regenerateHandles` and serialization logic are updated accordingly.
