# Segmentation Editing Architecture (Gaussian Vertex Mode)
*last update 10/25*
## Purpose and Scope

The segmentation editor lets annotators reshape an existing `QuadSurface` by
pulling its grid vertices with a Gaussian falloff. Editing operates directly on every valid vertex in the surface.

Unless noted otherwise, paths are relative to `apps/VC3D`.


## High-Level Data Flow

1. **Session start** – `SegmentationModule::beginEditingSession` receives the
   active `QuadSurface` from the segmentation tooling and forwards it to
   `SegmentationEditManager::beginSession`. The manager snapshots the original
   point grid, clones a preview surface, and mirrors metadata on the clone
   (`SegmentationEditManager.cpp`).
2. **Vertex edit tracking** – The manager maintains an unordered-map of
   `VertexEdit { row, col, originalWorld, currentWorld, isGrowth }`. Entries are
   added or updated whenever a drag mutates a vertex. No falloff data is cached;
   neighbours are recomputed on demand.
3. **Interaction loop** – The module listens to mouse events from every
   `CVolumeViewer`, resolves the nearest vertex via
   `SegmentationEditManager::worldToGridIndex`, and drives the drag lifecycle:
   * `beginActiveDrag` snapshots the local neighbourhood inside the configured
     radius.
   * `updateActiveDrag` applies the Gaussian falloff to that sample set.
   * `commitActiveDrag` clears temporary caches once the drag ends.
   The module pushes preview updates back to `CSurfaceCollection` so the viewers
   render changes in real time.
4. **Overlay feedback** – `SegmentationOverlayController` now renders a single
   active marker plus the currently affected neighbours (as reported by
   `SegmentationEditManager::recentTouched`). During a drag it projects the
   Gaussian radius into viewer space so users understand the footprint.
5. **Persistence** – "Apply" copies the preview grid back into the base surface
   and clears the edit map. Note that these actions (applying changes and clearing the edit map) also occur immediately after a manipulation of any vertex, so using "Apply" is not typically necessary. "Reset" discards active drags and rebuilds the
   preview from the original snapshot. A session ends by destroying the preview
   clone and clearing overlay state.


## Component Overview

| Component | Responsibility |
| --- | --- |
| `SegmentationWidget` | Minimal Qt sidebar with an editing toggle, Gaussian controls (radius/sigma in grid steps), and Apply/Reset/Stop affordances. Growth volume selection is still exposed for downstream tooling. |
| `SegmentationModule` | Binds the widget, edit manager, overlay, and viewers. Routes input events, maintains drag/hover state, updates overlays, and exposes helper signals (status messages, growth requests). |
| `SegmentationEditManager` | Owns the preview surface, vertex edit map, Gaussian sampling cache for active drags, and dirty tracking. Provides helpers to map world coordinates to grid indices and to query vertex positions. |
| `SegmentationOverlayController` | Draws the active vertex marker, affected neighbours, optional mask overlay points, and the projected Gaussian radius. |
| `QuadSurface` (core layer) | Stores the dense `cv::Mat_<cv::Vec3f>` grid and metadata that represent the segmentation patch. |


## Session Lifecycle

1. **Begin** – `SegmentationModule::beginEditingSession` primes the edit manager
   with the base surface, updates overlay parameters, and publishes the preview
   to the shared surface collection.
2. **Edit** – All vertex mutations occur on the preview grid. The module
   continuously mirrors dirty state back to the widget so the UI can flag
   pending changes.
3. **Apply / Reset** – `SegmentationEditManager::applyPreview` copies the preview
   grid into the base surface and refreshes the original snapshot. `resetPreview`
   clears the edit map and restores the preview from the original snapshot.
4. **End** – `SegmentationModule::endEditingSession` cancels any active drag,
   clears overlay markers, and releases the preview surface.


## Gaussian Falloff Workflow

- Radius and sigma are configured in **grid steps**. The manager converts those
  values to world units using the surface scale so the footprint is circular in
  world space.
- `beginActiveDrag` collects all valid vertices within the radius into
  `ActiveDrag::samples`. Each sample stores a pointer to the base world position
  and its squared distance (already in world units).
- `updateActiveDrag` computes the new world position for the centre vertex and
  applies a Gaussian weight to every sample: `exp(-(d^2)/(2*sigma^2))`. The
  preview grid is mutated in-place, the edit map updated, and the `recentTouched`
  cache refreshed for overlays.
- `commitActiveDrag` clears the temporary sample list. `cancelActiveDrag` reverts
  the affected vertices to their pre-drag positions.


## Interaction Model

### Mouse

- **Left click + drag** – Picks the nearest valid vertex, locks the Gaussian
  neighbourhood, and drags it along the viewer plane. Neighbours update on every
  move and the overlay highlights the active footprint.
- **Left click release** – Commits the drag and leaves the surface in its new
  state. Escape cancels the drag and restores the touched vertices.
- **Mouse wheel** – Adjusts the radius in quarter-step increments while showing
  a transient status message. The overlay radius indicator updates immediately.

Ctrl/Alt modifiers previously reserved for handle management are currently
unused.

### Keyboard

- **Esc** – Cancels the active drag.


## Overlay Behaviour

`SegmentationOverlayController` receives:

- `setActiveVertex(optional<VertexMarker>)` – Active or hovered vertex (row,
  col, world).
- `setTouchedVertices(vector<VertexMarker>)` – The neighbours that were updated
  during the last drag frame.
- `setGaussianParameters(radiusSteps, sigmaSteps, gridStepWorld)` – Used to
  project the radius indicator when an active drag is present.
- Optional mask overlay data for tooling parity.

The controller renders simple filled circles for neighbours, a highlighted circle
for the active vertex, and (during drag) an outline representing the world-space
radius.

